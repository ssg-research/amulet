"""LoRA-adapted HuggingFace causal-LM victim, usable as classifier, scorer, or generator."""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AmuletModel

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

# The plan's license-free fallback victim (Llama architecture, not gated).
_DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Llama attention projections that LoRA adapts.
_DEFAULT_TARGET_MODULES = ["q_proj", "v_proj"]

_LLM_INSTALL_HINT = (
    "HFCausalLM requires the optional LLM stack. Install it with "
    "`pip install amuletml[llm]` (or `uv sync --extra llm`)."
)


class HFCausalLM(AmuletModel):
    """A HuggingFace causal (decoder-only) LM adapted with LoRA for three roles.

    One shared, LoRA-adapted decoder backs all three capabilities, so the same object is
    the classification victim, the perplexity scorer a defense like ONION consumes, and a
    text generator:

    - **Classify** — ``forward`` pools the last real-token hidden state and passes it
      through a trainable classification head, returning the bare ``(batch, num_labels)``
      logits **tensor** (not a ``SequenceClassifierOutput``). That is exactly what the
      single-tensor loops (``train_classifier``, ``DPSGD.train_private``) and
      ``get_accuracy`` expect, so they drive it unchanged.
    - **Score** — ``perplexity`` uses the pretrained LM head. By default it scores the
      clean base (LoRA adapters disabled), so a defense gets a reference LM unperturbed by
      any poisoned fine-tuning; ``use_adapter=True`` scores through the adapters (the
      actual fine-tuned model).
    - **Generate** — ``generate`` delegates to the base LM's decoding.

    This works for **causal / decoder-only** LMs (Llama, GPT-2, Mistral, Mixtral-style
    MoE decoders): anything that loads as ``AutoModelForCausalLM``. Encoder-only models
    (BERT) and seq2seq models (T5) do not fit and are out of scope.

    Under differential privacy the trainable params must be fp32: bf16 grad samples break
    Opacus per-sample clipping. Pass ``dtype=torch.float32`` for the DP run; bf16 is fine
    otherwise. The optional 4-bit load path is off by default and must never be used under
    DP (Opacus hooks do not compose with bitsandbytes 4-bit layers).

    Attributes:
        lm: The PEFT-wrapped causal LM (frozen base + LoRA adapters).
        classifier: The trainable linear classification head over the pooled hidden state.
        pad_id: Padding token id used to build attention masks.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        num_labels: int = 2,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: list[str] | None = None,
        dtype: torch.dtype = torch.bfloat16,
        load_in_4bit: bool = False,
        config: PretrainedConfig | None = None,
        pad_token_id: int | None = None,
    ):
        """Build the LoRA-adapted causal LM with a classification head.

        Args:
            model_name: Hub id of the base decoder LLM to adapt.
            num_labels: Number of classification output classes.
            lora_r: LoRA rank.
            lora_alpha: LoRA scaling factor.
            lora_dropout: Dropout applied inside the LoRA layers.
            target_modules: Module names LoRA adapts; defaults to the Llama attention
                projections ``["q_proj", "v_proj"]``.
            dtype: Parameter dtype of the base model. Use ``torch.float32`` for the DP
                run; ``torch.bfloat16`` is fine otherwise.
            load_in_4bit: Load the frozen base in 4-bit (bitsandbytes, GPU/Linux only).
                Off by default; never valid under DP.
            config: Optional Hugging Face config for a random-init base (used by fast
                tests). When given, it overrides ``model_name`` and no weights or
                tokenizer are downloaded.
            pad_token_id: Explicit padding token id. Defaults to the tokenizer's pad token
                (or its eos token) for the pretrained path, or the config's
                ``pad_token_id`` for the random-init path.

        Raises:
            ImportError: If the optional ``llm`` extra is not installed, or 4-bit is
                requested without bitsandbytes.
        """
        super().__init__()
        try:
            from peft import LoraConfig, TaskType, get_peft_model
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise ImportError(_LLM_INSTALL_HINT) from exc

        if target_modules is None:
            target_modules = list(_DEFAULT_TARGET_MODULES)

        base, resolved_pad = self._build_base(
            AutoModelForCausalLM,
            model_name=model_name,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            config=config,
            pad_token_id=pad_token_id,
        )
        base_dtype = base.dtype
        hidden_size = int(base.config.hidden_size)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
        )
        self.lm = get_peft_model(base, lora_config)  # pyright: ignore[reportUnknownMemberType]
        # Classification head lives outside PEFT so it is trainable by default (a LoRA-only
        # optimizer would leave a SEQ_CLS head at random init). Match the base dtype so the
        # pooled hidden state feeds it without a cast.
        self.classifier = nn.Linear(hidden_size, num_labels, dtype=base_dtype)
        self.pad_id = resolved_pad

    @staticmethod
    def _build_base(
        model_cls: type,
        model_name: str,
        dtype: torch.dtype,
        load_in_4bit: bool,
        config: PretrainedConfig | None,
        pad_token_id: int | None,
    ) -> tuple[PreTrainedModel, int]:
        """Construct the base causal LM and resolve its pad id."""
        auto_model: Any = model_cls
        if config is not None:
            # Random-init path (fast tests): no download, no tokenizer.
            base = auto_model.from_config(config)
            resolved_pad = (
                pad_token_id
                if pad_token_id is not None
                else config.pad_token_id
                if config.pad_token_id is not None
                else 0
            )
        else:
            from transformers import AutoTokenizer

            auto_tokenizer: Any = AutoTokenizer
            tokenizer = auto_tokenizer.from_pretrained(model_name)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            resolved_pad = (
                pad_token_id if pad_token_id is not None else tokenizer.pad_token_id
            )
            kwargs: dict[str, Any] = {"dtype": dtype}
            if load_in_4bit:
                kwargs["quantization_config"] = HFCausalLM._bnb_4bit_config()
            base = auto_model.from_pretrained(model_name, **kwargs)

        base.config.pad_token_id = resolved_pad
        return base, int(resolved_pad)

    @staticmethod
    def _bnb_4bit_config():
        """Build a bitsandbytes 4-bit quantization config, guarding the import.

        bitsandbytes is GPU/Linux-only and deliberately outside the ``llm`` extra, so its
        absence raises a clear message rather than a bare ImportError.
        """
        try:
            import bitsandbytes  # noqa: F401  # pyright: ignore[reportMissingImports]
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "4-bit loading requires bitsandbytes (GPU/Linux only), which is not part "
                "of the amuletml[llm] extra. Install it separately, and never use the "
                "4-bit path under differential privacy."
            ) from exc
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    def _attention_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Derive a ``(batch, seq)`` attention mask from the padded ``input_ids``."""
        return (x != self.pad_id).long()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a batch of padded token ids.

        Args:
            x: A ``(batch, seq)`` LongTensor of padded ``input_ids``. Named ``x`` to match
                the ``AmuletModel`` single-tensor contract; callers pass it positionally.

        Returns:
            A ``(batch, num_labels)`` logits tensor (not a ``SequenceClassifierOutput``).
        """
        return self.classifier(self.get_hidden(x))

    def get_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled last hidden state at each sequence's final real token.

        Args:
            x: A ``(batch, seq)`` LongTensor of padded ``input_ids``.

        Returns:
            A ``(batch, hidden_size)`` tensor pooled from the last non-pad position of the
            LoRA-adapted decoder.
        """
        attention_mask = self._attention_mask(x)
        outputs = self.lm(
            input_ids=x,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_index = attention_mask.sum(dim=1) - 1
        hidden = outputs.hidden_states[-1]
        return hidden[torch.arange(hidden.size(0), device=hidden.device), last_index]

    def perplexity(self, input_ids: torch.Tensor, use_adapter: bool = False) -> float:
        """Compute the causal LM's perplexity of a single token sequence.

        Args:
            input_ids: A ``(1, seq)`` or ``(seq,)`` LongTensor of token ids for one
                (unpadded) sequence.
            use_adapter: When ``False`` (default) the LoRA adapters are disabled and the
                clean pretrained base scores — a reference LM unperturbed by fine-tuning,
                matching canonical ONION. When ``True`` the adapters stay on, scoring
                through the actual fine-tuned model.

        Returns:
            The perplexity as a float, or ``inf`` for sequences too short to score (fewer
            than two tokens), so a one-token candidate never looks like a fluent outlier.
        """
        ids = input_ids if input_ids.dim() == 2 else input_ids.unsqueeze(0)
        if ids.size(1) < 2:
            return float("inf")
        ids = ids.to(next(self.parameters()).device)
        adapter_ctx = nullcontext() if use_adapter else self.lm.disable_adapter()
        with torch.no_grad(), adapter_ctx:
            loss = self.lm(input_ids=ids, labels=ids).loss
        return math.exp(loss.item())

    def perplexity_batch(
        self,
        sequences: list[torch.Tensor],
        use_adapter: bool = False,
        batch_size: int = 32,
    ) -> list[float]:
        """Per-sequence perplexity for many token sequences in few padded forwards.

        Equivalent to calling ``perplexity`` on each sequence — the same masked mean token
        cross-entropy, and ``inf`` for sub-two-token sequences — but scores up to
        ``batch_size`` sequences per forward instead of one. This is ONION's hot path: it
        replaces a forward per leave-one-out candidate with a single batched forward.

        Sequences are **right-padded** so every real token keeps its original position;
        under the causal mask a real token then attends to exactly the same tokens it would
        unpadded, so its logits — and the resulting per-sequence perplexity — are unchanged
        by batching (up to float reduction-order noise). Per-sequence loss is computed here
        as the masked mean token cross-entropy, never ``outputs.loss`` (which would average
        over the whole batch and corrupt every score).

        Args:
            sequences: Token-id tensors, each ``(seq,)`` or ``(1, seq)``, of possibly
                differing lengths.
            use_adapter: Score through the LoRA adapters (the fine-tuned model) rather than
                the clean pretrained base. Defaults to the clean base, matching
                ``perplexity``.
            batch_size: Maximum sequences per padded forward. Tune down if memory is tight.

        Returns:
            A perplexity per input sequence, in input order; ``inf`` where a sequence has
            fewer than two tokens.
        """
        flat = [s.squeeze(0) if s.dim() == 2 else s for s in sequences]
        results = [float("inf")] * len(flat)
        scorable = [(i, s) for i, s in enumerate(flat) if s.numel() >= 2]

        device = next(self.parameters()).device
        adapter_ctx = nullcontext() if use_adapter else self.lm.disable_adapter()
        with torch.no_grad(), adapter_ctx:
            for start in range(0, len(scorable), batch_size):
                chunk = scorable[start : start + batch_size]
                lengths = [int(s.numel()) for _, s in chunk]
                max_len = max(lengths)
                padded = torch.full(
                    (len(chunk), max_len), self.pad_id, dtype=torch.long, device=device
                )
                mask = torch.zeros(
                    (len(chunk), max_len), dtype=torch.long, device=device
                )
                for row, ((_, seq), n) in enumerate(zip(chunk, lengths, strict=True)):
                    padded[row, :n] = seq.to(device)
                    mask[row, :n] = 1
                logits = self.lm(input_ids=padded, attention_mask=mask).logits
                # Causal shift by one; drop positions whose target token is padding.
                shift_logits = logits[:, :-1, :]
                shift_labels = padded[:, 1:].clone()
                keep = mask[:, 1:].bool()
                shift_labels[~keep] = -100
                tok_loss = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view(shift_labels.shape)
                per_seq = tok_loss.sum(dim=1) / keep.sum(dim=1)
                for row, (orig_i, _) in enumerate(chunk):
                    results[orig_i] = math.exp(per_seq[row].item())
        return results

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Autoregressively generate a continuation of ``input_ids``.

        Args:
            input_ids: A ``(batch, seq)`` LongTensor prompt.
            **kwargs: Forwarded to the base LM's ``generate`` (e.g. ``max_new_tokens``).
                Defaults to greedy decoding.

        Returns:
            A ``(batch, seq + generated)`` LongTensor with the prompt preserved as a prefix.
        """
        kwargs.setdefault("do_sample", False)
        kwargs.setdefault("pad_token_id", self.pad_id)
        attention_mask = self._attention_mask(input_ids)
        return self.lm.generate(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Return every trainable parameter (LoRA adapters plus the classification head).

        Build the optimizer over these: the pretrained decoder and LM head stay frozen,
        and a LoRA-only optimizer would leave the classification head at random init.
        """
        return [p for p in self.parameters() if p.requires_grad]
