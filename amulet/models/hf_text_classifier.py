"""LoRA sequence-classifier wrapper around a decoder LLM."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .base import AmuletModel

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

# The plan's license-free fallback victim (Llama architecture, not gated).
_DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Llama attention projections that LoRA adapts, and the SEQ_CLS head kept trainable.
_DEFAULT_TARGET_MODULES = ["q_proj", "v_proj"]
_HEAD_MODULE = "score"

_LLM_INSTALL_HINT = (
    "HFTextClassifier requires the optional LLM stack. Install it with "
    "`pip install amuletml[llm]` (or `uv sync --extra llm`)."
)


class HFTextClassifier(AmuletModel):
    """Decoder LLM used as a LoRA sequence classifier, wired for the Amulet pipeline.

    Wraps ``AutoModelForSequenceClassification`` with a PEFT LoRA config: the base is
    frozen and only the LoRA adapters and the classification head (``modules_to_save=
    ["score"]``) train. A LoRA-only optimizer would leave the head at random init, so
    the head must stay trainable for the classifier to learn.

    The ``forward`` signature is deliberately narrow so the existing single-tensor
    training loops drive it unchanged: it takes one padded ``input_ids`` tensor,
    derives the attention mask internally, and returns the raw logits **tensor** (not
    the ``SequenceClassifierOutput`` object). That is exactly what
    ``DPSGD.train_private`` (``output = model(data)``; ``torch.max(output, 1)``) and
    ``get_accuracy`` expect.

    Under differential privacy the trainable params must be fp32: bf16 grad samples
    break Opacus per-sample clipping (fp32 clip factor x bf16 grad in
    ``torch.tensordot``). Pass ``dtype=torch.float32`` for the DP run; bf16 is fine
    otherwise. The optional 4-bit load path is off by default and must never be used
    under DP (Opacus hooks do not compose with bitsandbytes 4-bit layers).

    Attributes:
        model: The wrapped PEFT model.
        pad_id: Padding token id used to build the attention mask.
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
        """Build the LoRA sequence classifier.

        Args:
            model_name: Hub id of the base decoder LLM to fine-tune.
            num_labels: Number of output classes.
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
            pad_token_id: Explicit padding token id. Defaults to the tokenizer's pad
                token (or its eos token) for the pretrained path, or the config's
                ``pad_token_id`` for the random-init path.

        Raises:
            ImportError: If the optional ``llm`` extra is not installed, or 4-bit is
                requested without bitsandbytes.
        """
        super().__init__()
        try:
            from peft import LoraConfig, TaskType, get_peft_model
            from transformers import AutoModelForSequenceClassification
        except ImportError as exc:
            raise ImportError(_LLM_INSTALL_HINT) from exc

        if target_modules is None:
            target_modules = list(_DEFAULT_TARGET_MODULES)

        base, resolved_pad = self._build_base(
            AutoModelForSequenceClassification,
            model_name=model_name,
            num_labels=num_labels,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            config=config,
            pad_token_id=pad_token_id,
        )

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            modules_to_save=[_HEAD_MODULE],
        )
        self.model = get_peft_model(base, lora_config)  # pyright: ignore[reportUnknownMemberType]
        self.pad_id = resolved_pad

    @staticmethod
    def _build_base(
        model_cls: type,
        model_name: str,
        num_labels: int,
        dtype: torch.dtype,
        load_in_4bit: bool,
        config: PretrainedConfig | None,
        pad_token_id: int | None,
    ) -> tuple[PreTrainedModel, int]:
        """Construct the base sequence-classification model and resolve its pad id."""
        if config is not None:
            # Random-init path (fast tests): no download, no tokenizer.
            base = model_cls.from_config(config)  # pyright: ignore[reportUnknownMemberType]
            resolved_pad = (
                pad_token_id
                if pad_token_id is not None
                else config.pad_token_id
                if config.pad_token_id is not None
                else 0
            )
        else:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_name)  # pyright: ignore[reportUnknownMemberType]
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            resolved_pad = (
                pad_token_id if pad_token_id is not None else tokenizer.pad_token_id
            )
            kwargs = {"num_labels": num_labels, "dtype": dtype}
            if load_in_4bit:
                kwargs["quantization_config"] = HFTextClassifier._bnb_4bit_config()
            base = model_cls.from_pretrained(model_name, **kwargs)  # pyright: ignore[reportUnknownMemberType]

        base.config.pad_token_id = resolved_pad
        return base, int(resolved_pad)

    @staticmethod
    def _bnb_4bit_config():
        """Build a bitsandbytes 4-bit quantization config, guarding the import.

        bitsandbytes is GPU/Linux-only and deliberately outside the ``llm`` extra, so
        its absence raises a clear message rather than a bare ImportError.
        """
        try:
            import bitsandbytes  # noqa: F401  # pyright: ignore[reportMissingImports]
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "4-bit loading requires bitsandbytes (GPU/Linux only), which is not "
                "part of the amuletml[llm] extra. Install it separately, and never "
                "use the 4-bit path under differential privacy."
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
            x: A ``(batch, seq)`` LongTensor of padded ``input_ids``. Named ``x`` to
                match the ``AmuletModel`` single-tensor contract; callers pass it
                positionally, exactly like ``DPSGD.train_private`` and ``get_accuracy``.

        Returns:
            A ``(batch, num_labels)`` logits tensor (not a ``SequenceClassifierOutput``).
        """
        attention_mask = self._attention_mask(x)
        return self.model(input_ids=x, attention_mask=attention_mask).logits

    def get_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled last hidden state at each sequence's final real token.

        Args:
            x: A ``(batch, seq)`` LongTensor of padded ``input_ids``.

        Returns:
            A ``(batch, hidden_size)`` tensor pooled from the last non-pad position.
        """
        attention_mask = self._attention_mask(x)
        outputs = self.model(
            input_ids=x,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_index = attention_mask.sum(dim=1) - 1
        hidden = outputs.hidden_states[-1]
        return hidden[torch.arange(hidden.size(0), device=hidden.device), last_index]

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Return every trainable parameter (LoRA adapters plus the classification head).

        Build the optimizer over these: a LoRA-only optimizer would leave the ``score``
        head at random init and the classifier would never learn.
        """
        return [p for p in self.parameters() if p.requires_grad]
