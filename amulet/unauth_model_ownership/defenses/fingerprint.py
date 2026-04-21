"""Fingerprinting implementation"""

import time

import torch
import torch.nn as nn
from scipy.stats import ttest_ind
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .unauth_model_ownership_defense import FingerprintDefense


class DatasetInference(FingerprintDefense):
    """
    Dataset inference fingerprinting: determine if a suspect model was trained on the owner's data.

    Reference:
        Dataset Inference: Ownership Resolution in Machine Learning,
        Pratyush Maini, Mohammad Yaghini, Nicolas Papernot
        ICLR 2021 — https://openreview.net/forum?id=hvdKKV2yt7T

    Attributes:
        target_model: The model to fingerprint.
        suspect_model: The potentially stolen model.
        train_loader: Training data loader used to compute distance features.
        test_loader: Test data loader used to compute distance features.
        num_classes: Number of output classes.
        device: Device used for computation. Example: "cuda:0".
        dataset: Input dimensionality flag. Use "1D" for tabular data, "2D" for images.
        alpha_l1: Step size for L1 attacks.
        alpha_l2: Step size for L2 attacks.
        alpha_linf: Step size for L-infinity attacks.
        k: Hyperparameter for L1 attack sparsity.
        gap: Hyperparameter for L1 attack boundary gap.
        num_iter: Maximum number of MinGD iterations per sample.
        regressor_embed: Whether to use embedding-based regressor. Options: 0 or 1.
        batch_size: Number of samples to collect per loader pass. The DataLoader's
            batch_size should be set to the same value so that one batch covers
            exactly this many samples.
    """

    def __init__(
        self,
        target_model: Module,
        suspect_model: Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_classes: int,
        device: str,
        dataset: str,  # '1D' for tabular, '2D' for images
        alpha_l1: float = 1.0,
        alpha_l2: float = 0.01,
        alpha_linf: float = 0.001,
        k: float = 1.0,
        gap: float = 0.001,
        num_iter: int = 500,
        regressor_embed: int = 0,
        batch_size: int = 256,
    ) -> None:
        super().__init__(target_model, device)
        self.suspect_model = suspect_model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.dataset = dataset

        self.alpha_l1 = alpha_l1
        self.alpha_l2 = alpha_l2
        self.alpha_linf = alpha_linf
        self.k = k
        self.gap = gap
        self.num_iter = num_iter
        self.regressor_embed = regressor_embed
        self.batch_size = batch_size

    def fingerprint(self) -> dict[str, dict[str, float]]:
        """
        Run the dataset inference algorithm and return p-values for both models.

        Returns:
            Nested dict with keys "target" and "suspect", each containing
            "p-value" and "mean_diff".
        """
        train_target, test_target = self.__feature_extracter(self.target_model)
        train_suspect, test_suspect = self.__feature_extracter(self.suspect_model)

        trains = {"target": train_target, "suspect": train_suspect}
        tests = {"target": test_target, "suspect": test_suspect}
        names = ["target", "suspect"]

        mean_distance = trains["target"].mean(dim=(0, 1))
        std_distance = trains["target"].std(dim=(0, 1))

        for name in names:
            trains[name] = trains[name].sort(dim=1)[0]
            tests[name] = tests[name].sort(dim=1)[0]

        for name in names:
            trains[name] = (trains[name] - mean_distance) / std_distance
            tests[name] = (tests[name] - mean_distance) / std_distance

        f_num = 3 * self.num_classes
        a_num = 3 * self.num_classes

        # Derive sample count from the actual collected tensor rather than
        # self.batch_size; the two agree when the DataLoader batch_size matches
        # the constructor batch_size, but using the real shape avoids a cryptic
        # reshape crash when they differ.
        n_samples = trains["target"].shape[0]
        split_index = n_samples // 2

        trains_n = {}
        tests_n = {}
        for name in names:
            trains_n[name] = (
                trains[name].permute(2, 1, 0).reshape(n_samples, f_num)[:, :a_num]
            )
            tests_n[name] = (
                tests[name].permute(2, 1, 0).reshape(n_samples, f_num)[:, :a_num]
            )

        n_ex = split_index
        train = torch.cat((trains_n["target"][:n_ex], tests_n["target"][:n_ex]), dim=0)
        y = torch.cat((torch.zeros(n_ex), torch.ones(n_ex)), dim=0)

        rand = torch.randperm(y.shape[0])
        train = train[rand]
        y = y[rand]

        model = nn.Sequential(
            nn.Linear(a_num, 200), nn.ReLU(), nn.Linear(200, 1), nn.Tanh()
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with tqdm(range(500)) as pbar:
            for _ in pbar:
                optimizer.zero_grad()
                outputs = model(train)
                loss = -1 * ((2 * y - 1) * (outputs.squeeze(-1))).mean()
                loss.backward()
                optimizer.step()
                pbar.set_description(f"loss {loss.item()}")

        outputs_tr = {}
        outputs_te = {}
        for name in names:
            outputs_tr[name] = model(trains_n[name])
            outputs_te[name] = model(tests_n[name])

        for name in names:
            outputs_tr[name] = outputs_tr[name][split_index:]
            outputs_te[name] = outputs_te[name][split_index:]

        results: dict[str, dict[str, float]] = {"target": {}, "suspect": {}}
        for name in names:
            print(f"{name}")
            outputs_train, outputs_test = outputs_tr[name], outputs_te[name]
            m1, m2 = outputs_test[:, 0].mean(), outputs_train[:, 0].mean()

            pred_test = outputs_test[:, 0].detach().cpu().numpy()
            pred_train = outputs_train[:, 0].detach().cpu().numpy()
            pval = ttest_ind(
                pred_test, pred_train, alternative="greater", equal_var=False
            ).pvalue  # type: ignore[reportAttributeAccessIssue] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
            if pval < 0:
                raise ValueError(f"p-value={pval}")

            mean_diff = (m1 - m2).item()

            print(f"p-value = {pval} \t| Mean difference = {mean_diff}")
            results[name]["p-value"] = pval
            results[name]["mean_diff"] = mean_diff

        return results

    def __feature_extracter(self, model: Module) -> tuple[torch.Tensor, torch.Tensor]:
        train_d = self.__get_mingd_vulnerability(self.train_loader, model)
        test_d = self.__get_mingd_vulnerability(self.test_loader, model)
        return train_d, test_d

    def __get_mingd_vulnerability(
        self, loader: DataLoader, model: Module, num_images: int = 100
    ) -> torch.Tensor:
        batch_size = self.batch_size
        max_iter = num_images / batch_size
        lp_dist: list[list[torch.Tensor]] = [[], [], []]
        ex_skipped = 0
        for i, batch in enumerate(loader):
            if self.regressor_embed == 1 and ex_skipped < num_images:
                y = batch[1]
                ex_skipped += y.shape[0]
                continue
            for j, distance in enumerate(["linf", "l2", "l1"]):
                temp_list = []
                for target_i in range(self.num_classes):
                    X, y = batch[0].to(self.device), batch[1].to(self.device)
                    delta = self.__mingd(model, X, distance, target=y * 0 + target_i)
                    _ = model(X + delta)
                    distance_dict = {
                        "linf": self.__norms_linf_squeezed,
                        "l1": self.__norms_l1_squeezed,
                        "l2": self.__norms_l2_squeezed,
                    }
                    distances = distance_dict[distance](delta)
                    temp_list.append(distances.cpu().detach().unsqueeze(-1))
                temp_dist = torch.cat(temp_list, dim=1)
                lp_dist[j].append(temp_dist)
            if i + 1 >= max_iter:
                break
        lp_d = [torch.cat(lp_dist[i], dim=0).unsqueeze(-1) for i in range(3)]
        # full_d shape: (n_samples, num_classes, 3)
        return torch.cat(lp_d, dim=-1)

    def __mingd(
        self, model: Module, X: torch.Tensor, distance: str, target: torch.Tensor
    ) -> torch.Tensor:
        start = time.time()
        is_training = model.training
        model.eval()
        alpha_map = {
            "l1": self.alpha_l1 / self.k,
            "l2": self.alpha_l2,
            "linf": self.alpha_linf,
        }
        alpha = float(alpha_map[distance])

        delta = torch.zeros_like(X, requires_grad=False)

        preds = model(X + delta)
        remaining = preds.max(1)[1] != target
        t = 0
        for t in range(self.num_iter):
            if t > 0:
                preds = model(X[remaining] + delta[remaining])
                new_remaining = preds.max(1)[1] != target[remaining]
                remaining_temp = remaining.clone()
                remaining[remaining_temp] = new_remaining

            if remaining.sum() == 0:
                break

            X_r = X[remaining]
            delta_r = delta[remaining]
            delta_r.requires_grad = True
            preds = model(X_r + delta_r)
            loss = -1 * self.__loss_mingd(preds, target[remaining])
            loss.backward()
            grads = delta_r.grad.detach()  # type: ignore[reportOptionalMemberAccess]
            if distance == "linf":
                delta_r.data += alpha * grads.sign()
            elif distance == "l2":
                if self.dataset == "1D":
                    delta_r.data += alpha * (
                        grads / torch.norm(grads + 1e-12, dim=1).unsqueeze(1)
                    )
                else:
                    delta_r.data += alpha * (grads / self.__norms_l2(grads + 1e-12))
            elif distance == "l1":
                if self.dataset == "1D":
                    delta_r.data += alpha * self.__l1_dir_topk_size2(
                        grads, delta_r.data, X_r, self.gap, self.k
                    )
                else:
                    delta_r.data += alpha * self.__l1_dir_topk(
                        grads, delta_r.data, X_r, self.gap, self.k
                    )
            delta_r.data = torch.min(torch.max(delta_r.detach(), -X_r), 1 - X_r)
            delta_r.grad.zero_()  # type: ignore[reportOptionalMemberAccess]
            delta[remaining] = delta_r.detach()

        print(
            f"Steps = {t + 1} | Failed = {(model(X + delta).max(1)[1] != target).sum().item()} | Time = {time.time() - start:.2f}s"
        )
        if is_training:
            model.train()

        return delta

    @staticmethod
    def __loss_mingd(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        criterion = nn.CrossEntropyLoss()
        loss = criterion(preds, target)
        assert loss >= 0
        return loss

    def __l1_dir_topk(
        self,
        grad: torch.Tensor,
        delta: torch.Tensor,
        X: torch.Tensor,
        gap: float,
        k: float,
    ) -> torch.Tensor:
        X_curr = X + delta
        batch_size = X.shape[0]
        channels = X.shape[1]
        pix = X.shape[2]

        neg1 = (grad < 0) * (X_curr <= gap)
        neg2 = (grad > 0) * (X_curr >= 1 - gap)
        neg3 = X_curr <= 0
        neg4 = X_curr >= 1
        neg = neg1 + neg2 + neg3 + neg4
        u = neg.view(batch_size, 1, -1)
        grad_check = grad.view(batch_size, 1, -1)
        grad_check[u] = 0

        kval = self.__kthlargest(grad_check.abs().float(), k, dim=2)[0].unsqueeze(1)
        k_hot = (grad_check.abs() >= kval).float() * grad_check.sign()
        return k_hot.view(batch_size, channels, pix, pix)

    def __l1_dir_topk_size2(
        self,
        grad: torch.Tensor,
        delta: torch.Tensor,
        X: torch.Tensor,
        gap: float,
        k: float,
    ) -> torch.Tensor:
        X_curr = X + delta
        batch_size = X.shape[0]
        channels = X.shape[1]

        neg1 = (grad < 0) * (X_curr <= gap)
        neg2 = (grad > 0) * (X_curr >= 1 - gap)
        neg3 = X_curr <= 0
        neg4 = X_curr >= 1
        neg = neg1 + neg2 + neg3 + neg4
        u = neg.view(batch_size, 1, -1)
        grad_check = grad.view(batch_size, 1, -1)
        grad_check[u] = 0

        kval = self.__kthlargest(grad_check.abs().float(), k, dim=2)[0].unsqueeze(1)
        k_hot = (grad_check.abs() >= kval).float() * grad_check.sign()
        return k_hot.view(batch_size, channels)

    @staticmethod
    def __kthlargest(
        tensor: torch.Tensor, k: float, dim: int = -1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        val, idx = tensor.topk(int(k), dim=dim)
        return val[:, :, -1], idx[:, :, -1]

    @staticmethod
    def __norms(Z: torch.Tensor) -> torch.Tensor:
        return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None, None]

    def __norms_l2(self, Z: torch.Tensor) -> torch.Tensor:
        return self.__norms(Z)

    def __norms_l2_squeezed(self, Z: torch.Tensor) -> torch.Tensor:
        return self.__norms(Z).squeeze(1).squeeze(1).squeeze(1)

    @staticmethod
    def __norms_l1(Z: torch.Tensor) -> torch.Tensor:
        return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:, None, None, None]

    @staticmethod
    def __norms_l1_squeezed(Z: torch.Tensor) -> torch.Tensor:
        return (
            Z.view(Z.shape[0], -1)
            .abs()
            .sum(dim=1)[:, None, None, None]
            .squeeze(1)
            .squeeze(1)
            .squeeze(1)
        )

    @staticmethod
    def __norms_linf(Z: torch.Tensor) -> torch.Tensor:
        return (
            Z.view(Z.shape[0], -1)
            .abs()
            .max(dim=1)[0]
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

    @staticmethod
    def __norms_linf_squeezed(Z: torch.Tensor) -> torch.Tensor:
        return Z.view(Z.shape[0], -1).abs().max(dim=1)[0]
