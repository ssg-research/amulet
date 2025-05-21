"""Fingerprinting implementation"""

import time

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.nn import Module
from torch.utils.data import DataLoader
from scipy.stats import ttest_ind


class DatasetInference:
    """
    Implementation of algorithm to fingerprint models based on the
    data they are trained with. Code taken from:
    https://github.com/cleverhans-lab/dataset-inference/blob/main/src/generate_features.py
    Reference:
        A Privacy-Friendly Approach to Data Valuation,
        Jiachen T. Wang, Yuqing Zhu, Yu-Xiang Wang, Ruoxi Jia, Prateek Mittal
        Thirty-seventh Conference on Neural Information Processing Systems, 2023
        https://openreview.net/forum?id=FAZ3i0hvm0

    Attributes:
        target_model: :class:`~torch.nn.Module`
            The model to fingerprint.
        suspect_model: :class:`~torch.nn.Module`
            The potentially stolen model.
        train_loader: :class:`~torch.utils.data.DataLoader`
            Training data loader to train model.
        test_loader: :class:`~torch.utils.data.DataLoader`
            Test data loader to calculate Shapley values.
        num_classes: int
            Number of classes in the dataset
        device: str
            Device used to train model. Example: "cuda:0".
        alpha_l1: float
            Step size for L1 attacks.
        alpha_l2: float
            Step size for L2 attacks
        alpha_linf: float
            Step size for L_infinity attacks.
        k: float
            Hyperparameter for L1 attack.
        gap: float
            Hyperparameter for L1 attack.
        regression_embd: int
            Target embeddings for training regressor. Options: [0, 1]
        batch_size: int
            Batch size of training data.
    """

    def __init__(
        self,
        target_model: Module,
        suspect_model: Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_classes: int,
        device: str,
        dataset: str,  # ['1D' or '2D']
        alpha_l1: float = 1.0,
        alpha_l2: float = 0.01,
        alpha_linf: float = 0.001,
        k: float = 1.0,
        gap: float = 0.001,
        num_iter: int = 500,
        regressor_embed: int = 0,
        batch_size: int = 256,
    ) -> None:
        self.target_model = target_model
        self.suspect_model = suspect_model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.device = device
        self.dataset = dataset

        self.alpha_l1 = alpha_l1
        self.alpha_l2 = alpha_l2
        self.alpha_linf = alpha_linf
        self.k = k
        self.gap = gap
        self.num_iter = num_iter
        self.regressor_embed = regressor_embed
        self.batch_size = batch_size

    def fingerprint(self):
        """
        Runs the Dataset Inference algorithm and returns a p-value for
        the target and suspect models.
        """
        train_target, test_target = self.__feature_extracter(self.target_model)
        train_suspect, test_suspect = self.__feature_extracter(self.suspect_model)
        split_index = int(self.batch_size / 2)

        trains = {}
        tests = {}
        names = ["target", "suspect"]
        trains["target"] = train_target
        trains["suspect"] = train_suspect
        tests["target"] = test_target
        tests["suspect"] = test_suspect

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

        trains_n = {}
        tests_n = {}
        for name in names:
            trains_n[name] = trains[name].T.reshape(self.batch_size, f_num)[:, :a_num]
            tests_n[name] = tests[name].T.reshape(self.batch_size, f_num)[:, :a_num]

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
                inputs = train
                outputs = model(inputs)
                loss = -1 * ((2 * y - 1) * (outputs.squeeze(-1))).mean()
                loss.backward()
                optimizer.step()
                pbar.set_description("loss {}".format(loss.item()))

        outputs_tr = {}
        outputs_te = {}
        for name in names:
            outputs_tr[name] = model(trains_n[name])
            outputs_te[name] = model(tests_n[name])

        for name in names:
            outputs_tr[name], outputs_te[name] = (
                outputs_tr[name][split_index:],
                outputs_te[name][split_index:],
            )

        results = {"target": {}, "suspect": {}}
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
                raise Exception(f"p-value={pval}")

            mean_diff = m1 - m2
            mean_diff = mean_diff.item()

            print(f"p-value = {pval} \t| Mean difference = {mean_diff}")
            results[name]["p-value"] = pval
            results[name]["mean_diff"] = mean_diff

        return results

    @staticmethod
    def print_inference(outputs_train, outputs_test):
        m1, m2 = outputs_test[:, 0].mean(), outputs_train[:, 0].mean()

        pred_test = outputs_test[:, 0].detach().cpu().numpy()
        pred_train = outputs_train[:, 0].detach().cpu().numpy()
        pval = ttest_ind(
            pred_test, pred_train, alternative="greater", equal_var=False
        ).pvalue  # type: ignore[reportAttributeAccessIssue] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
        if pval < 0:
            raise Exception(f"p-value={pval}")
        print(f"p-value = {pval} \t| Mean difference = {m1-m2}")

        return pval, m1 - m2

    def __feature_extracter(self, model):
        train_d = self.__get_mingd_vulnerability(self.train_loader, model)
        test_d = self.__get_mingd_vulnerability(self.test_loader, model)
        return train_d, test_d

    def __get_mingd_vulnerability(self, loader, model, num_images=100):
        batch_size = self.batch_size
        max_iter = num_images / batch_size
        lp_dist = [[], [], []]
        ex_skipped = 0
        for i, batch in enumerate(loader):
            if (
                self.regressor_embed == 1
            ):  ##We need an extra set of `distinct images for training the confidence regressor
                if ex_skipped < num_images:
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
                # temp_dist = [batch_size, num_classes)]
                temp_dist = torch.cat(temp_list, dim=1)
                lp_dist[j].append(temp_dist)
            if i + 1 >= max_iter:
                break
        # lp_d is a list of size three with each element being a tensor of shape [num_images,num_classes]
        lp_d = [torch.cat(lp_dist[i], dim=0).unsqueeze(-1) for i in range(3)]
        # full_d = [num_images, num_classes, num_attacks]
        full_d = torch.cat(lp_d, dim=-1)

        return full_d

    def __mingd(self, model, X, distance, target):
        start = time.time()
        is_training = model.training
        model.eval()  # Need to freeze the batch norm and dropouts
        alpha_map = {
            "l1": self.alpha_l1 / self.k,
            "l2": self.alpha_l2,
            "linf": self.alpha_linf,
        }
        alpha = float(alpha_map[distance])

        delta = torch.zeros_like(X, requires_grad=False)
        loss = 0

        # TODO: Adding this to deal with ruff, fix this later.
        preds = model(X + delta)
        remaining = preds.max(1)[1] != target
        X_r = X[remaining]
        delta_r = delta[remaining]
        t = 0
        for t in range(self.num_iter):
            if t > 0:
                preds = model(X_r + delta_r)
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
            # print(t, loss, remaining.sum().item())
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
            delta_r.data = torch.min(
                torch.max(delta_r.detach(), -X_r), 1 - X_r
            )  # clip X+delta_r[remaining] to [0,1]
            delta_r.grad.zero_()  # type: ignore[reportOptionalMemberAccess]
            delta[remaining] = delta_r.detach()

        print(
            f"Number of steps = {t+1} | Failed to convert = {(model(X+delta).max(1)[1]!=target).sum().item()} | Time taken = {time.time() - start}"
        )
        if is_training:
            model.train()

        return delta

    @staticmethod
    def __loss_mingd(preds, target):
        # loss =  (preds.max(dim = 1)[0] - preds[torch.arange(preds.shape[0]),target]).mean()
        criterion = nn.CrossEntropyLoss()
        loss = criterion(preds, target)
        assert loss >= 0
        return loss

    def __l1_dir_topk(self, grad, delta, X, gap, k):
        # Check which all directions can still be increased such that
        # they haven't been clipped already and have scope of increasing
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

    def __l1_dir_topk_size2(self, grad, delta, X, gap, k):
        # Check which all directions can still be increased such that
        # they haven't been clipped already and have scope of increasing
        # ipdb.set_trace()
        X_curr = X + delta
        batch_size = X.shape[0]
        channels = X.shape[1]
        # pix = X.shape[2]
        # print (batch_size)
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
    def __kthlargest(tensor, k, dim=-1):
        val, idx = tensor.topk(k, dim=dim)
        return val[:, :, -1], idx[:, :, -1]

    @staticmethod
    def __norms(Z):
        return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None, None]

    def __norms_l2(self, Z):
        return self.__norms(Z)

    def __norms_l2_squeezed(self, Z):
        return self.__norms(Z).squeeze(1).squeeze(1).squeeze(1)

    @staticmethod
    def __norms_l1(Z):
        return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:, None, None, None]

    @staticmethod
    def __norms_l1_squeezed(Z):
        return (
            Z.view(Z.shape[0], -1)
            .abs()
            .sum(dim=1)[:, None, None, None]
            .squeeze(1)
            .squeeze(1)
            .squeeze(1)
        )

    @staticmethod
    def __norms_l0(Z):
        return ((Z.view(Z.shape[0], -1) != 0).sum(dim=1)[:, None, None, None]).float()

    @staticmethod
    def __norms_l0_squeezed(Z):
        return (
            ((Z.view(Z.shape[0], -1) != 0).sum(dim=1)[:, None, None, None])
            .float()
            .squeeze(1)
            .squeeze(1)
            .squeeze(1)
        )

    @staticmethod
    def __norms_linf(Z):
        return (
            Z.view(Z.shape[0], -1)
            .abs()
            .max(dim=1)[0]
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

    @staticmethod
    def __norms_linf_squeezed(Z):
        return Z.view(Z.shape[0], -1).abs().max(dim=1)[0]
