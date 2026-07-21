from operator import itemgetter

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics as sk_metrics
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader


class DiscriminatoryBehavior:
    """Evaluate models for discriminatory behavior between data subgroups.

    Assumes that the input data contains features, labels, and sensitive
    attributes.

    Attributes:
        model: The model to evaluate for discriminatory behavior.
        test_loader: Input data used to test the model. Should contain sensitive attributes too.
        device: Device used for inference. Example: "cuda:0".
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: str,
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = device

    @staticmethod
    def accuracy(
        predictions: np.ndarray, targets: np.ndarray, attributes: np.ndarray
    ) -> tuple[float, float]:
        """Compute per-subgroup accuracy split by a binary sensitive attribute.

        Args:
            predictions: Predicted labels.
            targets: Ground-truth labels.
            attributes: Binary sensitive attribute values (0 or 1).

        Returns:
            A tuple (acc_true, acc_false) of accuracy percentages for the
            subgroups where the attribute is 1 and 0, respectively.
        """
        output_attr_true = predictions[np.argwhere(attributes == 1)]
        output_attr_false = predictions[np.argwhere(attributes == 0)]

        target_attr_true = targets[np.argwhere(attributes == 1)]
        target_attr_false = targets[np.argwhere(attributes == 0)]

        acc_true = accuracy_score(target_attr_true, output_attr_true) * 100
        acc_false = accuracy_score(target_attr_false, output_attr_false) * 100

        return acc_true, acc_false

    @staticmethod
    def demographic_parity(predictions: np.ndarray, attributes: np.ndarray) -> float:
        """Compute demographic parity between the two attribute subgroups.

        Measures |P(pred=1|a=0) - P(pred=1|a=1)|.

        Args:
            predictions: Predicted labels.
            attributes: Binary sensitive attribute values (0 or 1).

        Returns:
            The absolute difference in positive-prediction rates between subgroups.
        """
        p_joint_att_positive = np.mean((predictions == 1) * (attributes == 1))
        p_joint_att_negative = np.mean((predictions == 1) * (attributes == 0))

        p_margin_att_positive = np.mean(attributes == 1)
        p_margin_att_negative = np.mean(attributes == 0)

        p_condition_att_positive = p_joint_att_positive / (p_margin_att_positive + 1e-8)
        p_condition_att_negative = p_joint_att_negative / (p_margin_att_negative + 1e-8)
        demo_parity = np.abs(p_condition_att_positive - p_condition_att_negative)

        return demo_parity

    @staticmethod
    def true_positive_parity(
        predictions: np.ndarray, targets: np.ndarray, attributes: np.ndarray
    ) -> float:
        """Compute true-positive-rate parity between the two attribute subgroups.

        Measures |P(y_hat=1|y=1,a=0) - P(y_hat=1|y=1,a=1)|.

        Args:
            predictions: Predicted labels.
            targets: Ground-truth labels.
            attributes: Binary sensitive attribute values (0 or 1).

        Returns:
            The absolute difference in true-positive rates between subgroups.
        """
        p_joint_att_positive = np.mean(
            (predictions == targets) * (targets == 1) * (attributes == 1)
        )
        p_joint_att_negative = np.mean(
            (predictions == targets) * (targets == 1) * (attributes == 0)
        )

        p_margin_att_positive = np.mean((targets == 1) * (attributes == 1))
        p_margin_att_negative = np.mean((targets == 1) * (attributes == 0))

        p_condition_att_positive = p_joint_att_positive / (p_margin_att_positive + 1e-8)
        p_condition_att_negative = p_joint_att_negative / (p_margin_att_negative + 1e-8)

        tp_parity = np.abs(p_condition_att_positive - p_condition_att_negative)

        return tp_parity

    @staticmethod
    def false_positive_parity(
        predictions: np.ndarray, targets: np.ndarray, attributes: np.ndarray
    ) -> float:
        """Compute false-positive-rate parity between the two attribute subgroups.

        Measures |P(y_hat=1|y=0,a=0) - P(y_hat=1|y=0,a=1)|.

        Args:
            predictions: Predicted labels.
            targets: Ground-truth labels.
            attributes: Binary sensitive attribute values (0 or 1).

        Returns:
            The absolute difference in false-positive rates between subgroups.
        """
        p_joint_att_positive = np.mean(
            (predictions != targets) * (targets == 0) * (attributes == 1)
        )
        p_joint_att_negative = np.mean(
            (predictions != targets) * (targets == 0) * (attributes == 0)
        )

        p_margin_att_positive = np.mean((targets == 0) * (attributes == 1))
        p_margin_att_negative = np.mean((targets == 0) * (attributes == 0))

        p_condition_att_positive = p_joint_att_positive / (p_margin_att_positive + 1e-8)
        p_condition_att_negative = p_joint_att_negative / (p_margin_att_negative + 1e-8)
        fp_parity = np.abs(p_condition_att_positive - p_condition_att_negative)

        return fp_parity

    @staticmethod
    def p_rule(predictions: np.ndarray, attributes: np.ndarray) -> float:
        """Compute the p% rule between the two attribute subgroups.

        Args:
            predictions: Predicted labels.
            attributes: Binary sensitive attribute values (0 or 1).

        Returns:
            The smaller of the two positive-rate odds ratios, scaled by 100.
            Returns 0.0 when the reference subgroup has no positive predictions.
        """
        y_z_1 = predictions[attributes == 1].mean()
        y_z_0 = predictions[attributes == 0].mean()
        if y_z_0 == 0:
            return 0.0
        odds = y_z_1 / y_z_0
        return float(np.min([odds, 1 / odds]) * 100)

    def evaluate_subgroup_metrics(self):
        """
        Calculate fairness metrics for each sensitive attribute subgroup.

        Returns:
            Nested dict mapping attribute index to a dict with keys
            "acc_true", "acc_false", "demographic_parity", "true_positive_parity",
            "false_positive_parity", "p_rule", and "equalized_odds".
        """
        self.model.eval()
        predictions = []
        targets = []
        attributes = []
        with torch.no_grad():
            for data, target, sensitive_attributes in self.test_loader:
                data, target, sensitive_attributes = (
                    data.to(self.device),
                    target.to(self.device),
                    sensitive_attributes.to(self.device),
                )

                outputs = self.model(data)
                _, pred = torch.max(outputs, 1)
                predictions.append(pred.data.cpu().numpy())
                targets.append(target.data.cpu().numpy())
                attributes.append(sensitive_attributes.data.cpu().numpy())

            # Concatenate the per-batch arrays directly. A dtype=object wrap
            # leaves the result object-typed when batches are uniform-length or
            # there is a single batch, which accuracy_score rejects ("unknown is
            # not supported"); plain concatenate yields int64 for every shape.
            attributes = np.concatenate(attributes)
            predictions = np.concatenate(predictions)
            targets = np.concatenate(targets)

        num_attributes = attributes.shape[1]
        metrics = {}

        for i in range(num_attributes):
            metrics[i] = {}
            attribute_arr = attributes[:, i]

            metrics[i]["acc_true"], metrics[i]["acc_false"] = self.accuracy(
                predictions, targets, attribute_arr
            )
            metrics[i]["demographic_parity"] = self.demographic_parity(
                predictions, attribute_arr
            )
            metrics[i]["true_positive_parity"] = self.true_positive_parity(
                predictions, targets, attribute_arr
            )
            metrics[i]["false_positive_parity"] = self.false_positive_parity(
                predictions, targets, attribute_arr
            )
            metrics[i]["p_rule"] = self.p_rule(predictions, attribute_arr)
            metrics[i]["equalized_odds"] = (
                metrics[i]["true_positive_parity"] + metrics[i]["false_positive_parity"]
            )

        return metrics

    @staticmethod
    def adversary_auc(
        discriminator: nn.Module,
        main_model: nn.Module,
        test_loader: DataLoader,
        device: str,
    ) -> list[float]:
        """
        Measure how well the adversary discriminator infers sensitive attributes after debiasing.

        Args:
            discriminator: The adversary network from AdversarialDebiasing.
            main_model: The debiased main model.
            test_loader: DataLoader yielding (X, y, Z) batches.
            device: Device string. Example: "cuda:0".

        Returns:
            Best-threshold AUC per sensitive attribute.
        """
        discriminator.eval()
        main_model.eval()
        n_attrs = None
        preds: list[list[float]] = []
        trues: list[list[int]] = []

        with torch.no_grad():
            for X_batch, _, Z_batch in test_loader:
                X_batch = X_batch.to(device)
                Z_batch = Z_batch.to(device)
                z_pred = discriminator(main_model(X_batch)).cpu().numpy()
                z_true = Z_batch.cpu().numpy()
                if n_attrs is None:
                    n_attrs = z_pred.shape[1]
                    preds = [[] for _ in range(n_attrs)]
                    trues = [[] for _ in range(n_attrs)]
                for i in range(n_attrs):
                    preds[i].extend(z_pred[:, i].tolist())
                    trues[i].extend(int(v) for v in z_true[:, i].tolist())

        if n_attrs is None:
            return []

        def _best_threshold_auc(
            pred: list[float], true: list[int]
        ) -> tuple[float, float]:
            candidates = [
                (
                    t,
                    float(
                        sk_metrics.roc_auc_score(
                            true, [1 if v > t else 0 for v in pred]
                        )
                    ),
                )
                for t in np.arange(0.0, 1.0, 0.01)
            ]
            return max(candidates, key=itemgetter(1))

        aucs = []
        for i in range(n_attrs):
            t, _ = _best_threshold_auc(preds[i], trues[i])
            thresholded = [1 if v > t else 0 for v in preds[i]]
            auc = float(sk_metrics.roc_auc_score(trues[i], thresholded))
            print(f"Adversary AUC [attr {i}]: {auc:.4f}")
            aucs.append(auc)
        return aucs
