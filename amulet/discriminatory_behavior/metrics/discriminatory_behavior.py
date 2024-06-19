import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


class DiscriminatoryBehavior:
    """
    Implementation of algorithm to evaluate models for discriminatory
    behavior between different subgroups of data. Assumes that the input
    data contains features, labels, and sensitive attributes.

    Attributes:
        target_model: :class:`~torch.nn.Module`
            This model will be extracted.
        test_loader: :class:`~torch.utils.data.DataLoader`
            Input data used to test the model.
            Should contain sensitive attributes too.
        device: str
            Device used to train model. Example: "cuda:0".
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
        output_attr_true = predictions[np.argwhere(attributes == 1)]
        output_attr_false = predictions[np.argwhere(attributes == 0)]

        target_attr_true = targets[np.argwhere(attributes == 1)]
        target_attr_false = targets[np.argwhere(attributes == 0)]

        acc_true = accuracy_score(target_attr_true, output_attr_true) * 100
        acc_false = accuracy_score(target_attr_false, output_attr_false) * 100

        return acc_true, acc_false

    @staticmethod
    def demographic_parity(predictions: np.ndarray, attributes: np.ndarray) -> float:
        """
        |P(pred=1|a=0) - P(pred=1|a=1)|
        """
        p_joint_att_positive = np.mean((predictions == 1) * (attributes == 1))
        p_joint_att_negative = np.mean((predictions == 1) * (attributes == 0))

        p_margin_att_positive = np.mean((attributes == 1))
        p_margin_att_negative = np.mean((attributes == 0))

        p_condition_att_positive = p_joint_att_positive / (p_margin_att_positive + 1e-8)
        p_condition_att_negative = p_joint_att_negative / (p_margin_att_negative + 1e-8)
        demo_parity = np.abs(p_condition_att_positive - p_condition_att_negative)

        return demo_parity

    @staticmethod
    def true_positive_parity(
        predictions: np.ndarray, targets: np.ndarray, attributes: np.ndarray
    ) -> float:
        """
        P(y_hat=1|y=1,a=0) - P(y_hat=1|y=1,a=1)
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
        """
        |P(y_hat=1|y=0,a=0) - P(y_hat=1|y=0,a=1)|
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
    def p_rule(predictions: np.ndarray, attributes: np.ndarray):
        y_z_1 = predictions[attributes == 1]
        y_z_0 = predictions[attributes == 0]
        odds = y_z_1.mean() / y_z_0.mean()
        return np.min([odds, 1 / odds]) * 100

    def evaluate_subgroup_metrics(self):
        """
        Calculates verious metrics on different subgroups of data

        Returns:
            Nested dictionary where the first key is the index of
            the attribute being tested, and the second key is the
            metric being calculated.
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

            attributes = np.concatenate(np.array(attributes, dtype=object))
            predictions = np.concatenate(np.array(predictions, dtype=object))
            targets = np.concatenate(np.array(targets, dtype=object))

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
