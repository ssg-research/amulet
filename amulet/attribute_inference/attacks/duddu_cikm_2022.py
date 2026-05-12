"""Implementation of attribute inference algorithm"""

import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_curve
from sklearn.neural_network import MLPClassifier

from ...utils import get_predictions_numpy
from .attribute_inference_attack import AttributeInferenceAttack


class DudduCIKM2022(AttributeInferenceAttack):
    """
    Attribute inference attack using an MLP trained on target model outputs.

    Reference: https://github.com/vasishtduddu/AttInfExplanations

    Attributes:
        target_model: Target model whose sensitive attributes are inferred.
        x_train_adv: Input features for training the adversary attack model.
        x_test: Input features for testing the adversary attack model.
        z_train_adv: Sensitive attribute labels for training the adversary.
        device: Device used to train model. Example: "cuda:0".
    """

    def __init__(
        self,
        target_model: nn.Module,
        x_train_adv: np.ndarray,
        x_test: np.ndarray,
        z_train_adv: np.ndarray,
        batch_size: int,
        device: str,
    ):
        super().__init__(target_model, x_train_adv, x_test, z_train_adv, device)
        self.batch_size = batch_size

    def attack(self) -> dict[int, dict[str, np.ndarray]]:
        """
        Run the attribute inference attack.

        Trains an MLP on the target model's outputs to predict sensitive attributes.

        Returns:
            Nested dictionary mapping attribute index to a dict with keys
            "predictions" and "confidence_values".
        """
        attack_model_train_x = get_predictions_numpy(
            self.x_train_adv, self.target_model, self.batch_size, self.device
        )
        attack_model_test_x = get_predictions_numpy(
            self.x_test, self.target_model, self.batch_size, self.device
        )

        num_sensitive_attributes = self.z_train_adv.shape[1]
        results = {}

        for i in range(num_sensitive_attributes):
            attack_model = MLPClassifier(
                solver="adam",
                alpha=1e-3,
                hidden_layer_sizes=(
                    64,
                    128,
                    32,
                ),
                max_iter=300,
                random_state=1337,
            )
            attack_model.fit(attack_model_train_x, self.z_train_adv[:, i])
            z_pred_prob = attack_model.predict_proba(attack_model_train_x)
            z_pred_prob = z_pred_prob[:, 1]  # type: ignore[reportArgumentType]

            fpr, tpr, thresholds = roc_curve(self.z_train_adv[:, i], z_pred_prob)
            gmeans = np.sqrt(tpr * (1 - fpr))
            ix = np.argmax(gmeans)
            best_thresh = thresholds[ix]

            # Thresholding on test dataset
            z_pred_prob = attack_model.predict_proba(attack_model_test_x)
            z_pred_prob = z_pred_prob[:, 1]  # type: ignore[reportArgumentType]
            z_pred = z_pred_prob > best_thresh

            results[i] = {"predictions": z_pred, "confidence_values": z_pred_prob}

        return results
