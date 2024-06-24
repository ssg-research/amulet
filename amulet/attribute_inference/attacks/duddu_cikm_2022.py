"""Implementation of attribute inference algorithm"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve

from .attribute_inference_attack import AttributeInferenceAttack


class DudduCIKM2022(AttributeInferenceAttack):
    """
    Implementation of attribute inference attack from the method from:
    https://github.com/vasishtduddu/AttInfExplanations


    Attributes:
        target_model: :class:`~nn.Module`
            This model will be extracted.
        x_train_adv: :class:`~numpy.ndarray`
            input features for training adversary' attack model
        x_test: :class:`~numpy.ndarray`
            input features for testing adversary' attack model
        z_train_adv: :class:`~numpy.ndarray`
            sensitive attributes for training adversary' attack model
        device: str
            Device used to train model. Example: "cuda:0".
    """

    def __init__(
        self,
        target_model: nn.Module,
        x_train_adv: np.ndarray,
        x_test: np.ndarray,
        z_train_adv: np.ndarray,
        device: str,
    ):
        super().__init__(target_model, x_train_adv, x_test, z_train_adv, device)

    def attack_predictions(self) -> dict[int, dict[str, np.ndarray]]:
        """
        Runs the attribute inference attack by training an attack model
        to predict the sensitive attributes of a model using the predictions
        of the target model

        Returns:
            Nested dictionary:
                {i: int, result: dict}
                    where i is the index of the sensitive attribute
                    and result is a dictionary containing:
                        {'predictions': np.ndarray,
                         'confidence_values': np.ndarray}
        """
        predictions_train = self.model(
            torch.from_numpy(self.x_train_adv).type(torch.float).to(self.device)
        )
        predictions_train = predictions_train.detach().cpu().numpy()
        attack_model_train_x = pd.DataFrame(
            predictions_train,
            columns=["class1", "class2"],  # type: ignore[reportArgumentType]
        )

        predictions_test = self.model(
            torch.from_numpy(self.x_test).type(torch.float).to(self.device)
        )
        predictions_test = predictions_test.detach().cpu().numpy()
        attack_model_test_x = pd.DataFrame(
            predictions_test,
            columns=["class1", "class2"],  # type: ignore[reportArgumentType]
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
