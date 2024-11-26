"""Adversarial Debiasing Implementation"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn import metrics

from .discr_behavior_defense import DicriminatoryBehaviorDefense


class AdversaryModel(nn.Module):
    """
    Model used as a discriminator to identify the sensitive
    attribute given the output of the model.
    Attributes:
        n_sensitive_attrs: int
            Number of sensitive attributes in the dataset.
        n_classes: int
            Number of classes in the dataset.
    """

    def __init__(self, n_sensitive_attrs: int = 2, n_classes: int = 2):
        super(AdversaryModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_classes, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, n_sensitive_attrs),
        )

    def forward(self, x):
        return torch.sigmoid(self.network(x))


class AdversarialDebiasing(DicriminatoryBehaviorDefense):
    """
    Implementation of Adversarial Training algorithm from the method from cleverhans:
    https://xebia.com/blog/towards-fairness-in-ml-with-adversarial-networks/

    Reference:
        learning to Pivot with Adversarial Networks
        Gilles Louppe, Michael Kagan, Kyle Cranmer
        Learning to Pivot with Adversarial Networks.

    Attributes:
        model: :class:`~torch.nn.Module`
            The model on which to apply adversarial training.
        criterion: :class:`~torch.nn.Module`
            Loss function for adversarial training.
        optimizer: :class:`~torch.optim.Optimizer`
            Optimizer for adversarial training.
        train_loader: :class:`~torch.utils.data.DataLoader`
            Training data loader to adversarial training.
        test_loader: :class:`~torch.utils.data.DataLoader`
            Testing data loader to adversarial training.
        n_sensitive_attrs: int
            Number of sensitive attributes in the dataset.
        n_classes: int
            Number of classes in the dataset.
        lambdas: :class:`torch.Tensor`
            Hyperparameters for fairness objective function.
        device: str
            Device used to train model. Example: "cuda:0".
        epochs: int
            Determines number of iterations over training data.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        test_loader: DataLoader,
        n_sensitive_attrs: int,
        n_classes: int,
        lambdas: torch.Tensor,
        device: str,
        epochs: int = 5,
    ):
        super().__init__(model, criterion, optimizer, train_loader, test_loader, device)

        self.lambdas = lambdas.to(device)
        self.epochs = epochs

        self.discmodel = AdversaryModel(n_sensitive_attrs, n_classes).to(self.device)
        self.adv_criterion = nn.BCELoss(reduce=False)
        self.adv_optimizer = torch.optim.Adam(self.discmodel.parameters())
        self.discmodel = self.__pretrain_adversary()

    def train_fair(self) -> nn.Module:
        print("Training Model with Adversarial Debiasing")
        self.model.train()
        for _ in range(1, 165):
            for x, y, z in self.train_loader:  # Train adversary
                x, y, z = x.to(self.device), y.to(self.device), z.to(self.device)
                p_y = self.model(x)
                self.discmodel.zero_grad()
                p_z = self.discmodel(p_y)
                loss_adv = (self.adv_criterion(p_z, z) * self.lambdas).mean()
                loss_adv.backward()
                self.adv_optimizer.step()

            for x, y, z in self.train_loader:  # train on a single batch
                x, y, z = x.to(self.device), y.to(self.device), z.to(self.device)
                pass
            p_y = self.model(x)  # type: ignore[reportPossiblyUnboundVariable]
            p_z = self.discmodel(p_y)
            self.model.zero_grad()
            p_z = self.discmodel(p_y)
            loss_adv = (self.adv_criterion(p_z, z) * self.lambdas).mean()  # type: ignore[reportPossiblyUnboundVariable]
            model_loss = (
                self.model_criterion(p_y, y)  # type: ignore[reportPossiblyUnboundVariable]
                - (self.adv_criterion(self.discmodel(p_y), z) * self.lambdas).mean()
            )
            model_loss.backward()
            self.model_optimizer.step()

        return self.model

    def adversary_report(self) -> tuple[float, float]:
        self.discmodel.eval()
        self.model.eval()
        Z_pred_list = []
        Z_true_list = []
        with torch.no_grad():
            for X_batch, _, Z_batch in self.test_loader:
                X_batch, Z_batch = X_batch.to(self.device), Z_batch.to(self.device)
                y_test_pred = self.model(X_batch)
                Z_test_pred = self.discmodel(y_test_pred)
                Z_pred_list.append(Z_test_pred.cpu().numpy())
                Z_true_list.append(Z_batch.cpu().numpy())

        race_pred = []
        sex_pred = []
        for i in Z_pred_list:
            for j in i:
                race_pred.append(j[0])
                sex_pred.append(j[1])

        race_true = []
        sex_true = []
        for i in Z_true_list:
            for j in i:
                race_true.append(j[0])
                sex_true.append(j[1])

        race_pred = [a.squeeze().tolist() for a in race_pred]
        race_true = [a.squeeze().tolist() for a in race_true]
        sex_pred = [a.squeeze().tolist() for a in sex_pred]
        sex_true = [a.squeeze().tolist() for a in sex_true]

        def searchThreshold(race_pred, race_true, sex_pred, sex_true):
            from operator import itemgetter

            threshold = np.arange(0.0, 1.0, 0.01)
            race_auc = []
            sex_auc = []
            for t in threshold:
                race_temp = []
                sex_temp = []
                for val in race_pred:
                    if val <= t:
                        race_temp.append(0)
                    else:
                        race_temp.append(1)
                race_auc.append((t, metrics.roc_auc_score(race_true, race_temp)))
                for val in sex_pred:
                    if val < t:
                        sex_temp.append(0)
                    else:
                        sex_temp.append(1)
                sex_auc.append((t, metrics.roc_auc_score(sex_true, sex_temp)))
            t1 = max(race_auc, key=itemgetter(1))[0]
            t2 = max(sex_auc, key=itemgetter(1))[0]
            return t1, t2

        # Find Threshold
        sex_true = [int(a) for a in sex_true]
        race_true = [int(a) for a in race_true]
        t1, t2 = searchThreshold(race_pred, race_true, sex_pred, sex_true)
        new_race_pred = []
        new_sex_pred = []
        for val in race_pred:
            if val <= t1:
                new_race_pred.append(0)
            else:
                new_race_pred.append(1)
        for val in sex_pred:
            if val <= t2:
                new_sex_pred.append(0)
            else:
                new_sex_pred.append(1)
        race_pred = [int(a) for a in race_pred]
        sex_pred = [int(a) for a in sex_pred]

        auc_race = float(metrics.roc_auc_score(race_true, new_race_pred))
        auc_sex = float(metrics.roc_auc_score(sex_true, new_sex_pred))
        print("AUC [race]: {}, [sex]: {}".format(auc_race, auc_sex))
        return auc_race, auc_sex

    def __pretrain_adversary(self):
        print("Pretraining Adversary Model")
        for _ in range(5):
            for x, _, z in self.train_loader:
                x, z = x.to(self.device), z.to(self.device)
                p_y = self.model(x).detach()
                self.discmodel.zero_grad()
                p_z = self.discmodel(p_y)
                loss = (self.adv_criterion(p_z, z) * self.lambdas).mean()
                loss.backward()
                self.adv_optimizer.step()

        return self.discmodel
