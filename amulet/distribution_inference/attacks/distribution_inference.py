import sys
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit


def filter(df, condition, ratio, verbose=True):
    ratio = float(ratio)
    qualify = np.nonzero((condition(df)).to_numpy())[0]
    notqualify = np.nonzero(np.logical_not((condition(df)).to_numpy()))[0]
    current_ratio = len(qualify) / (len(qualify) + len(notqualify))
    # If current ratio less than desired ratio, subsample from non-ratio
    if verbose:
        print("Changing ratio from %.2f to %.2f" % (current_ratio, ratio))
    if current_ratio <= ratio:
        np.random.shuffle(notqualify)
        if ratio < 1:
            nqi = notqualify[: int(((1 - ratio) * len(qualify)) / ratio)]
            return pd.concat([df.iloc[qualify], df.iloc[nqi]])
        return df.iloc[qualify]
    else:
        np.random.shuffle(qualify)
        if ratio > 0:
            qi = qualify[: int((ratio * len(notqualify)) / (1 - ratio))]
            return pd.concat([df.iloc[qi], df.iloc[notqualify]])
        return df.iloc[notqualify]


def heuristic(
    df,
    condition,
    ratio,
    cwise_sample,
    class_imbalance=2.0,
    n_tries=1000,
    class_col="label",
    verbose=True,
):
    ratio = float(ratio)
    vals, pckds = [], []
    iterator = range(n_tries)
    if verbose:
        iterator = tqdm(iterator)
    for _ in iterator:
        pckd_df = filter(df, condition, ratio, verbose=False)

        # Class-balanced sampling
        zero_ids = np.nonzero(pckd_df[class_col].to_numpy() == 0)[0]
        one_ids = np.nonzero(pckd_df[class_col].to_numpy() == 1)[0]

        # Sub-sample data, if requested
        if cwise_sample is not None:
            if class_imbalance >= 1:
                zero_ids = np.random.permutation(zero_ids)[
                    : int(class_imbalance * cwise_sample)
                ]
                one_ids = np.random.permutation(one_ids)[:cwise_sample]
            else:
                zero_ids = np.random.permutation(zero_ids)[:cwise_sample]
                one_ids = np.random.permutation(one_ids)[
                    : int(1 / class_imbalance * cwise_sample)
                ]

        # Combine them together
        pckd = np.sort(np.concatenate((zero_ids, one_ids), 0))
        pckd_df = pckd_df.iloc[pckd]

        vals.append(condition(pckd_df).mean())
        pckds.append(pckd_df)

        # Print best ratio so far in descripton
        if verbose:
            iterator.set_description(  # type: ignore[reportAttributeAccessIssue]
                "%.4f" % (ratio + np.min([np.abs(zz - ratio) for zz in vals]))
            )

    vals = np.abs(np.array(vals) - ratio)
    # Pick the one closest to desired ratio
    picked_df = pckds[np.argmin(vals)]
    return picked_df.reset_index(drop=True)


def get_filter(df, filter_prop, split, ratio, dataset_name, is_test):
    if dataset_name == "census":
        if filter_prop == "sex":

            def lambda_fn(x):
                return x["sex"] == 1
        elif filter_prop == "race":

            def lambda_fn(x):
                return x["race"] == 1
        else:
            print("Incorrect filter prop")
            sys.exit()

        prop_wise_subsample_sizes = {
            "attacker": {
                "sex": (1100, 500),
                "race": (2000, 1000),
            },
            "victim": {
                "sex": (1100, 500),
                "race": (2000, 1000),
            },
        }
        subsample_size = prop_wise_subsample_sizes[split][filter_prop][is_test]
        return heuristic(
            df,
            lambda_fn,
            ratio,
            subsample_size,
            class_imbalance=3,
            n_tries=100,
            class_col="y",
            verbose=False,
        )

    else:
        if filter_prop == "sex":

            def lambda_fn(x):
                return x["sex"] == 1
        elif filter_prop == "race":

            def lambda_fn(x):
                return x["race"] == 1
        else:
            print("Incorrect filter prop")
            sys.exit()

        prop_wise_subsample_sizes = {
            "attacker": {
                "sex": (2200, 1200),
                "race": (2200, 1200),
            },
            "victim": {
                "sex": (2200, 1200),
                "race": (2200, 1200),
            },
        }
        subsample_size = prop_wise_subsample_sizes[split][filter_prop][is_test]
        return heuristic(
            df,
            lambda_fn,
            ratio,
            subsample_size,
            class_imbalance=3,
            n_tries=100,
            class_col="y",
            verbose=False,
        )


class DistributionInference:
    """
    Implementation of attribute inference attack from the method from:
    https://github.com/vasishtduddu/AttInfExplanations


    Attributes:
        target_model: :class:`~torch.nn.Module`
            This model will be extracted.
        x_train: :class:`~numpy.ndarray`
            input features for training adversary' attack model
        x_test: :class:`~numpy.ndarray`
            input features for testing adversary' attack model
        y_train: :class:`~numpy.ndarray`
            class labels for train dataset
        y_test: :class:`~numpy.ndarray`
            class labels for test dataset
        z_train: :class:`~numpy.ndarray`
            sensitive attributes for training adversary' attack model (includes both "race" and "sex")
        z_test: :class:`~numpy.ndarray`
            sensitive attributes for training adversary' attack model (includes both "race" and "sex")
        filter_prop: str
            Filter: "race", "sex"
        ratio1: float
            ratio of distribution 1
        ratio2: float
            ratio of distribution 2
    """

    def __init__(
        self,
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        z_train: np.ndarray,
        z_test: np.ndarray,
        filter_prop: str,
        ratio1: float,
        ratio2: float,
        device: str,
        args: argparse.Namespace,
    ):
        self.ratio1 = ratio1
        self.ratio2 = ratio2
        self.device = device
        self.filter_prop = filter_prop
        self.x_train, self.y_train, self.z_train = x_train, y_train, z_train
        self.x_test, self.y_test, self.z_test = x_test, y_test, z_test
        self.args = args

    def prepare_dataset(self):
        x_train = pd.DataFrame(self.x_train)
        y_train = pd.DataFrame(self.y_train, columns=["y"])  # type: ignore[reportArgumentType]
        z_train = pd.DataFrame(self.z_train, columns=["race", "sex"])  # type: ignore[reportArgumentType]
        df_train = pd.concat([x_train, y_train, z_train], axis=1)
        x_test = pd.DataFrame(self.x_test)
        y_test = pd.DataFrame(self.y_test, columns=["y"])  # type: ignore[reportArgumentType]
        z_test = pd.DataFrame(self.z_test, columns=["race", "sex"])  # type: ignore[reportArgumentType]
        df_test = pd.concat([x_test, y_test, z_test], axis=1)

        # temporary fix: race attribute has three labels instead of 2...
        # ..removing all records with attribute value==2...but ideally remove "Asians" in original data
        df_train = df_train[df_train.race != 2]
        df_test = df_test[df_test.race != 2]

        def s_split(this_df):
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
            splitter = sss.split(this_df, this_df[["sex", "race", "y"]])
            split_1, split_2 = next(splitter)
            return this_df.iloc[split_1], this_df.iloc[split_2]

        # Create train/test splits for victim/adv
        self.train_df_victim, self.train_df_adv = s_split(df_train)
        self.test_df_victim, self.test_df_adv = s_split(df_test)
        # print(self.train_df_victim.shape,self.train_df_adv.shape)

        def prepare_one_set(TRAIN_DF, TEST_DF, split, prop_ratio, filter_prop):
            TRAIN_DF = get_filter(
                TRAIN_DF, filter_prop, split, prop_ratio, self.args.dataset, is_test=0
            )
            TEST_DF = get_filter(
                TEST_DF, filter_prop, split, prop_ratio, self.args.dataset, is_test=1
            )  # keep the test dataset fixed
            (x_tr, y_tr, cols), (x_te, y_te, cols) = (
                self.get_x_y(TRAIN_DF),
                self.get_x_y(TEST_DF),
            )
            return (x_tr, y_tr), (x_te, y_te), cols

        (X_train_victim_1, y_train_victim_1), (X_test_victim_1, y_test_victim_1), _ = (
            prepare_one_set(
                self.train_df_victim,
                self.test_df_victim,
                "victim",
                self.ratio1,
                self.filter_prop,
            )
        )
        (
            (X_train_attacker_1, y_train_attacker_1),
            (X_test_attacker_1, y_test_attacker_1),
            _,
        ) = prepare_one_set(
            self.train_df_adv,
            self.test_df_adv,
            "attacker",
            self.ratio1,
            self.filter_prop,
        )
        (X_train_victim_2, y_train_victim_2), (X_test_victim_2, y_test_victim_2), _ = (
            prepare_one_set(
                self.train_df_victim,
                self.test_df_victim,
                "victim",
                self.ratio2,
                self.filter_prop,
            )
        )
        (
            (X_train_attacker_2, y_train_attacker_2),
            (X_test_attacker_2, y_test_attacker_2),
            _,
        ) = prepare_one_set(
            self.train_df_adv,
            self.test_df_adv,
            "attacker",
            self.ratio2,
            self.filter_prop,
        )

        self.vic_traindata_1 = TensorDataset(
            torch.from_numpy(np.array(X_train_victim_1)).type(torch.float),
            torch.from_numpy(np.array(y_train_victim_1)).type(torch.long).squeeze(1),
        )
        self.att_traindata_1 = TensorDataset(
            torch.from_numpy(np.array(X_train_attacker_1)).type(torch.float),
            torch.from_numpy(np.array(y_train_attacker_1)).type(torch.long).squeeze(1),
        )
        self.vic_traindata_2 = TensorDataset(
            torch.from_numpy(np.array(X_train_victim_2)).type(torch.float),
            torch.from_numpy(np.array(y_train_victim_2)).type(torch.long).squeeze(1),
        )
        self.att_traindata_2 = TensorDataset(
            torch.from_numpy(np.array(X_train_attacker_2)).type(torch.float),
            torch.from_numpy(np.array(y_train_attacker_2)).type(torch.long).squeeze(1),
        )

        self.vic_testdata_1 = TensorDataset(
            torch.from_numpy(np.array(X_test_victim_1)).type(torch.float),
            torch.from_numpy(np.array(y_test_victim_1)).type(torch.long).squeeze(1),
        )
        self.att_testdata_1 = TensorDataset(
            torch.from_numpy(np.array(X_test_attacker_1)).type(torch.float),
            torch.from_numpy(np.array(y_test_attacker_1)).type(torch.long).squeeze(1),
        )
        self.vic_testdata_2 = TensorDataset(
            torch.from_numpy(np.array(X_test_victim_2)).type(torch.float),
            torch.from_numpy(np.array(y_test_victim_2)).type(torch.long).squeeze(1),
        )
        self.att_testdata_2 = TensorDataset(
            torch.from_numpy(np.array(X_test_attacker_2)).type(torch.float),
            torch.from_numpy(np.array(y_test_attacker_2)).type(torch.long).squeeze(1),
        )

        testdata_1 = ConcatDataset([self.att_testdata_1, self.vic_testdata_1])
        testdata_2 = ConcatDataset([self.att_testdata_2, self.vic_testdata_2])

        vic_trainloader_1 = DataLoader(
            dataset=self.vic_traindata_1, batch_size=256, shuffle=False
        )
        vic_trainloader_2 = DataLoader(
            dataset=self.vic_traindata_2, batch_size=256, shuffle=False
        )
        att_trainloader_1 = DataLoader(
            dataset=self.att_traindata_1, batch_size=256, shuffle=False
        )
        att_trainloader_2 = DataLoader(
            dataset=self.att_traindata_2, batch_size=256, shuffle=False
        )
        test_loader_1 = DataLoader(dataset=testdata_1, batch_size=256, shuffle=False)
        test_loader_2 = DataLoader(dataset=testdata_2, batch_size=256, shuffle=False)

        return (
            vic_trainloader_1,
            vic_trainloader_2,
            att_trainloader_1,
            att_trainloader_2,
            test_loader_1,
            test_loader_2,
        )

    def get_x_y(self, P):
        # Scale X values
        Y = P["y"].to_numpy()
        X = P.drop(columns="y", axis=1)
        # uncomment below to check if property ratios are correct
        # if self.filter_prop == "sex":
        #     print(X["sex"].value_counts())
        # if self.filter_prop == "race":
        #     print(X["race"].value_counts())
        cols = X.columns
        X = X.to_numpy()
        return (X.astype(float), np.expand_dims(Y, 1), cols)

    def _get_kl_preds(self, ka, kb, kc1, kc2):
        def sigmoid(x):
            exp = np.exp(x)
            return exp / (1 + exp)

        def KL(x, y):
            small_eps = 1e-4
            x_ = np.clip(x, small_eps, 1 - small_eps)
            y_ = np.clip(y, small_eps, 1 - small_eps)
            x__, y__ = 1 - x_, 1 - y_
            first_term = x_ * (np.log(x_) - np.log(y_))
            second_term = x__ * (np.log(x__) - np.log(y__))
            return np.mean(first_term + second_term, 1)

        def _check(x):
            if np.sum(np.isinf(x)) > 0 or np.sum(np.isnan(x)) > 0:
                print("Invalid values:", x)
                raise ValueError("Invalid values found!")

        def _pairwise_compare(x, y, xx, yy):
            x_ = np.expand_dims(x, 2)
            y_ = np.expand_dims(y, 2)
            y_ = np.transpose(y_, (0, 2, 1))
            pairwise_comparisons = x_ - y_
            preds = np.array([z[xx, yy] for z in pairwise_comparisons])
            return preds

        ka_, kb_ = ka, kb
        kc1_, kc2_ = kc1, kc2

        ka_, kb_ = sigmoid(ka), sigmoid(kb)
        kc1_, kc2_ = sigmoid(kc1), sigmoid(kc2)

        small_eps = 1e-4
        log_vals_a = np.log((small_eps + ka_) / (small_eps + 1 - ka_))
        log_vals_b = np.log((small_eps + kb_) / (small_eps + 1 - kb_))
        ordering = np.mean(np.abs(log_vals_a - log_vals_b), 0)
        ordering = np.argsort(ordering)[::-1]
        # Pick only first half
        ordering = ordering[: len(ordering) // 2]
        ka_, kb_ = ka_[:, ordering], kb_[:, ordering]
        kc1_, kc2_ = kc1_[:, ordering], kc2_[:, ordering]

        # Consider all unique pairs of models
        xx, yy = np.triu_indices(ka.shape[0], k=1)

        # Randomly pick pairs of models
        random_pick = np.random.permutation(xx.shape[0])[: int(0.8 * xx.shape[0])]
        xx, yy = xx[random_pick], yy[random_pick]

        # Compare the KL divergence between the two distributions
        # For both sets of victim models
        KL_vals_1_a = np.array([KL(ka_, x) for x in kc1_])
        _check(KL_vals_1_a)
        KL_vals_1_b = np.array([KL(kb_, x) for x in kc1_])
        _check(KL_vals_1_b)
        KL_vals_2_a = np.array([KL(ka_, x) for x in kc2_])
        _check(KL_vals_2_a)
        KL_vals_2_b = np.array([KL(kb_, x) for x in kc2_])
        _check(KL_vals_2_b)

        preds_first = _pairwise_compare(KL_vals_1_a, KL_vals_1_b, xx, yy)
        preds_second = _pairwise_compare(KL_vals_2_a, KL_vals_2_b, xx, yy)

        return preds_first, preds_second

    def get_preds(self, loader, models):
        """
        Get predictions for given models on given data
        """

        predictions = []
        ground_truth = []
        # Accumulate all data for given loader
        for data in loader:
            labels = data[1]
            ground_truth.append(labels.cpu().numpy())
            # if preload:
            #     inputs.append(features.cuda())
        ground_truth = np.concatenate(ground_truth, axis=0)

        iterator = tqdm(models, desc="Generating Predictions")
        for model in iterator:
            # Shift model to GPU
            model = model.to(self.device)
            # Make sure model is in evaluation mode
            model.eval()
            # Clear GPU cache
            torch.cuda.empty_cache()

            with torch.no_grad():
                predictions_on_model = []
                for data in loader:
                    if len(data) == 2:
                        data_points, labels = data[0], data[1]
                    else:
                        data_points, labels, _ = data[0], data[1], data[2]
                    prediction = model(data_points.to(self.device)).detach()
                    # if not multi_class:
                    prediction = prediction[:, 0]
                    predictions_on_model.append(prediction)
            predictions_on_model = torch.cat(predictions_on_model).cpu().numpy()
            predictions.append(predictions_on_model)
            # Shift model back to CPU
            model = model.cpu()
            del model
            torch.cuda.empty_cache()
        predictions = np.stack(predictions, 0)
        torch.cuda.empty_cache()

        return predictions, ground_truth

    def attack(
        self,
        models_vic_1,
        models_vic_2,
        models_adv_1,
        models_adv_2,
        testloader_1,
        testloader_2,
    ):
        preds_vic_prop1_dist1, _ = self.get_preds(testloader_1, models_vic_1)
        preds_vic_prop2_dist1, _ = self.get_preds(testloader_1, models_vic_2)
        preds_adv_prop1_dist1, _ = self.get_preds(testloader_1, models_adv_1)
        preds_adv_prop2_dist1, _ = self.get_preds(testloader_1, models_adv_2)

        preds_vic_prop1_dist2, _ = self.get_preds(testloader_2, models_vic_1)
        preds_vic_prop2_dist2, _ = self.get_preds(testloader_2, models_vic_2)
        preds_adv_prop1_dist2, _ = self.get_preds(testloader_2, models_adv_1)
        preds_adv_prop2_dist2, _ = self.get_preds(testloader_2, models_adv_2)

        preds_1_first, preds_1_second = self._get_kl_preds(
            preds_adv_prop1_dist1,
            preds_adv_prop2_dist1,
            preds_vic_prop1_dist1,
            preds_vic_prop2_dist1,
        )
        preds_2_first, preds_2_second = self._get_kl_preds(
            preds_adv_prop1_dist2,
            preds_adv_prop2_dist2,
            preds_vic_prop1_dist2,
            preds_vic_prop2_dist2,
        )

        # Combine data
        preds_first = np.concatenate((preds_1_first, preds_2_first), 1)
        preds_second = np.concatenate((preds_1_second, preds_2_second), 1)
        preds = np.concatenate((preds_first, preds_second))

        # if not self.config.kl_voting:
        preds -= np.min(preds, 0)
        preds /= np.max(preds, 0)

        preds = np.mean(preds, 1)
        gt = np.concatenate(
            (np.zeros(preds_first.shape[0]), np.ones(preds_second.shape[0]))
        )
        acc = 100 * np.mean((preds >= 0.5) == gt)
        return acc
