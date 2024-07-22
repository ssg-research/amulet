import sys
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from tqdm import tqdm

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

# TODO: List of issues to fix:
# - Does not use target model, instead trains many "victim" models.
# - Hardcoded values for attributes. Needs to be generalized.
# - Need to figure out a design that attacks a single target model. 
#   For evaluation using metrics we may need to figure out a more complex pipeline.
class DistributionInferenceAttack:
    def __init__(
        self,
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        z_train: np.ndarray,
        z_test: np.ndarray,
        dataset_name: str,
        ratio1: float,
        ratio2: float,
        filter_prop: str,
    ) -> None:
        self.x_train, self.y_train, self.z_train = x_train, y_train, z_train
        self.x_test, self.y_test, self.z_test = x_test, y_test, z_test
        self.dataset_name = dataset_name
        self.ratio1 = ratio1
        self.ratio2 = ratio2
        self.filter_prop = filter_prop

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
        train_df_victim, train_df_adv = s_split(df_train)
        test_df_victim, test_df_adv = s_split(df_test)

        def prepare_one_set(TRAIN_DF, TEST_DF, split, prop_ratio, filter_prop):
            TRAIN_DF = get_filter(
                TRAIN_DF, filter_prop, split, prop_ratio, self.dataset_name, is_test=0
            )
            TEST_DF = get_filter(
                TEST_DF, filter_prop, split, prop_ratio, self.dataset_name, is_test=1
            )  # keep the test dataset fixed
            (x_tr, y_tr, cols), (x_te, y_te, cols) = (
                self.get_x_y(TRAIN_DF),
                self.get_x_y(TEST_DF),
            )
            return (x_tr, y_tr), (x_te, y_te), cols

        (X_train_victim_1, y_train_victim_1), (X_test_victim_1, y_test_victim_1), _ = (
            prepare_one_set(
                train_df_victim,
                test_df_victim,
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
            train_df_adv,
            test_df_adv,
            "attacker",
            self.ratio1,
            self.filter_prop,
        )
        (X_train_victim_2, y_train_victim_2), (X_test_victim_2, y_test_victim_2), _ = (
            prepare_one_set(
                train_df_victim,
                test_df_victim,
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
            train_df_adv,
            test_df_adv,
            "attacker",
            self.ratio2,
            self.filter_prop,
        )

        vic_traindata_1 = TensorDataset(
            torch.from_numpy(np.array(X_train_victim_1)).type(torch.float),
            torch.from_numpy(np.array(y_train_victim_1)).type(torch.long).squeeze(1),
        )
        att_traindata_1 = TensorDataset(
            torch.from_numpy(np.array(X_train_attacker_1)).type(torch.float),
            torch.from_numpy(np.array(y_train_attacker_1)).type(torch.long).squeeze(1),
        )
        vic_traindata_2 = TensorDataset(
            torch.from_numpy(np.array(X_train_victim_2)).type(torch.float),
            torch.from_numpy(np.array(y_train_victim_2)).type(torch.long).squeeze(1),
        )
        att_traindata_2 = TensorDataset(
            torch.from_numpy(np.array(X_train_attacker_2)).type(torch.float),
            torch.from_numpy(np.array(y_train_attacker_2)).type(torch.long).squeeze(1),
        )

        vic_testdata_1 = TensorDataset(
            torch.from_numpy(np.array(X_test_victim_1)).type(torch.float),
            torch.from_numpy(np.array(y_test_victim_1)).type(torch.long).squeeze(1),
        )
        att_testdata_1 = TensorDataset(
            torch.from_numpy(np.array(X_test_attacker_1)).type(torch.float),
            torch.from_numpy(np.array(y_test_attacker_1)).type(torch.long).squeeze(1),
        )
        vic_testdata_2 = TensorDataset(
            torch.from_numpy(np.array(X_test_victim_2)).type(torch.float),
            torch.from_numpy(np.array(y_test_victim_2)).type(torch.long).squeeze(1),
        )
        att_testdata_2 = TensorDataset(
            torch.from_numpy(np.array(X_test_attacker_2)).type(torch.float),
            torch.from_numpy(np.array(y_test_attacker_2)).type(torch.long).squeeze(1),
        )

        testdata_1 = ConcatDataset([att_testdata_1, vic_testdata_1])
        testdata_2 = ConcatDataset([att_testdata_2, vic_testdata_2])

        vic_trainloader_1 = DataLoader(
            dataset=vic_traindata_1, batch_size=256, shuffle=False
        )
        vic_trainloader_2 = DataLoader(
            dataset=vic_traindata_2, batch_size=256, shuffle=False
        )
        att_trainloader_1 = DataLoader(
            dataset=att_traindata_1, batch_size=256, shuffle=False
        )
        att_trainloader_2 = DataLoader(
            dataset=att_traindata_2, batch_size=256, shuffle=False
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