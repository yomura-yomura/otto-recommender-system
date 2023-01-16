import pathlib
import cudf
import otto_recommender_system as ors
import otto_recommender_system.data
import otto_recommender_system.validating
import numpy as np
from typing import Optional, List, Union, Dict, Callable
import tqdm
import pandas as pd
import json


this_dir_path = pathlib.Path(__file__).resolve().parent
saved_dir_path = this_dir_path / "data" / "co-visitation-matrix"
saved_dir_path.mkdir(exist_ok=True, parents=True)


class Runner:
    def __init__(
            self, all_train_df, dirname, max_memory_gb=1, n_seperated_aid=8,
            types_to_use: Optional[List[Union[int, str]]] = None,
            weight_func: Optional[Callable[[cudf.DataFrame], Dict[Union[int, str], Union[int, float]]]] = None
    ):
        all_aid = np.unique(all_train_df["aid"])
        assert all_aid.min() == 0
        assert all_aid.max() == len(all_aid) - 1

        self.dirname = saved_dir_path / dirname
        self.dirname.mkdir(exist_ok=True)

        def _validate_type_as_int(type_):
            return ors.data.all_types.index(type_) if isinstance(type_, str) else int(type_)

        self.types_to_use = types_to_use
        if self.types_to_use is not None:
            self.types_to_use = [_validate_type_as_int(type_to_use) for type_to_use in self.types_to_use]

        if weight_func is not None:
            def _wrapper(df):
                ret = weight_func(df)
                if isinstance(ret, dict):
                    ret = {_validate_type_as_int(type_): float(v) for type_, v in ret.items()}
                return ret

            self.weight_func = lambda df: _wrapper(df)
        else:
            self.weight_func = lambda _: 1

        n_unique_aid = len(all_aid)
        self.aid_edges = np.linspace(0, n_unique_aid, n_seperated_aid).astype(np.int32)
        self.total_weight_cudf: Optional[cudf.DataFrame] = None
        self.n_seperated_aid = n_seperated_aid

        dtype_itemsize = sum(dtype.itemsize for dtype in all_train_df.dtypes)

        # indices_to_divide = indices[np.unique(indices // n_iters, return_index=True)[1]]
        indices_to_divide = indices_in_all_sessions[
            np.unique(
                dtype_itemsize
                *
                np.cumsum((indices_in_all_sessions[1:] - indices_in_all_sessions[:-1]) ** 2)
                //
                (max_memory_gb * 1e9),
                return_index=True
            )[1]
        ]

        self.df_list = [
            all_train_df.iloc[f:l]
            for f, l in zip(indices_to_divide, [*indices_to_divide[1:], len(all_train_df)])
        ]
        assert len(all_train_df) == sum(map(len, self.df_list))

    def get_df(self, i_seperated_aid):
        if not (0 <= i_seperated_aid < self.n_seperated_aid):
            raise IndexError("list index out of range")
        target_fn = self.dirname / f"{i_seperated_aid}.parquet"
        if not target_fn.exists():
            raise FileNotFoundError(target_fn)
        return pd.read_parquet(target_fn)

    def get_top_df(self, i_seperated_aid, top=20):
        target_fn = self.dirname / f"top{top}" / f"{i_seperated_aid}.parquet"
        if target_fn.exists():
            return pd.read_parquet(target_fn)

        target_fn.parent.mkdir(exist_ok=True)

        df = self.get_df(i_seperated_aid)
        df = df.sort_values(["aid_x", "weight"], ascending=[True, False])
        df["i_top_weight"] = df.groupby("aid_x")["aid_y"].cumcount()
        df = df.loc[df["i_top_weight"] < top].drop(columns=["i_top_weight"])
        df.to_parquet(target_fn)
        return df

    def get_dict(self, top=20):
        target_fn = self.dirname / f"top{top}" / "top.json"
        if target_fn.exists():
            with open(target_fn, "r") as f:
                return {int(k): v for k, v in json.load(f).items()}

        top_20_dict = {}
        for i in tqdm.trange(self.n_seperated_aid, desc=f"get_dict at {self.dirname.name}"):
            top_20_df = self.get_top_df(i, top)
            top_20_dict.update(top_20_df.groupby("aid_x")["aid_y"].apply(tuple).to_dict())

        with open(target_fn, "w") as f:
            json.dump(top_20_dict, f)

        return top_20_dict

    def run(self, max_timedelta: np.timedelta64):
        for i_seperated_aid in range(self.n_seperated_aid):
            print(f"({i_seperated_aid + 1}/{self.n_seperated_aid}) split aid at {self.dirname.name}")
            target_fn = self.dirname / f"{i_seperated_aid}.parquet"
            if target_fn.exists():
                continue
            self.total_weight_cudf = None
            for df in tqdm.tqdm(self.df_list, desc="iter over split df"):
                self.each_step(df, i_seperated_aid, max_timedelta)
            total_weight_df = self.total_weight_cudf.to_pandas()
            total_weight_df.reset_index().to_parquet(target_fn)

    def each_step(self, chunk_df, i_seperated_aid: int, max_timedelta: np.timedelta64):
        assert isinstance(i_seperated_aid, int)
        assert isinstance(max_timedelta, np.timedelta64)

        chunk_cudf = cudf.from_pandas(chunk_df)
        if self.types_to_use is not None:
            chunk_cudf = chunk_cudf.loc[chunk_cudf["type"].isin(self.types_to_use)]

        chunk_cudf = chunk_cudf.merge(chunk_cudf, on="session")
        aid_edges = self.aid_edges
        chunk_cudf = chunk_cudf.query(
            "@aid_edges[@i_seperated_aid] <= aid_x < @aid_edges[@i_seperated_aid + 1]"
        )
        chunk_cudf = chunk_cudf.query(
            "-@max_timedelta < ts_x - ts_y < @max_timedelta"
        )
        chunk_cudf.drop_duplicates(["session", "aid_x", "aid_y"], inplace=True)

        ret = self.weight_func(chunk_cudf)
        if isinstance(ret, dict):
            chunk_cudf["weight"] = chunk_cudf["type_y"].map(ret)
        else:
            chunk_cudf["weight"] = ret

        chunk_total_weight_cudf = chunk_cudf.groupby(["aid_x", "aid_y"])["weight"].sum().astype(np.int32)

        if self.total_weight_cudf is None:
            self.total_weight_cudf = chunk_total_weight_cudf
        else:
            self.total_weight_cudf = self.total_weight_cudf.add(chunk_total_weight_cudf, fill_value=0)
        del chunk_total_weight_cudf


if __name__ == "__main__":
    train_df, valid_df, test_df, valid_labels_df = ors.data.get_datasets(
        type_of_tidy_data="parquet", return_valid_labels=True
    )
    all_train_df = ors.data.get_pd_tidy_data("train")

    # train_df = train_df.set_index(["session", "aid", "ts"])
    # all_train_df = all_train_df.set_index(["session", "aid", "ts"])


        # return len(merged_cudf) * dtype_itemsize * 1e-9

    all_sessions, indices_in_all_sessions = np.unique(all_train_df["session"], return_index=True)

    runner_clicks = Runner(all_train_df, dirname="clicks", weight_func=lambda df: 1 + 3 * (df["ts_x"] - all_train_df["ts"].min()) / (all_train_df["ts"].max() - all_train_df["ts"].min()))
    runner_clicks.run(max_timedelta=np.timedelta64(1, "D"))
    top_20_clicks_dict = runner_clicks.get_dict()


    runner_carts_orders = Runner(all_train_df, dirname="carts_orders", weight_func=lambda _: {0: 1, 1: 6, 2: 3})
    runner_carts_orders.run(max_timedelta=np.timedelta64(1, "D"))
    top_20_buys_dict = runner_carts_orders.get_dict()


    runner_buy2buy = Runner(all_train_df, dirname="buy2buy", types_to_use=["carts", "orders"])
    runner_buy2buy.run(max_timedelta=np.timedelta64(14, "D"))
    top_20_buy2buy_dict = runner_buy2buy.get_dict()


    import collections

    def suggest_clicks(df):
        global top_20_clicks

        unique_aids = set(df["aid"])

        co_visitation_aids = [
            co_visitation_aid
            for aid in unique_aids
            if aid in top_20_clicks_dict.keys()
            for co_visitation_aid in top_20_clicks_dict[aid]
        ]

        top_predicted_without_unique_aids = [
            aid
            for aid, _ in collections.Counter(co_visitation_aids).most_common(20)  # -> (aid, cnt)
            if aid not in unique_aids
        ]

        # Temp
        if len(unique_aids) > 20:
            unique_aids = set(aid for i, aid in enumerate(unique_aids) if i < 20)

        # unique_aidsを最優先
        top_predicted = unique_aids.union(top_predicted_without_unique_aids[:20 - len(unique_aids)])
        # 20に達していないときは、全体で良くクリックされるものを足す
        return top_predicted.union(top_20_clicks[:20 - len(top_predicted)])


    def suggest_buys(df):
        global top_20_orders

        unique_aids = set(df["aid"])
        df = df.loc[df["type"].isin([ors.data.all_types.index("carts"), ors.data.all_types.index("orders")])]
        unique_buys_aids = set(df["aid"])

        co_visitation_buys_aids = [
            co_visitation_aid
            for aid in unique_aids
            if aid in top_20_buys_dict.keys()
            for co_visitation_aid in top_20_buys_dict[aid]
        ]

        co_visitation_buy2buy_aids = [
            co_visitation_aid
            for aid in unique_buys_aids
            if aid in top_20_buy2buy_dict.keys()
            for co_visitation_aid in top_20_buy2buy_dict[aid]
        ]

        top_predicted_without_unique_aids = [
            aid
            for aid, _ in collections.Counter(co_visitation_buys_aids + co_visitation_buy2buy_aids).most_common(20)  # -> (aid, cnt)
            if aid not in unique_aids
        ]

        # Temp
        if len(unique_aids) > 20:
            unique_aids = set(aid for i, aid in enumerate(unique_aids) if i < 20)

        # unique_aidsを最優先
        top_predicted = unique_aids.union(top_predicted_without_unique_aids[:20 - len(unique_aids)])
        # 20に達していないときは、全体で良くクリックされるものを足す
        return top_predicted.union(top_20_orders[:20 - len(top_predicted)])


    # df = valid_df[valid_df["session"].unique()[2] == valid_df["session"]]

    def run(dataset_type):
        if dataset_type == "validation":
            target_df = valid_df
        elif dataset_type == "test":
            target_df = test_df
        else:
            raise ValueError(dataset_type)

        global top_20_clicks, top_20_orders
        top_20_clicks = target_df.loc[target_df["type"] == ors.data.all_types.index("clicks"), "aid"].value_counts().index[:20].tolist()
        top_20_orders = target_df.loc[target_df["type"] == ors.data.all_types.index("orders"), "aid"].value_counts().index[:20].tolist()

        (this_dir_path / "data" / dataset_type).mkdir(exist_ok=True)

        target_fn = this_dir_path / "data" / dataset_type / "predicted_clicks.parquet"
        if target_fn.exists():
            clicks_predicted_df = pd.read_parquet(target_fn)
        else:
            clicks_predicted_df = pd.DataFrame([
                (f"{session}_clicks", " ".join(map(str, suggest_clicks(df))))
                for session, df in tqdm.tqdm(target_df.groupby("session"), desc="predicting clicks")
            ], columns=["session_type", "labels"])
            clicks_predicted_df.to_parquet(target_fn)


        target_fn = this_dir_path / "data" / dataset_type / "predicted_buys.parquet"
        if target_fn.exists():
            buys_predicted_df = pd.read_parquet(target_fn)
        else:
            buys_predicted_df = pd.DataFrame([
                (f"{session}_", " ".join(map(str, suggest_clicks(df))))
                for session, df in tqdm.tqdm(target_df.groupby("session"), desc="predicting buys")
            ], columns=["session_type", "labels"])
            buys_predicted_df.to_parquet(target_fn)
        orders_predicted_df = buys_predicted_df.copy()
        orders_predicted_df["session_type"] = orders_predicted_df["session_type"] + "orders"
        carts_predicted_df = buys_predicted_df
        carts_predicted_df["session_type"] = carts_predicted_df["session_type"] + "carts"
        del buys_predicted_df

        total_predicted_df = pd.concat([clicks_predicted_df, orders_predicted_df, carts_predicted_df])
        total_predicted_df.to_csv(f"{dataset_type}_predictions.csv", index=False)


    # run("validation")
    # cv_score_dict = ors.validating.validate("validation_predictions.csv", days=7)
    # total_cv = cv_score_dict.pop("total")
    # print(f"CV: {total_cv:.4f} ({{{', '.join(f'{k}: {v:.4f}' for k, v in cv_score_dict.items())}}})")

    run("test")

    # for i_seperated_aid in range(n_seperated_aid)


