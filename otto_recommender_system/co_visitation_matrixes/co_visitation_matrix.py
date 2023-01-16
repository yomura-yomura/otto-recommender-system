import pathlib

import cudf
from .. import data as _data_module
import numpy as np
from typing import Optional, List, Union, Dict, Callable
import tqdm
import pandas as pd
import json


class CoVisitationMatrix:
    def __init__(
            self, all_train_df, dirname, cache_dir_path,
            max_memory_gb_for_each_split_aid: Union[float, int] = 1,
            n_seperated_aid=8,
            types_to_use: Optional[List[Union[int, str]]] = None,
            weight_func: Optional[Callable[[cudf.DataFrame], Dict[Union[int, str], Union[int, float]]]] = None
    ):
        all_aid = np.unique(all_train_df["aid"])
        assert all_aid.min() == 0
        assert all_aid.max() == len(all_aid) - 1

        all_sessions, indices_in_all_sessions = np.unique(all_train_df["session"], return_index=True)

        self.dirname = pathlib.Path(cache_dir_path) / dirname
        self.dirname.mkdir(exist_ok=True)

        def _validate_type_as_int(type_):
            return _data_module.all_types.index(type_) if isinstance(type_, str) else int(type_)

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
                (max_memory_gb_for_each_split_aid * 1e9),
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

    def make(self, max_timedelta: np.timedelta64):
        for i_seperated_aid in range(self.n_seperated_aid):
            print(f"({i_seperated_aid + 1}/{self.n_seperated_aid}) split aid at {self.dirname.name}")
            target_fn = self.dirname / f"{i_seperated_aid}.parquet"
            if target_fn.exists():
                continue

            self.total_weight_cudf: Optional[cudf.DataFrame] = None
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
