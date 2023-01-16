import collections
import pathlib
import tqdm
import pandas as pd
from .. import data as _data_module


def suggest_clicks(df, top_20_clicks_dict, top_20_clicks):
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


def suggest_buys(df, top_20_buys_dict, top_20_buy2buy_dict, top_20_orders):
    unique_aids = set(df["aid"])
    df = df.loc[df["type"].isin([_data_module.all_types.index("carts"), _data_module.all_types.index("orders")])]
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
        for aid, _ in collections.Counter(co_visitation_buys_aids + co_visitation_buy2buy_aids).most_common(20)
        # -> (aid, cnt)
        if aid not in unique_aids
    ]

    # Temp
    if len(unique_aids) > 20:
        unique_aids = set(aid for i, aid in enumerate(unique_aids) if i < 20)

    # unique_aidsを最優先
    top_predicted = unique_aids.union(top_predicted_without_unique_aids[:20 - len(unique_aids)])
    # 20に達していないときは、全体で良くクリックされるものを足す
    return top_predicted.union(top_20_orders[:20 - len(top_predicted)])


def get_predictions_df(
        dataset_type, target_df,
        top_20_clicks_dict, top_20_buys_dict, top_20_buy2buy_dict,
        cache_dir_path
):
    if dataset_type in ("validation", "test"):
        pass
    else:
        raise ValueError(dataset_type)

    cache_dir_path = pathlib.Path(cache_dir_path)
    (cache_dir_path / dataset_type).mkdir(exist_ok=True)

    target_fn = cache_dir_path / dataset_type / "predicted_clicks.parquet"
    if target_fn.exists():
        clicks_predicted_df = pd.read_parquet(target_fn)
    else:
        top_20_clicks = target_df.loc[
                            target_df["type"] == _data_module.all_types.index("clicks"),
                            "aid"
                        ].value_counts().index[:20].tolist()

        clicks_predicted_df = pd.DataFrame([
            (f"{session}_clicks", " ".join(map(str, suggest_clicks(df, top_20_clicks_dict, top_20_clicks))))
            for session, df in tqdm.tqdm(target_df.groupby("session"), desc="predicting clicks")
        ], columns=["session_type", "labels"])
        clicks_predicted_df.to_parquet(target_fn)

    target_fn = cache_dir_path / dataset_type / "predicted_buys.parquet"
    if target_fn.exists():
        buys_predicted_df = pd.read_parquet(target_fn)
    else:
        top_20_orders = target_df.loc[
                            target_df["type"] == _data_module.all_types.index("orders"),
                            "aid"
                        ].value_counts().index[:20].tolist()

        buys_predicted_df = pd.DataFrame([
            (f"{session}_", " ".join(map(str, suggest_buys(df, top_20_buys_dict, top_20_buy2buy_dict, top_20_orders))))
            for session, df in tqdm.tqdm(target_df.groupby("session"), desc="predicting buys")
        ], columns=["session_type", "labels"])
        buys_predicted_df.to_parquet(target_fn)
    orders_predicted_df = buys_predicted_df.copy()
    orders_predicted_df["session_type"] = orders_predicted_df["session_type"] + "orders"
    carts_predicted_df = buys_predicted_df
    carts_predicted_df["session_type"] = carts_predicted_df["session_type"] + "carts"
    del buys_predicted_df

    total_predicted_df = pd.concat([clicks_predicted_df, orders_predicted_df, carts_predicted_df])
    return total_predicted_df
