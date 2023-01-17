import pathlib
import otto_recommender_system as ors
import otto_recommender_system.data
import otto_recommender_system.validating
import otto_recommender_system.co_visitation_matrixes
import numpy as np
import argparse


this_dir_path = pathlib.Path(__file__).resolve().parent


def main(target, df_to_train, n_seperated_aid, max_memory_gb_for_each_split_aid):
    saved_dir_path = this_dir_path / "data" / "co-visitation-matrix" / target
    saved_dir_path.mkdir(exist_ok=True, parents=True)

    def past_time_weight(df):
        return 1 + 3 * (df["ts_x"] - df_to_train["ts"].min()) / (df_to_train["ts"].max() - df_to_train["ts"].min())

    clicks_cv_matrix = ors.co_visitation_matrixes.CoVisitationMatrix(
        df_to_train, "clicks", saved_dir_path,
        weight_func=past_time_weight,
        n_seperated_aid=n_seperated_aid,
        max_memory_gb_for_each_split_aid=max_memory_gb_for_each_split_aid
    )
    clicks_cv_matrix.make(max_timedelta=np.timedelta64(1, "D"))
    top_20_clicks_dict = clicks_cv_matrix.get_dict()

    carts_orders_cv_matrix = ors.co_visitation_matrixes.CoVisitationMatrix(
        df_to_train, "carts_orders", saved_dir_path,
        weight_func=lambda _: {0: 1, 1: 6, 2: 3},
        n_seperated_aid=n_seperated_aid,
        max_memory_gb_for_each_split_aid=max_memory_gb_for_each_split_aid
    )
    carts_orders_cv_matrix.make(max_timedelta=np.timedelta64(1, "D"))
    top_20_buys_dict = carts_orders_cv_matrix.get_dict()

    buy2buy_cv_matrix = ors.co_visitation_matrixes.CoVisitationMatrix(
        df_to_train, "buy2buy", saved_dir_path,
        types_to_use=["carts", "orders"],
        n_seperated_aid=n_seperated_aid,
        max_memory_gb_for_each_split_aid=max_memory_gb_for_each_split_aid
    )
    buy2buy_cv_matrix.make(max_timedelta=np.timedelta64(14, "D"))
    top_20_buy2buy_dict = buy2buy_cv_matrix.get_dict()

    return (
        top_20_clicks_dict, top_20_buys_dict, top_20_buy2buy_dict,
        saved_dir_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seperated-aid", default=8, type=int)
    parser.add_argument("--max-memory-gb-for-each-split-aid", default=1, type=int)
    args = parser.parse_args()

    all_train_df = ors.data.get_pd_tidy_data("train")
    test_df = ors.data.get_pd_tidy_data("test")

    (
        top_20_clicks_dict, top_20_buys_dict, top_20_buy2buy_dict,
        saved_dir_path
    ) = main("all-train", all_train_df, args.n_seperated_aid, args.max_memory_gb_for_each_split_aid)

    test_predictions_df = ors.co_visitation_matrixes.get_predictions_df(
        "test", test_df,
        top_20_clicks_dict, top_20_buys_dict, top_20_buy2buy_dict,
        saved_dir_path
    )
    test_predictions_df.to_csv(this_dir_path / f"test_predictions.csv", index=False)
