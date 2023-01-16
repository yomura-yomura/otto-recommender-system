import pathlib
import otto_recommender_system as ors
import otto_recommender_system.data
import otto_recommender_system.validating
import otto_recommender_system.co_visitation_matrixes
import numpy as np
import argparse


this_dir_path = pathlib.Path(__file__).resolve().parent
saved_dir_path = this_dir_path / "data" / "co-visitation-matrix"
saved_dir_path.mkdir(exist_ok=True, parents=True)


def past_time_weight(df):
    return 1 + 3 * (df["ts_x"] - all_train_df["ts"].min()) / (all_train_df["ts"].max() - all_train_df["ts"].min())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seperated-aid", default=8, type=int)
    parser.add_argument("--max-memory-gb-for-each-split-aid", default=1, type=int)
    args = parser.parse_args()
    
    train_df, valid_df, test_df, valid_labels_df = ors.data.get_datasets(
        type_of_tidy_data="parquet", return_valid_labels=True
    )
    all_train_df = ors.data.get_pd_tidy_data("train")

    clicks_cv_matrix = ors.co_visitation_matrixes.CoVisitationMatrix(
        all_train_df, "clicks", saved_dir_path,
        weight_func=past_time_weight,
        n_seperated_aid=args.n_seperated_aid,
        max_memory_gb_for_each_split_aid=args.max_memory_gb_for_each_split_aid
    )
    clicks_cv_matrix.make(max_timedelta=np.timedelta64(1, "D"))
    top_20_clicks_dict = clicks_cv_matrix.get_dict()

    carts_orders_cv_matrix = ors.co_visitation_matrixes.CoVisitationMatrix(
        all_train_df, "carts_orders", saved_dir_path,
        weight_func=lambda _: {0: 1, 1: 6, 2: 3},
        n_seperated_aid=args.n_seperated_aid,
        max_memory_gb_for_each_split_aid=args.max_memory_gb_for_each_split_aid
    )
    carts_orders_cv_matrix.make(max_timedelta=np.timedelta64(1, "D"))
    top_20_buys_dict = carts_orders_cv_matrix.get_dict()

    buy2buy_cv_matrix = ors.co_visitation_matrixes.CoVisitationMatrix(
        all_train_df, "buy2buy", saved_dir_path,
        types_to_use=["carts", "orders"],
        n_seperated_aid=args.n_seperated_aid,
        max_memory_gb_for_each_split_aid=args.max_memory_gb_for_each_split_aid
    )
    buy2buy_cv_matrix.make(max_timedelta=np.timedelta64(14, "D"))
    top_20_buy2buy_dict = buy2buy_cv_matrix.get_dict()

    print("* Validation")
    valid_predictions_df = ors.co_visitation_matrixes.get_predictions_df(
        "validation", valid_df,
        top_20_clicks_dict, top_20_buys_dict, top_20_buy2buy_dict,
        saved_dir_path
    )
    valid_predictions_df.to_csv(f"validation_predictions.csv", index=False)
    cv_score_dict = ors.validating.validate("validation_predictions.csv", days=7)
    total_cv = cv_score_dict.pop("total")
    print(f"CV: {total_cv:.4f} ({{{', '.join(f'{k}: {v:.4f}' for k, v in cv_score_dict.items())}}})")

    print("* Test")
    test_predictions_df = ors.co_visitation_matrixes.get_predictions_df(
        "test", test_df,
        top_20_clicks_dict, top_20_buys_dict, top_20_buy2buy_dict,
        saved_dir_path
    )
    test_predictions_df.to_csv(f"test_predictions.csv", index=False)
