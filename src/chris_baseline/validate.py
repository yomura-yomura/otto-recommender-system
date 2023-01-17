import pathlib
import otto_recommender_system as ors
import otto_recommender_system.data
import otto_recommender_system.validating
import otto_recommender_system.co_visitation_matrixes
import numpy as np
import argparse
from src.chris_baseline.run import main


this_dir_path = pathlib.Path(__file__).resolve().parent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seperated-aid", default=8, type=int)
    parser.add_argument("--max-memory-gb-for-each-split-aid", default=1, type=int)
    args = parser.parse_args()

    train_df, valid_df, test_df, valid_labels_df = ors.data.get_datasets(
        type_of_tidy_data="parquet", return_valid_labels=True
    )

    (
        top_20_clicks_dict, top_20_buys_dict, top_20_buy2buy_dict,
        saved_dir_path
    ) = main("all-train", train_df, args.n_seperated_aid, args.max_memory_gb_for_each_split_aid)

    valid_predictions_df = ors.co_visitation_matrixes.get_predictions_df(
        "validation", valid_df,
        top_20_clicks_dict, top_20_buys_dict, top_20_buy2buy_dict,
        saved_dir_path
    )
    valid_predictions_df.to_csv(this_dir_path / f"validation_predictions.csv", index=False)
    cv_score_dict = ors.validating.validate(this_dir_path / "validation_predictions.csv", days=7)
    total_cv = cv_score_dict.pop("total")
    print(f"CV: {total_cv:.4f} ({{{', '.join(f'{k}: {v:.4f}' for k, v in cv_score_dict.items())}}})")
