import otto_recommender_system as ors
import otto_recommender_system.data
import cudf
import tqdm
import pandas as pd
import otto_recommender_system.validating


if __name__ == "__main__":
    train_df, valid_df, test_df = ors.data.get_datasets(type_of_tidy_data="parquet")

    train_df["type"] = train_df["type"].str.decode("utf-8")
    train_cudf = cudf.from_pandas(train_df)

    unique_types = train_cudf["type"].unique().to_numpy("U8")

    count_cudf_series_dict = {
        type_: train_cudf[train_cudf["type"] == type_]["aid"].value_counts()
        for type_ in unique_types
    }

    print([(v.cumsum() / v.sum()).iloc[:20] for k, v in count_cudf_series_dict.items()])

    aid_top20_dict = {
        type_: series.iloc[:20].index.to_numpy()
        for type_, series in count_cudf_series_dict.items()
    }

    def get_submission_df(target_df):
        records = []
        for session in tqdm.tqdm(target_df["session"].unique(), desc="making submission df"):
            records.extend([
                {
                    "session_type": f"{session}_{type_}",
                    "labels": " ".join(map(str, aid_top20_dict[type_]))
                }
                for type_ in unique_types
            ])

        return pd.DataFrame(records)


    submission_df = get_submission_df(valid_df)
    submission_df.to_csv("validation_predictions.csv", index=False)

    cv_score_dict = ors.validating.validate("validation_predictions.csv", days=7)
    total_cv = cv_score_dict.pop("total")
    print(f"CV: {total_cv:.4f} ({{{', '.join(f'{k}: {v:.4f}' for k, v in cv_score_dict.items())}}})")

    submission_df = get_submission_df(test_df)
    submission_df.to_csv("test_predictions.csv", index=False)

    # count_cudf_series = train_cudf.groupby("session")["session"].count()
    # matched_train_cudf = train_cudf[
    #     train_cudf["session"].isin(count_cudf_series[10 < count_cudf_series].index)
    # ]

    # aid_count_cudf_series = train_cudf["aid"].value_counts()
    # aid_count_cudf_series.index = aid_count_cudf_series.index.astype(str)
    #
    # import plotly_utility.express as pux
    # import plotly.express as px
    # px.bar(
    #     aid_count_cudf_series.reset_index().rename(
    #         columns={"index": "aid", "aid": "count"}
    #     ).iloc[:20],
    #     x="aid", y="count"
    # ).show()
    #
    # aid_count_cudf_series.cumsum() / aid_count_cudf_series.sum()
    # print()
