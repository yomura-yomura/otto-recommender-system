import warnings
import polars as pl
import tqdm
import argparse
import numpy as np


def read_sub(path):
    """a helper function for loading and preprocessing submissions"""
    return (
        pl.read_csv(path)
        .with_column(pl.col("labels").str.strip())
        .with_column(pl.col("labels").str.split(by=" "))
        .explode("labels")
        .rename({"labels": "aid"})
        .with_column(pl.col("aid").cast(pl.UInt32))
    )


def add_vote_col(df, weight):
    return (
        df
        .with_column(pl.lit(weight).alias("vote"))
        .with_column(pl.col("vote").cast(pl.UInt8))
    )


def add_rank_weight_col(df, weight, max_session_type_count):
    return (
        df
        .with_column(pl.lit(-1).alias("rank_weight"))
        .with_column(pl.col("rank_weight").cumsum().over("session_type"))
        .with_column((pl.col("rank_weight") + max_session_type_count + 1) * weight)
    )


def main(submission_csv_paths, ensemble_type, n_top, weights, output):
    if weights is None:
        weights = [1] * len(submission_csv_paths)
    else:
        if len(weights) == len(submission_csv_paths):
            weights = args.weights
        else:
            raise ValueError("option -w/--weights must have the same length values as the given submission_csv_paths")
    output = output.format(weights=weights, ensemble_type=ensemble_type, n_top=n_top)

    print("* Given Parameters:")
    print("{:<25}  =  {:<}".format("submission_csv_paths", str(submission_csv_paths)))
    print("{:<25}  =  {:<}".format("weights", str(weights)))
    print("{:<25}  =  {:<}".format("ensemble_type", str(ensemble_type)))
    print("{:<25}  =  {:<}".format("n_top", str(n_top)))
    print("{:<25}  =  {:<}".format("output", str(output)))
    print()

    if n_top is None:
        if ensemble_type == "voting":
            warnings.warn("n_top should be given due to the bad performance on scoring", UserWarning)

    print("(1/5) Reading submission csv")
    subs = list(map(read_sub, tqdm.tqdm(submission_csv_paths, desc="Reading the given submission files")))

    if n_top is not None:
        for i in range(len(subs)):
            subs[i] = subs[i].groupby("session_type").head(n_top)

    if ensemble_type == "voting":
        subs = [
            add_vote_col(sub, weight)
            for sub, weight in zip(tqdm.tqdm(subs, desc="Adding vote col"), weights)
        ]
    elif ensemble_type == "rank-weighting":
        max_count = max(sub.groupby("session_type").count().max()[0, "count"] for sub in subs)
        subs = [
            add_rank_weight_col(sub, weight, max_count)
            for sub, weight in zip(tqdm.tqdm(subs, desc="Adding rank-weight col"), weights)
        ]

        # rank_weights checks
        max_rank_weights = np.array([sub["rank_weight"].max() / weight for sub, weight in zip(subs, weights)])
        assert len(np.unique(max_rank_weights)) == 1
        if n_top is not None:
            assert np.all(max_rank_weights == n_top)
        min_rank_weights = np.array([sub["rank_weight"].min() / weight for sub, weight in zip(subs, weights)])
        assert np.all(min_rank_weights == 1)
    else:
        raise NotImplementedError(f"ensemble_type == {ensemble_type}")

    # given sub-records-length checks
    given_records_lengths = [
        len(sub["session_type"].unique())
        for sub in tqdm.tqdm(subs, desc="Checking records-length consistency of the given submissions")
    ]
    if len(set(given_records_lengths)) != 1:
        raise RuntimeError(f"Inconsistent records length for the given submissions: {given_records_lengths}")

    main_sub, *left_subs = subs

    # given sub-column-value checks
    if not all(
            main_sub["session_type"].is_in(sub["session_type"]).to_numpy().all()
            for sub in tqdm.tqdm(left_subs, desc="Checking session_type consistency of the given submissions")
    ):
        raise RuntimeError("Inconsistent session_type for the given submissions")

    print("(2/5) Outer-join operation")
    for i_sub, sub in enumerate(tqdm.tqdm(left_subs, desc="Outer-join operating")):
        main_sub = main_sub.join(sub, how="outer", on=["session_type", "aid"], suffix=f"_right{i_sub + 1}")

    main_sub = main_sub.fill_null(0)

    if ensemble_type == "voting":
        print("(3/5) Calculating total votes")

        target_cols = [col for col in main_sub.columns if col.startswith("vote")]
        main_sub = (
            main_sub
            .with_column(pl.sum(target_cols).alias("vote_sum"))
            .drop(target_cols)
            .sort(by="vote_sum")
            .reverse()
        )
    elif ensemble_type == "rank-weighting":
        print("(3/5) Calculating total rank_weight")

        target_cols = [col for col in main_sub.columns if col.startswith("rank_weight")]
        main_sub = (
            main_sub
            .with_column(pl.sum(target_cols).alias("rank_weight_sum"))
            .drop(target_cols)
            .sort(by="rank_weight_sum")
            .reverse()
        )
    else:
        raise NotImplementedError(f"ensemble_type == {ensemble_type}")

    print("(4/5) Creating submission-format csv")
    preds = (
        main_sub
        .groupby("session_type")
        .agg([pl.col("aid").head(20).alias("labels")])
        .with_column(pl.col("labels").apply(lambda labels: " ".join(map(str, labels))))
    )

    if len(preds) != 5015409:
        warnings.warn("""
        if the given submission is for submissions, submission file must have 5015409 rows.
        """, UserWarning)

    target_submission_csv_path = output
    print(f"(5/5) Exporting as {target_submission_csv_path}")
    preds.write_csv(target_submission_csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_csv_paths", nargs="+")
    parser.add_argument("-w", "--weights", nargs="?", action="append", default=None, type=float)
    parser.add_argument("-e", "--ensemble-type", choices=["voting", "rank-weighting"], default="rank-weighting")
    parser.add_argument("-n", "--n-top", default=40, type=int, help="各sessionで使う上位n個のラベルの指定。Noneだと全部使う。")
    parser.add_argument("-o", "--output", default="{ensemble_type}_submission.csv")
    args = parser.parse_args()

    main(args.submission_csv_paths, args.ensemble_type, args.n_top, args.weights, args.output)
