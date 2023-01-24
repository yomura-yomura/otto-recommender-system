import warnings
import polars as pl
import tqdm
import argparse


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
        .with_column(pl.lit(-weight).alias("rank_weight"))
        .with_column(pl.col("rank_weight").cumsum().over("session_type"))
        .with_column(pl.col("rank_weight") + max_session_type_count)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_csv_paths", nargs="+")
    parser.add_argument("-w", "--weights", nargs="?", action="append", default=None, type=float)
    parser.add_argument("-e", "--ensemble-type", required=True, choices=["voting", "rank-weighting"])
    parser.add_argument("-n", "--n-top", default=None, type=int, help="各sessionで使う上位n個のラベルの指定。Noneだと全部使う。")
    args = parser.parse_args()

    submission_csv_paths = args.submission_csv_paths
    ensemble_type = args.ensemble_type
    n_top = args.n_top
    if args.weights is None:
        weights = [1] * len(submission_csv_paths)
    else:
        if len(args.weights) == len(submission_csv_paths):
            weights = args.weights
        else:
            raise ValueError("option -w/--weights must have the same length values as the given submission_csv_paths")

    print("* Given Parameters:")
    print("{:<25}  =  {:<}".format("submission_csv_paths", str(submission_csv_paths)))
    print("{:<25}  =  {:<}".format("weights", str(weights)))
    print("{:<25}  =  {:<}".format("ensemble_type", str(ensemble_type)))
    print("{:<25}  =  {:<}".format("n_top", str(n_top)))
    print()

    if n_top is None:
        if ensemble_type == "voting":
            warnings.warn("n_top should be given due to the bad performance on scoring", UserWarning)

    print("(1/5) Reading submission csv")
    subs = list(map(read_sub, tqdm.tqdm(submission_csv_paths, desc="Reading the given submission files")))

    if n_top is not None:
        for sub in subs:
            sub = sub.groupby("session_type").head(n_top)

    if ensemble_type == "voting":
        subs = [add_vote_col(sub, weight) for sub, weight in zip(subs, weights)]
    elif ensemble_type == "rank-weighting":
        max_count = max(sub.groupby("session_type").count().max()[0, "count"] for sub in subs)
        subs = [add_rank_weight_col(sub, weight, max_count) for sub, weight in zip(subs, weights)]
    else:
        raise NotImplementedError(f"ensemble_type == {ensemble_type}")

    main_sub, *left_subs = subs

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

    assert len(preds) == 5015409, "Submission must have 5015409 rows"

    target_submission_csv_path = f"{ensemble_type}_submission.csv"
    print(f"(5/5) Exporting as {target_submission_csv_path}")
    preds.write_csv(target_submission_csv_path)
