import polars as pl
import numpy as np
import tqdm
import argparse
from predict import read_sub


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("predicted_file")
    parser.add_argument("type", nargs="*", choices=[[], "clicks", "carts", "orders"])
    args = parser.parse_args()

    event_type = args.type
    if len(event_type) == 0:
        event_type = ["clicks", "carts", "orders"]

    print("* Given Parameters:")
    print("{:<25}  =  {:<}".format("type", str(event_type)))
    print()

    print(f"* Reading {args.predicted_file}")
    df = read_sub(args.predicted_file)
    df = (
        df
        .with_column(pl.col("session_type").str.split(by="_"))
        .select([
            pl.col("session_type").arr.get(0).alias("session").cast(pl.Int64),
            pl.col("session_type").arr.get(1).alias("type"),
            pl.col("aid")
        ])
    )

    df = df.groupby(["session", "type"]).head(20)
    df = df.unique(subset=["session", "aid"])

    print("* Reading test_labels.parquet")
    test_labels_df = pl.read_parquet("../../shaoroon/01_Data/add/local-validation_chris/test_labels.parquet")
    test_labels_df = (
        test_labels_df
        .explode("ground_truth")
        .with_column(pl.col("ground_truth").cast(pl.UInt32))
    )

    assert (df["session"].is_in(test_labels_df["session"])).to_numpy().all()
    test_labels_df = test_labels_df.filter(test_labels_df["type"].is_in(event_type))
    df = df.join(test_labels_df, on=["session", "type"], how="semi")

    inner_joined_df = (
        test_labels_df
        .join(df, left_on=["session", "type", "ground_truth"], right_on=["session", "type", "aid"], how="inner")
    )

    hits_df = inner_joined_df.groupby(["session", "type"]).count()
    gt_count_df = test_labels_df.groupby(["session", "type"]).agg(pl.count().clip(0, 20))

    recall = hits_df["count"].sum() / gt_count_df["count"].sum()
    print(f"orders recall = {recall:}")
