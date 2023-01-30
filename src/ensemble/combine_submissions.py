import argparse
import polars as pl
import pathlib


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base_submission_csv", type=pathlib.Path)
    parser.add_argument("submission_csv_to_add", type=pathlib.Path)
    parser.add_argument("type", choices=["clicks", "carts", "orders"])
    args = parser.parse_args()

    base_df = pl.read_csv(args.base_submission_csv)
    to_add_df = pl.read_csv(args.submission_csv_to_add)
    event_type = args.type

    n_original_base_records = len(base_df)

    to_add_df = to_add_df.filter(to_add_df["session_type"].str.ends_with(f"_{event_type}"))
    base_df = base_df.filter(~base_df["session_type"].str.ends_with(f"_{event_type}"))
    assert len(to_add_df) + len(base_df) == n_original_base_records

    concat_df = pl.concat([base_df, to_add_df])
    assert len(concat_df) == n_original_base_records
    assert len(concat_df) == len(concat_df["session_type"].unique())

    fn = f"concat_{args.base_submission_csv.stem}_on_{event_type}.csv"
    print(f"* Saving as {fn}")
    concat_df.to_pandas().to_csv(fn, index=False)







