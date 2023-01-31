import argparse

import pandas as pd
import polars as pl
import pathlib


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base_submission_csv", type=pathlib.Path)
    parser.add_argument("submission_csv_to_add", type=pathlib.Path)
    parser.add_argument("type", choices=["clicks", "carts", "orders"])
    args = parser.parse_args()

    base_df = pl.read_csv(args.base_submission_csv)
    n_original_base_records = len(base_df)

    concat_df = pd.DataFrame()

    # assert len(args.submission_csv_to_add) == len(args.type)
    for submission_csv_to_add, event_type in zip([args.submission_csv_to_add], [args.type]):
        to_add_df = pl.read_csv(submission_csv_to_add)

        to_add_df = to_add_df.filter(to_add_df["session_type"].str.ends_with(f"_{event_type}"))
        filtered_base_df = base_df.filter(~base_df["session_type"].str.ends_with(f"_{event_type}"))
        assert len(to_add_df) + len(filtered_base_df) == n_original_base_records

        concat_df = pl.concat([filtered_base_df, to_add_df])
        assert len(concat_df) == n_original_base_records
        assert len(concat_df) == len(concat_df["session_type"].unique())
        base_df = concat_df

    fn = f"concat_{args.base_submission_csv.stem}_on_{'_'.join([args.type])}.csv"
    print(f"* Saving as {fn}")
    concat_df.to_pandas().to_csv(fn, index=False)
