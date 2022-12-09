import numpy as np
import pandas as pd
import pathlib
import tqdm
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


project_root_path = pathlib.Path(__file__).resolve().parent.parent.parent
official_data_path = project_root_path / "data" / "otto-recommender-system"
if not official_data_path.exists():
    raise FileNotFoundError(official_data_path)


def get_n_lines():
    with open(official_data_path / "train.jsonl") as f:
        return sum(1 for _ in f)



def event_dict_to_np_array(event_dict: dict):
    assert tuple(event_dict.keys()) == ("aid", "ts", "type")
    return event_dict["aid"], event_dict["ts"], event_dict["type"]


max_n_events = 500
n_records_dict = {
    "train": 12_899_779,
    "test": 1_671_803
}


def get_np_tidy_data(
        dataset_type: str,
        data_path=official_data_path,
        tidy_data_path=project_root_path / "data" / "otto-recommender-system-tidy-data"
):
    if dataset_type not in ("train", "test"):
        raise ValueError("dataset_type must be 'train' or 'test'")

    np_tidy_data_path = tidy_data_path / "npz"
    np_tidy_data_path.mkdir(exist_ok=True)
    np_tidy_data_cachefile_path = np_tidy_data_path / f"{dataset_type}.npz"

    if np_tidy_data_cachefile_path.exists():
        logger.info(f"loading {np_tidy_data_cachefile_path}")
        return np.load(np_tidy_data_cachefile_path)["arr_0"]

    logger.info(f"creating {np_tidy_data_cachefile_path}")

    np_tidy_data = np.empty(
        0,
        dtype=[
            ("session", "i4"),
            ("aid", "i4"),
            ("ts", "M8[ms]"),
            ("type", "S6")
        ]
    )
    chunk_size = 100000

    target_fn = data_path / f"{dataset_type}.jsonl"
    if not target_fn.exists():
        target_fn = data_path / f"{dataset_type}_sessions.jsonl"
    if not target_fn.exists():
        raise FileNotFoundError(data_path / f"[{dataset_type}/{dataset_type}_sessions].jsonl")
    
    chunks = pd.read_json(target_fn, lines=True, chunksize=chunk_size)

    for chunk in tqdm.tqdm(chunks, total=int(np.ceil(n_records_dict["train"] / chunk_size))):
        assert np.all(chunk.columns == ["session", "events"])

        np_tidy_data = np.concatenate([
            np_tidy_data,
            np.array([
                (session, *event_dict_to_np_array(event))
                for session, events in zip(chunk["session"], chunk["events"])
                for event in events
            ], dtype=np_tidy_data.dtype)
        ])
        # print(f"{np_tidy_data.nbytes // 1024 ** 3:.4f} GB")


    # all_types = np.array(["clicks", "carts", "orders"], dtype="S")

    # a = np.empty(
    #     len(np_tidy_data),
    #     dtype=[
    #         ("session", "i4"),
    #         ("aid", "i4"),
    #         ("ts", "M8[ms]"),
    #         *(
    #             (f"is_{type_.decode()}", "?")
    #             for type_ in all_types
    #         )
    #     ]
    # )


    np.savez_compressed(np_tidy_data_cachefile_path, np_tidy_data)
    return np_tidy_data


def get_pd_tidy_data(
        dataset_type: str,
        data_path=official_data_path,
        tidy_data_path=project_root_path / "data" / "otto-recommender-system-tidy-data"
):
    if dataset_type not in ("train", "test"):
        raise ValueError("dataset_type must be 'train' or 'test'")

    pd_tidy_data_path = tidy_data_path / "parquet"
    pd_tidy_data_path.mkdir(exist_ok=True)
    pd_tidy_data_cachefile_path = pd_tidy_data_path / f"{dataset_type}.parquet"

    if pd_tidy_data_cachefile_path.exists():
        logger.info(f"loading {pd_tidy_data_cachefile_path}")
        return pd.read_parquet(pd_tidy_data_cachefile_path)

    logger.info(f"creating {pd_tidy_data_cachefile_path}")
    np_tidy_data = get_np_tidy_data(dataset_type, data_path, tidy_data_path)
    df = pd.DataFrame(np_tidy_data)
    df.to_parquet(pd_tidy_data_cachefile_path)

    return df



def get_datasets(days=7, type_of_tidy_data="npz"):
    dirname = f"{days}days"

    data_path = project_root_path / "data" / "otto-train-and-test-data-for-local-validation" / dirname / "jsonl"
    tidy_data_path = project_root_path / "data" / "otto-train-and-test-data-for-local-validation" / dirname
    
    if type_of_tidy_data == "npz":
        train_tidy_data = get_np_tidy_data("train", data_path, tidy_data_path)
        valid_tidy_data = get_np_tidy_data("test", data_path, tidy_data_path)
        test_tidy_data = get_np_tidy_data("test")
    elif type_of_tidy_data == "parquet":
        train_tidy_data = get_pd_tidy_data("train", data_path, tidy_data_path)
        valid_tidy_data = get_pd_tidy_data("test", data_path, tidy_data_path)
        test_tidy_data = get_pd_tidy_data("test")
    else:
        raise ValueError(type_of_tidy_data)

    return train_tidy_data, valid_tidy_data, test_tidy_data

