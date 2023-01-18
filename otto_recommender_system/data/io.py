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


def get_n_lines(file_path):
    with open(file_path) as f:
        return sum(1 for _ in f)



def event_dict_to_np_array(event_dict: dict):
    assert tuple(event_dict.keys()) == ("aid", "ts", "type")
    return event_dict["aid"], event_dict["ts"], all_types.index(event_dict["type"])


def labels_dict_to_np_array(labels_dict: dict):
    return [
        (aid, np.datetime64("NaT"), all_types.index(type_))
        for type_ in all_types
        if type_ in labels_dict.keys()
        for aid in (
            labels_dict[type_] if isinstance(labels_dict[type_], list) else [labels_dict[type_]]
        )
    ]


max_n_events = 500
all_types = ["clicks", "carts", "orders"]


def get_np_tidy_data(
        dataset_type: str,
        data_path=official_data_path,
        tidy_data_path=project_root_path / "data" / "otto-recommender-system-tidy-data"
):
    if dataset_type not in ("train", "test", "test_labels"):
        raise ValueError("dataset_type must be 'train', 'test' or 'test_labels'")

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
            ("type", "i1")
        ]
    )
    chunk_size = 100_000

    target_fn = data_path / f"{dataset_type}.jsonl"
    if not target_fn.exists():
        target_fn = data_path / f"{dataset_type}_sessions.jsonl"
    if not target_fn.exists():
        raise FileNotFoundError(data_path / f"[{dataset_type}/{dataset_type}_sessions].jsonl")
    
    chunks = pd.read_json(target_fn, lines=True, chunksize=chunk_size)
    n_records = get_n_lines(target_fn)

    for chunk in tqdm.tqdm(chunks, total=int(np.ceil(n_records / chunk_size))):
        if dataset_type == "test_labels":
            assert np.all(chunk.columns == ["session", "labels"])
            np_tidy_data = np.concatenate([
                np_tidy_data,
                np.array([
                    (session, *args)
                    for session, labels_dict in zip(chunk["session"], chunk["labels"])
                    for args in labels_dict_to_np_array(labels_dict)
                ], dtype=np_tidy_data.dtype)
            ])
        else:
            assert np.all(chunk.columns == ["session", "events"])
            np_tidy_data = np.concatenate([
                np_tidy_data,
                np.array([
                    (session, *event_dict_to_np_array(event))
                    for session, events in zip(chunk["session"], chunk["events"])
                    for event in events
                ], dtype=np_tidy_data.dtype)
            ])

    np.savez_compressed(np_tidy_data_cachefile_path, np_tidy_data)
    return np_tidy_data


def get_pd_tidy_data(
        dataset_type: str,
        data_path=official_data_path,
        tidy_data_path=project_root_path / "data" / "otto-recommender-system-tidy-data"
):
    if dataset_type not in ("train", "test", "test_labels"):
        raise ValueError("dataset_type must be 'train', 'test' or 'test_labels'")

    pd_tidy_data_path = tidy_data_path / "parquet"
    pd_tidy_data_path.mkdir(exist_ok=True)
    pd_tidy_data_cachefile_path = pd_tidy_data_path / f"{dataset_type}.parquet"

    if pd_tidy_data_cachefile_path.exists():
        logger.info(f"loading {pd_tidy_data_cachefile_path}")
        df = pd.read_parquet(pd_tidy_data_cachefile_path)
        if np.issubdtype(df["type"].dtype, np.object_):
            print("O -> i4")
            df["type"] = df["type"].str.decode("utf-8").map(lambda t: all_types.index(t))
            df.to_parquet(pd_tidy_data_cachefile_path)
        return df

    logger.info(f"creating {pd_tidy_data_cachefile_path}")
    np_tidy_data = get_np_tidy_data(dataset_type, data_path, tidy_data_path)
    df = pd.DataFrame(np_tidy_data)
    df.to_parquet(pd_tidy_data_cachefile_path)

    return df


def get_datasets(days=7, weeks=4, type_of_tidy_data="npz", return_train=True, return_valid=True, return_test=True, return_valid_labels=False):
    dirname = f"{days}days-of-{weeks}weeks"

    data_path = project_root_path / "data" / "otto-train-and-test-data-for-local-validation" / dirname / "jsonl"
    tidy_data_path = project_root_path / "data" / "otto-train-and-test-data-for-local-validation" / dirname

    valid_labels_tidy_data = None

    if type_of_tidy_data == "npz":
        train_tidy_data = get_np_tidy_data("train", data_path, tidy_data_path)
        valid_tidy_data = get_np_tidy_data("test", data_path, tidy_data_path)
        test_tidy_data = get_np_tidy_data("test")
        if return_valid_labels:
            valid_labels_tidy_data = get_np_tidy_data("test_labels", data_path, tidy_data_path)
    elif type_of_tidy_data == "parquet":
        train_tidy_data = get_pd_tidy_data("train", data_path, tidy_data_path)
        valid_tidy_data = get_pd_tidy_data("test", data_path, tidy_data_path)
        test_tidy_data = get_pd_tidy_data("test")
        if return_valid_labels:
            valid_labels_tidy_data =  get_pd_tidy_data("test_labels", data_path, tidy_data_path)
    else:
        raise ValueError(type_of_tidy_data)


    ret = []
    if return_train:
        ret += [train_tidy_data]
    if return_valid:
        ret += [valid_tidy_data]
    if return_test:
        ret += [test_tidy_data]
    if return_valid_labels:
        ret += [valid_labels_tidy_data]

    return ret
