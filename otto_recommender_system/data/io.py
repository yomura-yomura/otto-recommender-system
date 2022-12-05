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


def get_np_tidy_data(dataset_type: str):
    if dataset_type not in ("train", "test"):
        raise ValueError("dataset_type must be 'train' or 'test'")

    np_tidy_data_path = project_root_path / "data" / "np_tidy_data"
    np_tidy_data_path.mkdir(exist_ok=True)

    np_tidy_data_cachefile_path = np_tidy_data_path / f"{dataset_type}.npz"

    if np_tidy_data_cachefile_path.exists():
        logger.info(f"loading {np_tidy_data_cachefile_path}")
        return np.load(np_tidy_data_cachefile_path)["arr_0"]

    logger.info("creating np_tidy_data_cachefile_path")

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
    chunks = pd.read_json(official_data_path / f"{dataset_type}.jsonl", lines=True, chunksize=chunk_size)

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


train_np_tidy_data = get_np_tidy_data("train")
test_np_tidy_data = get_np_tidy_data("test")

# data = np.concatenate([train_np_tidy_data, test_np_tidy_data])
# df = pd.DataFrame(data)

