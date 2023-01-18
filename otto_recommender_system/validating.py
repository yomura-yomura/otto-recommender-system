import pathlib
import sys
import asyncio


project_root_path = pathlib.Path(__file__).resolve().parent.parent
competition_host_git_repos_path = project_root_path / "recsys-dataset"
train_test_local_validation_data_path = project_root_path / "data" / "otto-train-and-test-data-for-local-validation"
if not train_test_local_validation_data_path.exists():
    raise FileNotFoundError(train_test_local_validation_data_path)



async def _validate_coroutine(prediction_csv_path, days):
    prediction_csv_path = pathlib.Path(prediction_csv_path)
    assert prediction_csv_path.exists()
    prediction_csv_path = prediction_csv_path.resolve()

    test_labels_path = train_test_local_validation_data_path / f"{days}days" / "jsonl" / "test_labels.jsonl"

    await asyncio.create_subprocess_exec(
        *[
            "python", "-m", "pipenv", "sync"
        ],
        cwd=competition_host_git_repos_path
    )

    proc = await asyncio.create_subprocess_exec(
        *[
            "python", "-m", "pipenv", "run", "python", "-m", "src.evaluate",
            "--test-labels", f"{test_labels_path}",
            "--predictions", f"{prediction_csv_path}"
        ],
        cwd=competition_host_git_repos_path,
        stderr=asyncio.subprocess.PIPE,
    )

    total_output = b""
    while not proc.stderr.at_eof():
        buffer = await proc.stderr.read(64)
        total_output += buffer
        print(buffer.decode(errors="ignore"), end="", flush=True, file=sys.stderr)
        # await asyncio.sleep(1)

    await proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(proc.returncode)

    line = total_output.splitlines()[-1]
    return eval(line.split(maxsplit=1)[1])


def validate(prediction_csv_path, days: int):
    return asyncio.run(_validate_coroutine(prediction_csv_path, days))
