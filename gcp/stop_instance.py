import subprocess
import json
import pathlib


this_dir_path = pathlib.Path(__file__).resolve().parent


if __name__ == "__main__":
    with open(this_dir_path / "gcp_config.json", "r") as f:
        config = json.load(f)

    result = subprocess.run(
        " ".join([
            "gcloud.cmd", "compute", "instances", "stop", config["instance_name"]
        ]),
        shell=True
    )
    if result.returncode != 0:
        raise RuntimeError(result.returncode)
