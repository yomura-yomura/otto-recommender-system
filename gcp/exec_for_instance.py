import subprocess
import json
import pathlib
import argparse


this_dir_path = pathlib.Path(__file__).resolve().parent


if __name__ == "__main__":
    with open(this_dir_path / "gcp_config.json", "r") as f:
        config = json.load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["start", "stop"])
    parser.add_argument("instance_names", nargs="*", choices=[[], *config["instance_names"]])
    args = parser.parse_args()

    if len(args.instance_names) == 0:
        args.instance_names = [config["instance_name"]]

    print(args.command, args.instance_names)

    for instance_name in args.instance_names:
        result = subprocess.run(
            " ".join([
                "gcloud", "compute", "instances", args.command, instance_name
            ]),
            shell=True
        )
        if result.returncode != 0:
            raise RuntimeError(result.returncode)
