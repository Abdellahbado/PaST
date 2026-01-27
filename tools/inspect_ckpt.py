import argparse

import torch


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Inspect a saved torch checkpoint.")
    p.add_argument(
        "--path",
        type=str,
        default="runs_p100/ppo_q_seq/checkpoints/best.pt",
        help="Path to .pt checkpoint",
    )
    args = p.parse_args(argv)

    try:
        try:
            ckpt = torch.load(args.path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(args.path, map_location="cpu")
        print(f"Checkpoint type: {type(ckpt)}")
        if isinstance(ckpt, dict):
            print(f"Keys: {list(ckpt.keys())}")
            if "runner" in ckpt:
                print(f"Runner keys: {list(ckpt['runner'].keys())}")
                if "model" in ckpt["runner"]:
                    print("Found 'model' in 'runner'")
                else:
                    print("'model' NOT in 'runner'")

            if "model" in ckpt:
                print("Found 'model' in root")

            first_key = next(iter(ckpt))
            print(f"First key sample: {first_key}")
        else:
            print("Checkpoint is not a dict (maybe just state_dict?)")
        return 0
    except Exception as e:
        print(f"Error loading: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
