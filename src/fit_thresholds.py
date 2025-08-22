import argparse
from pathlib import Path

from src.config.thresholds import Thresholds


def main(model_dir: Path) -> None:
    thresholds = Thresholds(
        service_feedback=0.55,
        sensitive_exit=0.45,
        other=0.50,
        noise=0.70,
        review_band=0.40,
    )
    (model_dir / "thresholds.json").write_text(thresholds.model_dump_json(indent=2))
    print(f"✅ Thresholds updated at {model_dir}/thresholds.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()

    main(args.model_dir)
