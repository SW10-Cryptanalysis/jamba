import argparse
from classes.trainer import JambaTrainingPipeline
from classes.config import Config


def main() -> None:
    """Execute the training loop, handling checkpoints and final model saving."""
    parser = argparse.ArgumentParser(description="Train Jamba Cipher Model")

    parser.add_argument(
        "--spaced",
        action="store_true",
        help="Flag to train on the spaced dataset instead of the normal dataset.",
    )

    args = parser.parse_args()

    cfg = Config()

    cfg.use_spaces = args.spaced

    pipeline = JambaTrainingPipeline(config=cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
