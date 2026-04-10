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

    if not cfg.is_valid_init:
        raise ValueError(
            f"CRITICAL CONFIG ERROR: dimension was not initialized properly!\n"
            f"vocab_size: {cfg.vocab_size}\n"
            f"max_context: {cfg.max_context}\n"
            f"unique_homophones: {cfg.unique_homophones}\n"
            f"Check the Config class and load_homophones() method.",
        )

    pipeline = JambaTrainingPipeline(config=cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
