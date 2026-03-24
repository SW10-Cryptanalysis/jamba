from src.classes.trainer import JambaTrainingPipeline
from src.classes.config import Config


def main() -> None:
    """Execute the training loop, handling checkpoints and final model saving."""
    cfg = Config()
    pipeline = JambaTrainingPipeline(config=cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
