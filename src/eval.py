from src.classes.evaluator import JambaEvaluator
from src.classes.config import Config


def main() -> None:
    """Execute the evaluation loop, handling checkpoints and final model saving."""
    cfg = Config()
    evaluator = JambaEvaluator(config=cfg)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
