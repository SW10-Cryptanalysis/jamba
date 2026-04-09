import argparse
from classes.evaluator import CipherEvaluator
from classes.config import Config

def main() -> None:
    """Execute the evaluation loop, parsing arguments and handling evaluation."""
    # 1. Catch the arguments passed by the SLURM script
    parser = argparse.ArgumentParser()
    parser.add_argument("--spaced", action="store_true", help="Evaluate the model trained with spaces.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory.")
    args = parser.parse_args()

    # 2. Initialize the Config and apply the dynamic flag
    cfg = Config()
    cfg.use_spaces = args.spaced

    # 3. Initialize your evaluator
    evaluator = CipherEvaluator(config=cfg, model_path=args.model_path)

    # 4. Run the evaluation
    evaluator.evaluate()

if __name__ == "__main__":
    main()
