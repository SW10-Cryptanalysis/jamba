import argparse
from classes.evaluator import CipherEvaluator

def main() -> None:
    """Execute the evaluation loop by passing arguments to the engine."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--spaced", action="store_true")
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    # We just need to pass the two things the class __init__ actually asks for:
    evaluator = CipherEvaluator(
        model_path=args.model_path,
        use_spaces=args.spaced,
    )

    evaluator.run()

if __name__ == "__main__":
    main()
