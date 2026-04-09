import argparse
import re
import torch
from transformers import JambaForCausalLM
from src.classes.config import Config


def calculate_ser(true_plain: str, pred_plain: str) -> float:
    """Calculates the Symbol Error Rate (SER) between true and predicted strings."""
    if not true_plain:
        return 1.0 if pred_plain else 0.0

    mismatches = sum(t != p for t, p in zip(true_plain, pred_plain))
    length_diff = abs(len(true_plain) - len(pred_plain))

    raw_ser = (mismatches + length_diff) / len(true_plain)
    return min(raw_ser, 1.0)


def evaluate_custom_cipher(
    checkpoint_path: str, raw_cipher: str, true_plaintext: str, use_spaces: bool
) -> None:
    """Loads a checkpoint, predicts the plaintext, and calculates SER."""

    # 1. Setup Config
    config = Config()
    config.use_spaces = use_spaces
    config.load_homophones()

    print("-" * 50)
    print(f"Config loaded: {config.unique_homophones} Unique Homophones")

    # 2. Load Model from Checkpoint
    print(f"\nLoading model from {checkpoint_path}...")
    model = JambaForCausalLM.from_pretrained(checkpoint_path, device_map="auto")
    model.config.use_cache = True
    model.eval()

    print(f"Model Parameters: {model.num_parameters():,}")
    print("-" * 50)

    # 3. Tokenize the Input
    cipher_tokens = raw_cipher.strip().split()
    cipher_ids = [config.space_token_id if x == "_" else int(x) for x in cipher_tokens]

    input_ids = [config.bos_token_id] + cipher_ids + [config.sep_token_id]
    input_tensor = torch.tensor([input_ids]).to(model.device)

    # Clean up the true plaintext by stripping accidental newlines/spaces from the triple quotes
    if not use_spaces:
        clean_true_plain = "".join(true_plaintext.split())
    else:
        # This squashes all newlines, tabs, and multi-spaces into a single space
        clean_true_plain = re.sub(r"\s+", " ", true_plaintext).strip()

    # 4. Generate the Prediction
    target_length = len(cipher_ids)
    print("Decoding cipher...")

    # Define exactly which tokens the model is ALLOWED to predict
    allowed_tokens = set(
        [config.eos_token_id, config.space_token_id]
        + list(range(config.char_offset, config.char_offset + 26))
    )

    # Ban everything else (all the cipher numbers, BOS, SEP, etc.)
    banned_tokens = [i for i in range(config.vocab_size) if i not in allowed_tokens]

    with torch.no_grad():
        output = model.generate(
            input_tensor,
            max_new_tokens=target_length,
            min_new_tokens=target_length,
            do_sample=False,
            pad_token_id=config.pad_token_id,
            eos_token_id=config.eos_token_id,
            suppress_tokens=banned_tokens,
        )

    # 5. Decode the Output
    pred_ids = output[0][len(input_ids) :].tolist()

    # Let's peek at the first 20 raw token IDs the model predicted
    print(f"Raw Predicted IDs (first 20): {pred_ids[:20]}")
    print(f"Current char_offset is: {config.char_offset}")

    chars = []
    for idx in pred_ids:
        if idx == config.space_token_id:
            chars.append("_" if config.use_spaces else " ")
        elif idx >= config.char_offset:
            # We add a safety check so 'z' is the highest it goes,
            # preventing errors if the model hallucinates a massive number
            char_val = idx - config.char_offset
            if char_val < 26:
                chars.append(chr(char_val + ord("a")))
            else:
                chars.append("?")  # Out of bounds alphabet
        elif idx == config.eos_token_id:
            break
        else:
            chars.append("*")  # Catch ignored tokens!

    predicted_plaintext = "".join(chars)

    # 6. Calculate SER
    ser_score = calculate_ser(clean_true_plain, predicted_plaintext)

    # 7. Display Results
    print("-" * 50)
    print(f"Input Cipher Length: {len(cipher_ids)} tokens")
    print("-" * 50)
    print(f"True Plaintext: \n{clean_true_plain}\n")
    print(f"Prediction:     \n{predicted_plaintext}")
    print("-" * 50)
    # Print SER as a clean percentage
    print(f"Symbol Error Rate (SER): {ser_score * 100:.2f}%")
    print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a cipher against a saved checkpoint."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the checkpoint folder"
    )
    parser.add_argument("--spaced", action="store_true", help="Flag to enable spaces")

    args = parser.parse_args()

    # ---> HARDCODE YOUR CIPHER HERE <---
    TEST_CIPHER = """
1 2 3 4 5 6 7 7 8 5 2 3 6 7 9 10 11 12 13 12 3 1 14 15 16 17 13 18 14 19 7 20 21 16 22 8 7 23 24 25 26 21 16 24 27 17 14 18 19 4 24 3 4 13 28 10 17 5 12 29 30 31 31 10 27 10 18 14 20 15 22 10 32 27 5 19 1 5 6 33 30 6 17 23 24 10 11 3 17 29 20 20 8 34 8 24 16 25 18 19 2 2 28 21 32 10 33 21 18 14 21 35 23 36 27 15 20 9 12 3 20 21 37 7 32 28 15 37 2 28 15 38 28 29 7 24 11 28 8 7 16 38 38 29 2 10 22 21 27 4 21 31 37 24 31 27 27 38 21 34 16 38 24 8 24 25 39 16 20 20 13 12 15 5 26 3 11 10 5 1 6 14 2 12 27 17 1 5 28 18 18 20 29 1 11 12 21 15 29 26 20 3 8 17 32 16 5 5 18 19 7 13 6 14 13 12 8 18 26 27 34 10 7 18 14 2 12 8 14 15 29 1 11 37 14 19 7 20 8 16 22 8 4 8 24 25 26 21 29 24 26 21 14 15 26 17 18 4 20 3 9 32 18 2 28 23 12 3 17 32 6 26 3 5 16 20 8 22 23 7 11 37 14 32 28 15 23 31 18 24 19 17 29 1 13 28 23 37 5 5 16 1 27 6 7 37 14 3 11 17 3 7 1 13 27 13 19 11 10 18 7 10 14 9 15 34 38 29 4 11 13 28 23 29 1 17 15 26 11 8 24 24 3 14 14 15 38 15 7 5 15 10 7 29 34 15 6 14 32 12 15 1 15 11 9 37 30 29 38 11 17 37 14 11 12 15 30 8 7 2 29 2 23 19 5 12 37 4 21 9 37 19 20 24 11 28 19 17 33 18 38 21 4 16 2 19 26 16 20 20 39 5 37 7 5 20 19 24 21 11 28 16 11 2 12 8 12 10 17 13 6 38 27 5 16 20 23 22 21 4 32 17 9 23 26 23 2 28 15 6 26 10 34 10 7 29 20 37 5 5 16 17 10 18 7 6 14 32 28 8 27 4 17 13 27 11 19 13 3 18 7 18 14 2 28 15 14 8 1 32 10 22 29 20 16 7 24 11 12 29 2 2 12 15 26 23 14 23 26 15 4 5 8 32 37 2 12 8 12 29 26 22 21 17 11 27 4 32 28 8 30 26 8 1 23 7 13 29 32 10 18 7 37 14 2 28 8 1 12 23 16 14 37 14 14 10 26 17 32 14 38 19 3 32 17 9 16 1 2 12 8 20 29 13 8 26 3 4 2 38 18 24 19 5 13 3 37 4 27 4 13 37 13 28 8 5 21 26 8 33 6 7 3 21 1 18 14 13 12 23 9 23 15 36 25 19 13 32 28 21 13 2 6 19 11 28 27 17 11 28 16 32 13 12 10 17 7 16 2 19 26 29 20 3 17 11 3 5 10 24 23 7 2 10 14 10 5 16 13 3 6 7 6 14 13 12 21 17 23 28 21 25 26 15 9 14 21 29 1 32 17 9 10 32 12 32 28 8 28 16 38 22 21 1 32 14 21 29 1 2 1 6 14 6 2 28 15 26 4 29 32 3 37 7 17 10 17 29 33 27 17 32 16 36 8 27 4 37 38 24 23 38 13 18 33 16 36 8 10 11 18 19 32 3 13 3 1 7 21 5 23 17 17 29 38 39 11 6 10 34 7 37 26 15 18 26 30 15 26 22 21 38 32 33 37 1 32 30 16 32 23 7 13 14 16 5 32 17 2 28 23 17 8 1 18 5 29 20 20 15 24 12 29 38 22 21 17 32
"""

    # ---> HARDCODE YOUR TRUE PLAINTEXT HERE <---
    TEST_PLAINTEXT = """
stinconnectionwiththisfeastofunleavenedbreadisfoundinthischapxxiiiofleviticuscomposeditisallegedaboutthetimeofezekielwhileontheotherhandthenarrativeinexodxiiregardedbyallthecriticsofthisschoolastheearliestaccountoftheoriginofthefeastofunleavenedbreadrefersonlytothehistoricaleventoftheexodusastheoccasionofitsinstitutionifwegranttheasserteddifferenceinageofthesetwopartsofthepentateuchonewouldthusmorenaturallyconcludethatthehistoricaleventsweretheoriginaloccasionoftheinstitutionofthefestivalandthatthereferencetotheharvestinthepresentationofthesheafoffirstfruitswasthelaterintroductionintotheceremoniesoftheweekbutthetruthisthatthisnaturalisticidentificationofthesehebrewfeastswiththeharvestfeastsofothernationsisamistakeinordertomakeitoutitisnecessarytoignoreorpervertmostpatentfactsthesesocalledharvest
"""

    evaluate_custom_cipher(
        checkpoint_path=args.checkpoint,
        raw_cipher=TEST_CIPHER,
        true_plaintext=TEST_PLAINTEXT,
        use_spaces=args.spaces,
    )