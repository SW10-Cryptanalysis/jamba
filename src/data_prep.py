import json
import torch
from torch.utils.data import Dataset
from pathlib import Path

class HomophonicCipherDataset(Dataset):
    def __init__(self, file_paths, max_len=10240):
        self.file_paths = file_paths
        self.max_len = max_len

        # --- VOCABULARY MAPPING ---
        # Homophones: 0 to 3000
        # Plaintext (a-z): 3001 to 3026
        self.char_to_id = {chr(i + 97): i + 3001 for i in range(26)}

        # Special Tokens
        self.PAD_ID = 3027
        self.BOS_ID = 3028
        self.EOS_ID = 3029
        self.SEP_ID = 3030

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Read JSON
        with open(self.file_paths[idx], 'r') as f:
            data = json.load(f)

        # Process Ciphertext
        cipher_ids = data['ciphertext']
        if isinstance(cipher_ids, str):
            cipher_ids = [int(x) for x in cipher_ids.split()]

        # Safety clip: ensure no cipher ID exceeds 3000
        cipher_ids = [min(c, 3000) for c in cipher_ids]

        # Process Plaintext
        plain_text = data['plaintext'].lower()
        plain_ids = [self.char_to_id[c] for c in plain_text if c in self.char_to_id]

        # Construct Sequence: <BOS> CIPHER <SEP> PLAIN <EOS>
        full_seq = [self.BOS_ID] + cipher_ids + [self.SEP_ID] + plain_ids + [self.EOS_ID]

        # Labels: -100 for the cipher part (we only want to calculate loss on the plaintext prediction)
        cipher_part_len = len(cipher_ids) + 2
        labels = ([-100] * cipher_part_len) + plain_ids + [self.EOS_ID]

        # Truncation
        input_ids = full_seq[:self.max_len]
        labels = labels[:self.max_len]

        # Padding & Attention Mask
        padding_len = self.max_len - len(input_ids)
        attention_mask = [1] * len(input_ids)

        if padding_len > 0:
            input_ids += [self.PAD_ID] * padding_len
            labels += [-100] * padding_len
            attention_mask += [0] * padding_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def prepare_data(folder_path, max_len=10240):
    path = Path(folder_path)

    # Locate all json files (handles uppercase/lowercase extensions)
    files = list(path.rglob("*.json")) + list(path.rglob("*.JSON"))

    if len(files) == 0:
        raise ValueError(f"No .json files found in {path.resolve()}. Check your folder structure!")

    print(f"Successfully loaded {len(files)} files from {path.name}")
    return HomophonicCipherDataset(files, max_len=max_len)