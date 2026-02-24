import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import glob

class HomophonicCipherDataset(Dataset):
    def __init__(self, file_paths, max_len=1024):
        self.file_paths = file_paths
        self.max_len = max_len
        
        # Mapping: 'a' -> 256, 'b' -> 257, ..., 'z' -> 281
        self.char_to_id = {chr(i + 97): i + 256 for i in range(26)}
        
        # Special Tokens
        self.PAD_ID = 282
        self.BOS_ID = 283
        self.EOS_ID = 284
        self.SEP_ID = 285

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with open(self.file_paths[idx], 'r') as f:
            data = json.load(f)
            
        # FIX: Ensure cipher_ids is a list of integers
        cipher_ids = data['ciphertext']
        if isinstance(cipher_ids, str):
            # If it's a string of numbers like "106 136 174", split and convert
            cipher_ids = [int(x) for x in cipher_ids.split()]
        
        # 2. Process Plaintext (convert chars to our 256+ range)
        # Only keeping alphabetic characters for this specific task
        plain_text = data['plaintext'].lower()
        plain_ids = [self.char_to_id[c] for c in plain_text if c in self.char_to_id]
        
        # 3. Construct Input: <BOS> CIPHER <SEP> PLAIN <EOS>
        # Format for Causal LM: The model sees cipher, then predicts plain
        full_seq = [self.BOS_ID] + cipher_ids + [self.SEP_ID] + plain_ids + [self.EOS_ID]
        
        # 4. Construct Labels
        # We want the loss to only be calculated on the plaintext portion.
        # Use -100 for parts we want to ignore (the cipher and sep).
        cipher_part_len = len(cipher_ids) + 2 # BOS + Cipher + SEP
        labels = ([-100] * cipher_part_len) + plain_ids + [self.EOS_ID]
        
        # 5. Padding and Truncation
        input_ids = full_seq[:self.max_len]
        labels = labels[:self.max_len]
        
        padding_len = self.max_len - len(input_ids)
        if padding_len > 0:
            input_ids += [self.PAD_ID] * padding_len
            labels += [-100] * padding_len
            
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def prepare_data(folder_path, test_size=0.1):
    # Get all .txt or .json files in the folder
    files = glob.glob(os.path.join(folder_path, "*.json")) # change extension if needed
    
    train_files, val_files = train_test_split(files, test_size=test_size, random_state=42)
    
    train_dataset = HomophonicCipherDataset(train_files)
    val_dataset = HomophonicCipherDataset(val_files)
    
    return train_dataset, val_dataset

# Usage:
# train_ds, val_ds = prepare_data("ciphers")