from transformers import JambaConfig, JambaForCausalLM, TrainingArguments, Trainer
from data_prep import prepare_data  # Ensure this import matches your filename
from pathlib import Path

# 1. Configure the Small Jamba Model
config = JambaConfig(
    vocab_size=300,            # Sufficient for 0-255 + special tokens
    hidden_size=256,           
    num_hidden_layers=8,       
    num_attention_heads=4,
    num_experts=1,             
    attn_layer_period=4,       # Every 4th layer is Attention
    attn_layer_offset=0,       # FIX: Must be < period. 0 means layers 0 and 4 are Attention.
    intermediate_size=1024,    # MLP expansion (usually 4x hidden_size)
    max_position_embeddings=1024,
)

model = JambaForCausalLM(config)

# 2. Setup Training Arguments
training_args = TrainingArguments(
    output_dir="./jamba-cipher-results",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    learning_rate=5e-4,
    logging_steps=50,
    eval_strategy="epoch",      # Changed from evaluation_strategy (deprecated)
    save_strategy="epoch",
    fp16=True,                  # Set to False if training on CPU
    push_to_hub=False,
    report_to="none"            # Prevents wandb errors if not logged in
)

# 3. Load Data
# Ensure the path points to where your .txt files are stored
data_dir = Path(__file__).parent.parent.parent.parent / "Ciphers"
train_data_dir: Path = data_dir / "Training"
valid_data_dir: Path = data_dir / "Validation"
train_ds = prepare_data(train_data_dir)
eval_ds = prepare_data(valid_data_dir)

# 4. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)

# 5. Start Training
trainer.train()