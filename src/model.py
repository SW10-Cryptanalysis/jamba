from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "ai21labs/Jamba-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

prompt = open("cipher_prompt.txt").read()

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

out = model.generate(
    **inputs,
    max_new_tokens=1200,
    temperature=0.15,
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True
)

print(tokenizer.decode(out[0], skip_special_tokens=True))