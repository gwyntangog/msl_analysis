from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

messages = [
    {"role": "user", "content": "Write a quicksort implementation in Python"}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=500
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
