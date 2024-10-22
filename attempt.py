from transformers import AutoModelForCausalLM, AutoTokenizer

name = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)

prompt = "How to make a cake? "

input_ids = tokenizer.encode(prompt, return_tensors="pt")
breakpoint()
output = model.generate(input_ids, max_new_tokens=100)

print(tokenizer.decode(output[0], skip_special_tokens=True))

