from transformers import AutoTokenizer, AutoModelForCausalLM
import pdb
import torch

model_path = '/data/linjc/trialfm/models-mistral/pretrain-v2'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, use_auth_token=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map='auto',
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

input_text = "The Interaction Between Metformin and Microbiota - the MEMO Study. Study Overview:"
input_ids = tokenizer.encode(input_text, return_tensors='pt').cuda()

output = model.generate(input_ids, max_length=512, do_sample=False)

print(tokenizer.decode(output[0], skip_special_tokens=True))
pdb.set_trace()