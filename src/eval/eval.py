from transformers import AutoTokenizer, AutoModelForCausalLM
import pdb
import torch

model_path = '/data/linjc/trialfm/models-new/pretrain-v1/checkpoint-4600'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, use_auth_token=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map='auto',
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

input_text = "Write a study plan for a clinical trial.\n"
input_ids = tokenizer.encode(input_text, return_tensors='pt').cuda()

output = model.generate(input_ids, max_length=512, do_sample=False)

print(tokenizer.decode(output[0], skip_special_tokens=True))
pdb.set_trace()