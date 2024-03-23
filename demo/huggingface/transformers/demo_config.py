# coding=utf-8
# @FileName     : demo_config.py
# @Time         : 2023/12/23 14:11
# @Author       : YueWen
# @Department   : AILAB
# @Description  :
from transformers import PretrainedConfig

from transformers import AutoTokenizer, LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)