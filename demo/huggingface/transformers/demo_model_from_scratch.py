# coding=utf-8
# @FileName     : demo_model_from_scratch.py
# @Time         : 2023/12/8 10:07
# @Author       : YueWen
# @Department   : AILAB
# @Description  : 根据Llama的结构，根据config文件顶一个满足个人要求的小规模参数的Llama模型
import torch
from transformers.models.llama import LlamaModel, LlamaConfig
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)

def run_llama2():
    config = LlamaConfig(vocab_size=64793, hidden_size=4096//2, intermediate_size=11008//2, num_hidden_layers=32//2,
                         num_attention_heads=32//2, max_position_embeddings=2048//2)
    print(config)
    llama = LlamaModel(config)
    print(llama)

    num_params = llama.num_parameters()
    print(f"LlamaModel的参数量：{num_params:,}")

    # # 模拟输入
    # input_ids = torch.randint(low=3, high=config.vocab_size, size=(4, 30))  # 批次大小（每批次样本数）: 4  句子长度：30
    #
    # res = llama(input_ids)
    # print(res[0].shape)  # hidden_states: torch.Size([4, 30, 2048])


def new_():
    config_path = '../../../save/tiny-llama-origin/config.json'
    config = AutoConfig.from_pretrained(config_path)  # 从配置文件加载配置
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)  # 使用配置创建模型
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    print(dir(model))
    print(f"Training new model from scratch - Total size={n_params / 2 ** 20:.2f}M params")  # 12040.4M -> 120亿；1204.04M -> 12亿；120.404M -> 1.2亿； 10的8次方是1亿，10的9次方是十亿，即B
    # 将模型和张量移动到 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()

    # 获取当前显存大小
    initial_memory = torch.cuda.memory_allocated(device)
    initial_memory_mb = initial_memory / 1024 ** 2  # 转换为 MB
    print(f"Initial GPU memory usage: {initial_memory_mb:.2f} MB")

    # 进行模型测试...

    # 获取测试完毕后的显存大小
    final_memory = torch.cuda.memory_allocated(device)
    final_memory_mb = final_memory / 1024 ** 2  # 转换为 MB
    print(f"Final GPU memory usage: {final_memory_mb:.2f} MB")

    # 释放模型占用的显存
    del model
    torch.cuda.empty_cache()

    # 确保显存已被释放
    cleared_memory = torch.cuda.memory_allocated(device)
    cleared_memory_mb = cleared_memory / 1024 ** 2  # 转换为 MB
    print(f"GPU memory after clearing: {cleared_memory_mb:.2f} MB")


def maa():
    # config = CONFIG_MAPPING['Llama-2-7b-hf']  # KeyError: 'Llama-2-7b-hf'
    # print(config)

    config = CONFIG_MAPPING['llama']  # 正确
    print(config)
    print(dir(config))


if __name__ == '__main__':
    # new_()
    maa()
    pass
