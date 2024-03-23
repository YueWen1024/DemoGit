# coding=utf-8
# @FileName     : llama_tokenizer.py
# @Time         : 2023/10/20 17:22
# @Author       : YueWen
# @Department   : AILAB
# @Description  : 读取大模型的分词

from transformers import AutoTokenizer

# MHA MQA 和 GQA的区别是什么？


def demo_llama2():
    # 模型的路径只需要截止到分词模型文件的父目录即可，且改目录下面必须要存在分词的配置文件，比如tokenizer_config.json
    model_name = "../../pretrained/Llama-2-7b"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # print(tokenizer)
    # print(tokenizer.eos_token)
    # print(tokenizer.pad_token)  # Using pad_token, but it is not set yet. 使用pad_token，但尚未设置。

    # vocab = tokenizer.get_vocab()  # 获取词汇表
    # print(vocab)

    sentence = "这是一句测试句子"
    tokens = tokenizer.tokenize(sentence)  # ['▁', '这是一', '句', '测试', '句子']
    print(tokens)


# def demo_llama2_tokenize():


def demo_chatglm2():
    model_name = "../chatglm2/"  # 模型的路径只需要截止到分词模型文件的父目录即可
    # model_name = r"../chatglm2/tokenizer.model"  # 加上分词模型的名称，则会发生报错

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(dir(tokenizer))  # 查看对象的属性

    # print(tokenizer)
    # print(tokenizer.eos_token)  # </s>
    # print(tokenizer.eos_token)  # </s>
    # print(tokenizer.pad_token)  # <unk>
    # print(tokenizer.unk_token)  # <unk>

    # vocab = tokenizer.get_vocab()  # 获取词汇表
    # print(vocab)

    # sentence = "这是一句测试句子"
    # tokens = tokenizer.tokenize(sentence)  # ['▁', '这是一', '句', '测试', '句子']
    # print(tokens)

    # 获取特殊标记
    special_tokens = tokenizer.get_special_tokens()
    print(special_tokens)


if __name__ == '__main__':
    demo_llama2()
    # demo_chatglm2()
    # demo_llama2_tokenize()
    pass