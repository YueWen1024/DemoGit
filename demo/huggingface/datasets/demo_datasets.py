# coding=utf-8
# @FileName     : demo_datasets.py
# @Time         : 2023/12/29 15:38
# @Author       : YueWen
# @Department   : AILAB
# @Description  : 演示datasets
from itertools import chain

import transformers
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.testing_utils import CaptureLogger
import datasets.formatting.formatting


def demo_load_local_dataset():
    def tokenize_function(examples):
        print("In tokenize_function:")
        print("type(examples) -->>", type(examples))  # <class 'datasets.formatting.formatting.LazyBatch'>
        print("list(examples.keys()) -->>", list(examples.keys()))  # ['text']
        tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
        with CaptureLogger(tok_logger) as cl:
            # 使用预先定义的分词器 (tokenizer) 对输入的文本进行分词
            # ds['train'][0]: {'text': '毒蕈中毒预防是什么？1.切勿采摘自己不认识的蘑菇食用。2.毫无识别毒蕈经验者，千万不要自采蘑菇。3.预防措施：加强宣传、避免误食。4.有毒野生菇（菌）类常具备以下特征：1）色泽鲜艳度高。2）伞形等菇（菌）表面呈鱼鳞状。3）菇柄上有环状突起物。4）菇柄底部有不规则突起物。5）野生菇（菌）采下或受损，其受损部流出乳汁。'}
            # ds是一个字典
            output = tokenizer(examples['text'])
        # clm input could be much much longer than block_size.  CLM 输入可能比 block_size 长得多
        if "Token indices sequence length is longer than the" in cl.out:  # “令牌索引序列长度长于”
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
                # 请忽略上面的警告 - 这个长输入将被分成更小的位，然后再传递给模型。
            )
        return output

    # 疑问：clm预训练任务种为什么要将输入组成block？他的训练逻辑是什么样子的？
    def group_texts(examples):
        # 数据块的对齐
        
        print("In group_texts:")
        # print("examples -->>", examples)
        print("type(examples) -->>", type(examples))  # <class 'datasets.formatting.formatting.LazyBatch'>
        print("list(examples.keys()) -->>", list(examples.keys()))  # ['input_ids', 'attention_mask']
        print("len(examples['input_ids']", len(examples['input_ids']))  # 100，正好是数据集样本个数
        print("type(examples['input_ids']", type(examples['input_ids']))  # <class 'list'>
        print('examples[\'input_ids\']', type(examples['input_ids']), len(examples['input_ids']))  # <class 'list'>
        print('examples[\'input_ids\'][0]', type(examples['input_ids'][0]), len(examples['input_ids'][0]))  # <class 'list'>
        print('examples[\'input_ids\'][0][0]', type(examples['input_ids'][0][0]))  # <class 'list'>
        print("len(examples['attention_mask'])", len(examples['attention_mask']))  # 100，正好是数据集样本个数

        block_size = 1024
        # Concatenate all texts.
        # examples[k]列表中的多个字符串元素合并为一个元素，
        # examples[k]：取出examples中键为 k 的值，如examples['input_ids']是一个嵌套列表，列表中嵌套列表，内部列表的元素为int类型元素，例如[[1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12]]
        # 相当于把所有样本中的token汇总为1个样本，list(chain(*examples[k]): [[1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12]] -> [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        #
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}  # 比如examples中只有一个key: {"text": []}
        total_length = len(concatenated_examples[list(examples.keys())[0]])  #
        print("total_length -->>", total_length)
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size  # 得到一个小于原始 total_length 且最接近 原始total_length 的、能被 block_size 整除的长度（最后的会被丢弃吗）
        print("total_length -->>", total_length)

        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()  # 标签与input_ids保持一致，但是这样不会导致语义割裂吗（是不是割裂也就割裂一个句子，可以忽略不记呢？）
        return result

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained('../../../refer/pretrained/Llama-2-7b/')
    print("tokenizer --->>>", tokenizer)
    print("tokenizer.model_max_length --->>>", tokenizer.model_max_length)

    # 数据集字典
    data_files = {"train": r"../../../data/medical/pretrain/train_demo.json",
                  "validation": r"../../../data/medical/pretrain/train_demo.json"}

    raw_datasets = load_dataset('json', data_files=data_files)
    print('raw_datasets', raw_datasets)
    print('raw_datasets["train"]', raw_datasets["train"])
    print('raw_datasets["train"][0]', raw_datasets["train"][0])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns='text',
    )
    print('tokenized_datasets', tokenized_datasets)
    print('tokenized_datasets["train"]', tokenized_datasets["train"])
    print('tokenized_datasets["train"][0]', tokenized_datasets["train"][0])
    print('tokenized_datasets["train"][1]', tokenized_datasets["train"][1])

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
    )
    print('lm_datasets', lm_datasets)
    print('lm_datasets["train"]', lm_datasets["train"])
    print('lm_datasets["train"][0]', lm_datasets["train"][0])
    print('lm_datasets["train"][1]', lm_datasets["train"][1])

    print('len(lm_datasets["train"][0][\'input_ids\'])', len(lm_datasets["train"][0]['input_ids']))


# 演示加载本地数据集，并且进行分词
def demo_load_dataset_tokenize():
    # 定义分词函数
    def tokenize_function(examples):
        # print(" -->> ")
        # print(f"examples ---->>>>> type(examples): {type(examples)}  len(examples): {len(examples)}  len(examples['text']): {len(examples['text'])}")
        # print(f"examples ---->>>>> {examples}")
        # 当raw_dataset.map()中的batch_size设置为2时，example为以下内容
        # 当数据集中只有一个列名时，len(examples)，即len(examples)和数据集中的列名数保持一致
        # examples ---->>>>> type(examples): <class 'datasets.formatting.formatting.LazyBatch'>  len(examples): 1  len(examples['text']): 2
        # examples ---->>>>> {'text': ['脸上皮肤过敏忌吃什么？脸上皮肤过敏，通常是由于脸皮或者是本身皮肤处于敏感状态，这时就很容易诱发脸部的过敏。一般情况下，如果是人体的免疫功能下降，或者是人体处于疲劳状态，或者是其他疾病导致的机体机能下降，就很容易出现过敏症状。当然，脸部由于处在外界接触的部位，所以脸部过敏就首当其冲。通常情况下，脸部过敏或者是脸上的皮肤出现过敏症状，首先是对阳光或者是日光的敏感。所以，在这时如果要出现脸上过敏的症状，尽量避免日晒，尽量避免日照。容易引起面部皮肤过敏的源头：一、花粉：在花粉传播的季节，微小的花粉颗粒在传播过程中会散布在空气中，并随着空气的流动而四处飞扬，而其中的一部分会被人吸入同时被皮肤吸收。二、灰尘：灰尘过敏是一种生活在灰尘中的微生物的过敏反应，是最常见的过敏。灰尘过敏包括棉纤、皮毛以及各种纤维，动物皮毛等等。三、化妆品：最典型的化妆品过敏是香精过敏，而收敛水等含有酒精成分的化妆品也会对肌肤产生一定的刺激。其它如生化防腐剂、果酸等等都会对不同的肌肤造成不同的刺激。四、食物：常见的是海鲜、芒果、果仁类食物会引起过敏。五、药物：青霉素、磺胺类药物等，都可能引发皮肤过敏。六、年龄：也许年龄的增长是肌肤敏感的一个重要原因。有些人前些年的肌肤并不敏感，而这几年却变得敏感了。这是因为年轻健康的肌肤表面有一层弱酸性的皮脂膜，保持水分，以保护肌肤不受到外界侵害，但是随着年龄的增长，这层皮脂膜却不如以前的健康，以至于一些敏感物质容易入侵皮肤。脸上皮肤过敏者应该注意：尽可能地避免用手搔抓局部，也不要用热水或肥皂水去清洗局部，可吃药物来治疗，脸上皮肤过敏忌吃什么？避免食用一些刺激性食物，如葱、姜、蒜、浓茶、咖啡、酒类及其他容易引起过敏的食物，如鱼、虾等海味。',
        #                              '冻干抗蝮蛇毒血清成分或处方？主要组成成分：经胃酶消化后的马蛇毒免疫球蛋白。']}

        output = tokenizer(examples["text"])
        # print(f"output  ---->>>>>  type(output): {type(output)},  len(output): {len(output)}")
        # print(f"output  ---->>>>>  output.keys(): {list(output.keys())}  len(output['input_ids']): {len(output['input_ids'])}  len(output['attention_mask']): {len(output['attention_mask'])}")
        # print(f"output  ---->>>>>  {output}")
        # output 有两个元素 {'attention_mask': [len(batch_size)个列表]}
        # output  ---->>>>>  type(output): <class 'transformers.tokenization_utils_base.BatchEncoding'>,  len(output): 2
        # output  ---->>>>>  output.keys(): ['input_ids', 'attention_mask']  len(output['input_ids']): 2  len(output['attention_mask']): 2
        # output  ---->>>>>  {'input_ids': [[1, 29871, 235, 135, 187, 30429, 234, 157, 177, 235, 133, 167, 31138, 233, 152, 146, 232, 194, 143, 232, 147, 134, 231, 190, 131, 31882, 30882, 235, 135, 187, 30429, 234, 157, 177, 235, 133, 167, 31138, 233, 152, 146, 30214, 30768, 31190, 30392, 31272, 30909, 235, 135, 187, 234, 157, 177, 31391, 30767, 30392, 30346, 31687, 234, 157, 177, 235, 133, 167, 31548, 30909, 233, 152, 146, 233, 135, 162, 31531, 31613, 30214, 30810, 30594, 31238, 232, 193, 139, 31294, 233, 155, 150, 235, 178, 180, 30910, 235, 135, 187, 30636, 30210, 31138, 233, 152, 146, 30267, 30287, 235, 139, 175, 30993, 232, 137, 184, 30557, 30214, 30847, 30801, 30392, 30313, 30988, 30210, 232, 136, 144, 234, 153, 174, 31134, 30815, 30557, 236, 156, 144, 30214, 31391, 30767, 30392, 30313, 30988, 31548, 30909, 234, 153, 181, 232, 141, 182, 31531, 31613, 30214, 31391, 30767, 30392, 31149, 31221, 234, 153, 193, 234, 154, 136, 31943, 235, 138, 183, 30210, 31429, 30988, 31429, 30815, 30557, 236, 156, 144, 30214, 31238, 232, 193, 139, 31294, 233, 155, 150, 30544, 31424, 31138, 233, 152, 146, 234, 154, 138, 31531, 30267, 30948, 31516, 30214, 235, 135, 187, 30636, 31272, 30909, 31548, 30505, 31066, 30967, 31092, 235, 170, 169, 30210, 30636, 30956, 30214, 30744, 30651, 235, 135, 187, 30636, 31138, 233, 152, 146, 31238, 31688, 30948, 31149, 232, 137, 181, 30267, 30768, 31190, 30993, 232, 137, 184, 30557, 30214, 235, 135, 187, 30636, 31138, 233, 152, 146, 31391, 30767, 30392, 235, 135, 187, 30429, 30210, 234, 157, 177, 235, 133, 167, 30544, 31424, 31138, 233, 152, 146, 234, 154, 138, 31531, 30214, 31688, 31244, 30392, 30783, 31430, 30867, 31391, 30767, 30392, 30325, 30867, 30210, 233, 152, 146, 233, 135, 162, 30267, 30744, 30651, 30214, 30505, 30810, 30594, 30847, 30801, 30698, 30544, 31424, 235, 135, 187, 30429, 31138, 233, 152, 146, 30210, 234, 154, 138, 31531, 30214, 232, 179, 192, 31180, 236, 132, 194, 232, 136, 144, 30325, 233, 156, 149, 30214, 232, 179, 192, 31180, 236, 132, 194, 232, 136, 144, 30325, 234, 136, 170, 30267, 31294, 233, 155, 150, 31674, 31558, 30806, 30636, 234, 157, 177, 235, 133, 167, 31138, 233, 152, 146, 30210, 31193, 31584, 30383, 30287, 30330, 30830, 234, 181, 140, 30383, 30505, 30830, 234, 181, 140, 31471, 233, 149, 176, 30210, 232, 176, 166, 31669, 30214, 31935, 30446, 30210, 30830, 234, 181, 140, 236, 165, 154, 234, 181, 149, 30505, 31471, 233, 149, 176, 31138, 31101, 30275, 30437, 233, 152, 166, 31454, 30505, 30816, 233, 179, 151, 30275, 30214, 31666, 236, 157, 146, 234, 160, 131, 30816, 233, 179, 151, 30210, 31151, 30846, 31325, 30928, 31548, 236, 166, 161, 233, 140, 175, 30214, 31325, 31149, 30275, 30210, 30287, 30636, 30748, 30437, 31407, 30313, 232, 147, 187, 30752, 30980, 30594, 31407, 234, 157, 177, 235, 133, 167, 232, 147, 187, 31997, 30267, 30685, 30330, 234, 132, 179, 232, 179, 155, 30383, 234, 132, 179, 232, 179, 155, 31138, 233, 152, 146, 30392, 30287, 31893, 30486, 31704, 30505, 234, 132, 179, 232, 179, 155, 30275, 30210, 31935, 30486, 30834, 30210, 31138, 233, 152, 146, 31908, 31370, 30214, 30392, 30878, 31190, 235, 170, 132, 30210, 31138, 233, 152, 146, 30267, 234, 132, 179, 232, 179, 155, 31138, 233, 152, 146, 31473, 233, 142, 175, 233, 166, 140, 234, 189, 167, 30330, 234, 157, 177, 233, 178, 158, 30651, 31436, 232, 147, 135, 31893, 234, 189, 167, 234, 190, 183, 30214, 30846, 30834, 234, 157, 177, 233, 178, 158, 31184, 31184, 30267, 30457, 30330, 30705, 232, 169, 137, 31399, 30383, 30878, 31259, 30883, 30210, 30705, 232, 169, 137, 31399, 31138, 233, 152, 146, 30392, 31113, 234, 181, 193, 31138, 233, 152, 146, 30214, 31325, 31997, 233, 152, 158, 30716, 31184, 232, 147, 174, 30417, 236, 136, 149, 234, 181, 193, 30494, 30748, 30210, 30705, 232, 169, 137, 31399, 30953, 30437, 30783, 235, 133, 143, 235, 133, 167, 231, 189, 170, 30486, 30287, 30495, 30210, 232, 139, 189, 233, 194, 131, 30267, 31149, 232, 177, 134, 30847, 30486, 30705, 236, 155, 181, 235, 136, 147, 232, 140, 133, 30330, 30801, 236, 136, 187, 31184, 31184, 30769, 30437, 30783, 30413, 30980, 30210, 235, 133, 143, 235, 133, 167, 31420, 30494, 30413, 30980, 30210, 232, 139, 189, 233, 194, 131, 30267, 30928, 30330, 31855, 30834, 30383, 31190, 235, 170, 132, 30210, 30392, 30581, 236, 181, 159, 30330, 235, 141, 149, 30801, 30330, 30801, 31337, 30832, 31855, 30834, 30437, 31674, 31558, 31138, 233, 152, 146, 30267, 30904, 30330, 235, 144, 178, 30834, 30383, 30986, 236, 159, 140, 31605, 30330, 234, 166, 189, 235, 134, 189, 30832, 235, 144, 178, 30834, 31184, 30214, 30769, 30682, 30815, 31674, 30910, 234, 157, 177, 235, 133, 167, 31138, 233, 152, 146, 30267, 31304, 30330, 30470, 236, 193, 135, 30383, 30953, 235, 177, 187, 30470, 236, 193, 135, 30210, 232, 165, 161, 31143, 30392, 235, 133, 143, 235, 133, 167, 233, 152, 146, 233, 135, 162, 30210, 30287, 30502, 30908, 30698, 30667, 31570, 30267, 30417, 31959, 30313, 30658, 31959, 30470, 30210, 235, 133, 143, 235, 133, 167, 31666, 30413, 233, 152, 146, 233, 135, 162, 30214, 31325, 30810, 232, 138, 163, 30470, 232, 144, 183, 31462, 31050, 233, 152, 146, 233, 135, 162, 30743, 30267, 30810, 30392, 31570, 30573, 30470, 235, 192, 190, 31863, 31577, 30210, 235, 133, 143, 235, 133, 167, 30746, 30806, 30417, 30287, 232, 180, 133, 232, 191, 180, 236, 136, 187, 30952, 30210, 234, 157, 177, 235, 135, 133, 235, 137, 159, 30214, 30982, 31695, 30716, 30748, 30214, 30651, 30982, 233, 141, 167, 235, 133, 143, 235, 133, 167, 30413, 232, 146, 154, 30780, 31066, 30967, 231, 193, 184, 232, 177, 182, 30214, 231, 192, 137, 30392, 236, 157, 146, 234, 160, 131, 30470, 236, 193, 135, 30210, 232, 165, 161, 31143, 30214, 30810, 232, 180, 133, 234, 157, 177, 235, 135, 133, 235, 137, 159, 232, 144, 183, 30413, 30847, 30651, 30658, 30210, 31863, 31577, 30214, 30651, 235, 138, 182, 30909, 30287, 31959, 233, 152, 146, 233, 135, 162, 30834, 235, 183, 171, 31294, 233, 155, 150, 30752, 231, 193, 184, 234, 157, 177, 235, 133, 167, 30267, 235, 135, 187, 30429, 234, 157, 177, 235, 133, 167, 31138, 233, 152, 146, 30767, 31370, 31751, 31368, 31474, 30383, 232, 179, 192, 30682, 30815, 30533, 236, 132, 194, 232, 136, 144, 30406, 30880, 233, 147, 151, 233, 141, 150, 31655, 30636, 30214, 30953, 30413, 30698, 30406, 234, 134, 176, 30716, 31391, 235, 133, 168, 234, 157, 133, 30716, 31475, 30989, 233, 183, 154, 31655, 30636, 30214, 30682, 232, 147, 134, 235, 144, 178, 30834, 30805, 31032, 234, 153, 154, 30214, 235, 135, 187, 30429, 234, 157, 177, 235, 133, 167, 31138, 233, 152, 146, 232, 194, 143, 232, 147, 134, 231, 190, 131, 31882, 30882, 236, 132, 194, 232, 136, 144, 31855, 30406, 30287, 31959, 232, 139, 189, 233, 194, 131, 30952, 31855, 30834, 30214, 30847, 235, 148, 180, 30330, 232, 170, 159, 30330, 235, 149, 159, 30330, 233, 184, 150, 31954, 30330, 232, 149, 153, 232, 152, 164, 30330, 236, 136, 149, 30832, 31436, 31149, 31221, 31294, 233, 155, 150, 31674, 31558, 31138, 233, 152, 146, 30210, 31855, 30834, 30214, 30847, 236, 180, 191, 30330, 235, 156, 193, 31184, 30581, 232, 148, 182, 30267],
        #                                   [1, 29871, 232, 137, 190, 232, 188, 181, 233, 141, 154, 235, 160, 177, 235, 158, 138, 233, 178, 149, 235, 164, 131, 30989, 30494, 30748, 31391, 31548, 30525, 30882, 30888, 30698, 31263, 30494, 30494, 30748, 30383, 31412, 235, 134, 134, 236, 136, 185, 31276, 30705, 30822, 30210, 31530, 235, 158, 138, 233, 178, 149, 232, 136, 144, 234, 153, 174, 31539, 235, 158, 142, 30868, 30267]],
        #                     'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #                                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
        # output中初始的attention_mask 全为1
        # Llama词表中，1: <s>, 29871: ▁, 30267: 。
        return output

    tokenizer = AutoTokenizer.from_pretrained("../../../refer/pretrained/Llama-2-7b", trust_remote_code=True)  # 加载分词器
    # print("token_id: 1", tokenizer.convert_ids_to_tokens(1))
    # print("token_id: 30267", tokenizer.convert_ids_to_tokens(30267))
    # print("token_id: 29871", tokenizer.convert_ids_to_tokens(29871))
    data_files = {"train": "./medical/pretrain/train_encyclopedia.json",
                  "validation": "./medical/pretrain/valid_encyclopedia.json"}
    raw_datasets = load_dataset("json", data_files=data_files)

    column_names = ['text']
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        # batch_size=2
    )
    print(tokenized_datasets)
    # DatasetDict({
    #     train: Dataset({
    #         features: ['input_ids', 'attention_mask'],
    #         num_rows: 361420
    #     })
    #     validation: Dataset({
    #         features: ['input_ids', 'attention_mask'],
    #         num_rows: 500
    #     })
    # })


if __name__ == '__main__':
    demo_load_local_dataset()
    pass