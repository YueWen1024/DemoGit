# coding=utf-8
# @FileName     : demo.py
# @Time         : 2023/12/21 13:45
# @Author       : YueWen
# @Department   : AILAB
# @Description  : 查看 Transformers 中的一些常量

from transformers import MODEL_FOR_CAUSAL_LM_MAPPING

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

print("MODEL_TYPES:", MODEL_TYPES)
