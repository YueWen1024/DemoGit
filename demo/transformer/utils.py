import copy
import torch.nn as nn


# 克隆多个编码层或解码层功能函数
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])