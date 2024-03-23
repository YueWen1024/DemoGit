# coding=utf-8
# @FileName     : transformer.py
# @Time         : 2023/10/2 8:10
# @Author       : YueWen
# @Department   : AILAB
# @Description  :
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_head_attention import MyMultiheadAttention


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        """
        :param tokens: shape : [len, batch_size]
        :return: shape: [len, batch_size, emb_size]
        """
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model) # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)  # [max_len, d_model/2]
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: [x_len, batch_size, emb_size]
        :return: [x_len, batch_size, emb_size]
        """
        x = x + self.pe[:x.size(0), :]  # [x_len, batch_size, d_model]
        return self.dropout(x)


# embedding使用实例
if __name__ == '__main__':
    x = torch.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]], dtype=torch.long)
    x = x.reshape(5, 2)  # [src_len, batch_size]
    token_embedding = TokenEmbedding(vocab_size=11, emb_size=512)
    x = token_embedding(tokens=x)
    pos_embedding = PositionalEncoding(d_model=512)
    x = pos_embedding(x=x)
    print(x.shape) # torch.Size([5, 2, 512])
