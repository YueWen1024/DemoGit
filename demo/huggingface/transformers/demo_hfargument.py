# coding=utf-8
# @FileName     : demo_hfargument.py
# @Time         : 2023/11/15 16:41
# @Author       : YueWen
# @Department   : AILAB
# @Description  :
import sys
from dataclasses import dataclass, field
from transformers import HfArgumentParser


@dataclass
class A:
    # a: str = field(default='AAAAAAA')  # 终端无需输入内容
    a: str = field()  # 终端如果不输入内容，则会发生报错  the following arguments are required: --a  命令行中输入参数：


@dataclass
class B:
    b: str = field(default='BBBBBB')


def main():
    parser = HfArgumentParser((A, B))
    a_args, b_args = parser.parse_args_into_dataclasses()
    print(len(sys.argv), sys.argv)
    print('a_args', a_args)
    print('b_args', b_args)


if __name__ == '__main__':
    main()
    pass
