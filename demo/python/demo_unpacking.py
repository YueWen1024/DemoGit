# coding=utf-8
# @FileName     : unpacking.py
# @Time         : 2023/12/4 16:39
# @Author       : YueWen
# @Department   : AILAB
# @Description  : 演示解包
from itertools import chain


# 疑问：解包的作用是什么？


# 对列表进行解包
def demo_unpacking_list():
    # * 对 list、tuple、dict、set 解包
    list1 = [1, 2, 3]
    print(*list1)

    # list2 = ["我爱中国", "我爱NLP", "我爱菁优"]
    # print(type(*list2))

    # 解包嵌套列表
    list3 = [[1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12]]
    print(*list3)
    # print(type(*list3))

    print(chain(*list3))
    print(list(chain(*list3)))


def demo_unpacking_02():
    # ** 对 dict 解包
    def myfun(name, age):
        print(name, age)

    dict1 = {'name': 'Mogu134', 'age': 19}
    myfun(**dict1)  # **dict1 的结果是 name = 'Mogu134',age = 19


def demo_unpacking_03():
    # concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}  # 比如examples中只有一个key: {"text": []}
    examples = {"text": ['这是第一句', '这是第二句', '这是第三句', '这是第四句']}
    concat_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    print(concat_examples)


if __name__ == '__main__':
    demo_unpacking_list()
    # demo_unpacking_02()
    # demo_unpacking_03()
    pass
