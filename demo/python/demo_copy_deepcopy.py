# coding=utf-8
# @FileName     : transformer.py
# @Time         : 2023/10/2 8:10
# @Author       : YueWen
# @Department   : AILAB
# @Description  : 学习 Python 中的浅拷贝、深拷贝

import sys
import copy

# 赋值是将一个变量的值或者一个对象的引用赋给另一个变量，赋值后两个变量指向同一个内存地址，对其中一个变量的修改会影响另一个变量。
# 浅拷贝是创建一个新的对象，但只复制原对象的第一层属性，如果原对象的属性是基本类型，那么复制的是值，如果原对象的属性是引用类型，那么复制的是引用，浅拷贝后两个对象的第一层属性互不影响，但第二层及以下的属性会共享同一个内存地址，对其中一个对象的修改会影响另一个对象。
# 深拷贝是创建一个新的对象，并递归地复制原对象的所有层级的属性，无论原对象的属性是基本类型还是引用类型，都会复制其实际的值，深拷贝后两个对象完全独立，互不影响。
# 赋值的好处是快速而且节省内存，但是缺点是不能保证数据的独立性和安全性，如果原对象被修改或者销毁，赋值后的对象也会受到影响。
# 浅拷贝 一般用于复制一些不包含引用类型属性的对象，或者对引用类型属性不关心的对象，比如一些基本类型的数组或者字典。
# 浅拷贝的好处是相对于深拷贝来说更快更省内存，但是缺点是不能完全复制原对象，如果原对象的引用类型属性被修改，浅拷贝后的对象也会受到影响。
# 需要注意的是，直接赋值并不是浅拷贝，Python中浅拷贝通过copy.copy()实现，深拷贝通过copy.deepcopy()是实现


# 直接赋值
def demo_direct_assignment():
    # 直接赋值：其实就是对象的引用（变量是对象的别名）。
    # 赋值缺点是不能保证数据的独立性和安全性，如果原对象被修改或者销毁，赋值后的对象也会受到影响

    # 不可变变量的直接赋值，注意：赋值后的对象不会受到原对象被修改或者销毁而产生的影响
    org_eqs = '2x<sup>2</sup>-3x+1=0'
    ass_eqs = org_eqs
    print(f"id(org_eqs): {id(org_eqs)}  org_eqs: {org_eqs}")
    print(f"id(prc_eqs): {id(ass_eqs)}  prc_eqs: {ass_eqs}")
    # 现在对ass进行一个移项操作 org_eqs进行操作
    org_eqs = '2x<sup>2</sup>-3x=-1'
    print(f"id(org_eqs): {id(org_eqs)}  org_eqs: {org_eqs}")
    print(f"id(prc_eqs): {id(ass_eqs)}  prc_eqs: {ass_eqs}")

    # 可变变量的直接赋值，注意：赋值后的对象可能会受到原对象被修改或者销毁而产生的影响
    org_lst = ['1', '2', '3']
    ass_lst = org_lst
    print(f"id(org_lst): {id(org_lst)}  org_lst: {org_lst}")
    print(f"id(ass_lst): {id(ass_lst)}  ass_lst: {ass_lst}")
    # 修改原始列表，处理赋值也会发生变化
    org_lst.append('4')
    print(f"id(org_lst): {id(org_lst)}  org_lst: {org_lst}")
    print(f"id(ass_lst): {id(ass_lst)}  pro_res: {ass_lst}")
    # 修改赋值列表，原始列表也会发生变化
    ass_lst.append('5')
    print(f"id(org_lst): {id(org_lst)}  org_lst: {org_lst}")
    print(f"id(ass_lst): {id(ass_lst)}  ass_lst: {ass_lst}")


# 不可变对象
def demo_immutable_obj():
    """
    Python中的不可变对象有int、float、str、tuple型对象
    :return:
    """
    # 变量中存放的是对象的引用（是不是可以看作是对象的内存地址？）
    # 在Python中，1是int对象，a和b都是存放该int对象引用的变量，
    # print(id(1))
    # a = 1
    # print(id(a))
    # b = 1
    # print(id(b))
    # print(a is b)
    #
    # b = 2  # 此时创建一个新的int对象，并使得b存放该int对象的引用
    # print(id(b))
    # print(a is b)

    print(f"id('文本1'): {id('文本1')}")
    tp_str = "文本1"
    print(f"id(tp_str): {id(tp_str)}")

    print(f"id('文本2'): {id('文本2')}")
    tp_str = "文本2"
    print(f"id(tp_str): {id(tp_str)}")

    pass


# Python中的可变对象和
def demo_mutable_obj():
    """
    Python中的可变对象有list，dict，set型对象
    :return:
    """
    print(id([5, 9]))
    m = [5, 9]
    print(id(m))
    m += [6]
    print(id(m))


# 浅拷贝
def demo_copy():
    """
    浅拷贝是指复制父对象，但不会复制对象内部的子对象。
    简而言之，它创建了一个新对象，然后将原始对象中的元素（或子对象的引用）复制到新对象中。
    但是，如果原始对象包含了可变对象（例如列表、字典等），则新对象中的对应元素仍然引用原始对象中相同的子对象。即仍有共享的部分。
    :return:
    """
    org_lst = [1, [2, 3]]  # list对象有两个子对象，一个是不可变对象1，一个是可变对象[2, 3]
    cpd_lst = copy.copy(org_lst)  # copied list 复制的列表

    print(f"id(org_lst[0]): {id(org_lst[0])}    id(org_lst[1]): {id(org_lst[1])}")
    print(f"id(cpd_lst[1]): {id(cpd_lst[1])}    id(cpd_lst[1]): {id(cpd_lst[1])}")

    cpd_lst[1][0] = 'X'  # 修改浅拷贝结果中的可变子对象，会对原对象中的可变子对象也产生影响
    cpd_lst[0] = 2  # 修改浅拷贝结果中的不可变子对象，不会对原对象中的可变子对象产生影响
    print("org_lst", org_lst)
    print("cpd_lst", cpd_lst)

    org_lst[1][1] = 'Y'  # 修改原对象的可变对象，会对浅拷贝结果中的可变对象也产生影响
    org_lst[0] = 3  # 修改原对象的不可变对象，不会对浅拷贝结果中的不可变对象产生影响
    print("org_lst", org_lst)
    print("cpd_lst", cpd_lst)

    print(f"id(org_lst[0]): {id(org_lst[0])}    id(org_lst[1]): {id(org_lst[1])}")
    print(f"id(cpd_lst[1]): {id(cpd_lst[1])}    id(cpd_lst[1]): {id(cpd_lst[1])}")
    print('—' * 200)

    # 只要修改不可变对象，则变量引用的对象引用就会发生变换


# 深拷贝
def demo_deepcopy():
    """
    深拷贝会复制父对象以及父对象内部的所有子对象，创建一个全新的对象结构，没有任何共享的引用。
    这意味着修改原始对象或其子对象不会影响深拷贝的结果。（完全是两个对象）
    :return:
    """
    org_lst = [1, [2, 3]]  # list对象有两个子对象，一个是不可变对象1，一个是可变对象[2, 3]
    cpd_lst = copy.deepcopy(org_lst)  # copied list 复制的列表

    print(f"id(org_lst[0]): {id(org_lst[0])}    id(org_lst[1]): {id(org_lst[1])}")
    print(f"id(cpd_lst[1]): {id(cpd_lst[1])}    id(cpd_lst[1]): {id(cpd_lst[1])}")

    cpd_lst[1][0] = 'X'  # 修改深拷贝结果中的可变子对象，不会对原对象中的可变子对象也产生影响
    cpd_lst[0] = 2  # 修改深拷贝结果中的不可变子对象，不会对原对象中的可变子对象也产生影响
    print("org_lst", org_lst)
    print("cpd_lst", cpd_lst)

    cpd_lst[1] = ['xx']
    print(f"id(org_lst[0]): {id(org_lst[0])}    id(org_lst[1]): {id(org_lst[1])}")
    print(f"id(cpd_lst[1]): {id(cpd_lst[1])}    id(cpd_lst[1]): {id(cpd_lst[1])}")
    print('—' * 200)


def demo_copy1():
    var1 = [1, 2, 3, 4, 5]  # list可变变量
    var2 = var1  # var1中的对象引用 赋值 给var2，此时var1和var2都指向list对象[1, 2, 3, 4, 5]的内存地址
    print(f"id(var1): {id(var1)}")
    print(f"id(var2): {id(var2)}")

    # 修改 var1 的值，var2的值不会发生改变
    var1 = 2  # 创建int对象2，并将该对象的引用赋值给变量var1，此时var1的内容由原来的[1, 2, 3, 4, 5]对象引用变成了int对象2的引用
    print(f'执行操作 var1 = 2后，var1={var1}，var2={var2}')  # 此时var1会指向int对象2的内存空间
    print(f"id(var1): {id(var1)}")
    print(f"id(var2): {id(var2)}")

    # 修改 var2 的值，var1的值不会发生改变
    var2 = 2
    print(f'执行操作 var2 = 2后，var1={var1}，var2={var2}')
    print(f"id(var1): {id(var1)}")
    print(f"id(var2): {id(var2)}")


# 内部函数存在同名变量
def demo_copy2():
    tl = [5, 6, 7, 8]
    print("操作前", tl)
    print("操作前地址", id(tl))

    def dif_one(tl):
        tl = [i-1 for i in tl]
        print("操作中", tl)
        print("操作中地址", id(tl))

        return tl

    _ = dif_one(tl)
    print("操作后", tl)
    print("操作后地址", id(tl))


# 引用计数 垃圾回收机制：引用计数
def demo_reference_count():
    """Python的垃圾回收机制采用引用计数机制为主，标记-清除和分代回收机制为辅的策略。"""
    a = [1, 2]
    print(f"变量 a 的引用计数：{sys.getrefcount(a)}")  # 获取对象a的引用次数
    # 注意：当把a作为参数传递给 getrefcount 时，会产生一个临时的引用，因此得出来的结果是：真实情况+1

    b = a
    print(f"变量 a 的引用计数：{sys.getrefcount(a)}")  # 获取对象a的引用次数

    del b  # 删除b的引用
    print(f"变量 a 的引用计数：{sys.getrefcount(a)}")  # 获取对象a的引用次数

    # c = list()
    # c.append(a)  ## 加入到容器中
    # sys.getrefcount(a)
    # 3
    # del c  ## 删除容器，引用-1
    # sys.getrefcount(a)
    # 2
    # b = a
    # sys.getrefcount(a)
    # 3
    # a = [3, 4]  ## 重新赋值
    # sys.getrefcount(a)


# 变量中存储的是引用
def demo_var_reference():
    a = 72
    print(f'a 的引用：{id(a)}，72的引用：{id(72)}')
    print(f'id(a)==id(72)：{id(a)==id(72)}')

    b = 73
    print(f'b 的引用：{id(b)}，73的引用：{id(73)}')

    a = b
    print(f'a 的引用：{id(a)}，73的引用：{id(73)}')
    print(f'id(a)==id(73)：{id(a) == id(73)}')
    print(f'id(a)==id(72)：{id(a) == id(72)}')


def same_name_in_func():
    def inner(tp_str):
        print(id(tp_str))

    tp_str = "用于测试"
    print(id(tp_str))
    inner(tp_str)

    tp = "用于测试"
    print(id(tp))
    inner(tp_str)


if __name__ == '__main__':
    # demo_direct_assignment()
    # demo_immutable_obj()
    # demo_mutable_obj()
    # demo_copy()
    # demo_deepcopy()
    demo_copy1()
    # demo_copy2()
    # demo_reference_count()
    # demo_var_reference()
    # same_name_in_func()
    pass
