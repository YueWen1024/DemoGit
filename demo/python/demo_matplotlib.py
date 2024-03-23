# coding=utf-8
# @FileName     : demo_matplotlib.py
# @Time         : 2024/1/22 15:41
# @Author       : YueWen
# @Department   : AILAB
# @Description  :
from math import exp
from matplotlib import pyplot as plt
import numpy as np
f = lambda x: exp(x * 2) / (exp(x) + exp(x) + exp(x * 2))
x = np.linspace(0, 100, 100)
y_3 = [f(x_i) for x_i in x]
plt.plot(x, y_3)
plt.show()