# coding: utf-8
import numpy as np

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def numerical_gradient(f, x):  # n维数组求梯度
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # 梯度的形状与x的形状相同

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    # multi_index将元素索引(0,0) (0,1)等取出来
    # readwrite，使用可读可写的格式，我们需要改变x的值来计算f，所以需要用此方式
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]  # 取出某个元素
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值
        it.iternext()

    return grad


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()
print("net.W：{}".format(net.W))

p = net.predict(x)
print("p：{}".format(p))

print("net.loss(x,t)：{}".format(net.loss(x,t)))

f = lambda w: net.loss(x, t) # 这里的f是一个函数，不是一个值
print(type(f))
dW = numerical_gradient(f, net.W)

print(dW)
