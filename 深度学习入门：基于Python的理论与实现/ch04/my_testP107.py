# 该方法错误！！！

# 如若不把gradient_simplenet.py写成类的话，梯度值为0
# 因为在第五步，这种形式，W并没有成为函数f真正的变量，W值改变，f值并不会改变。由于f的值不变，
# f(W+h) - f(W-h) = 0，所以梯度都为0。
import numpy as np

"""激活函数"""
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


"""损失函数，交叉熵误差"""
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


"""求一维的梯度"""
def _numerical_gradient_1d(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值

    return grad


"""求二维的梯度"""
def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f, x)

        return grad


# 1.定义x，t，W
x = np.array([0.6,0.9]) # 输入数据
t = np.array([0,0,1]) # 正确结果
W = np.random.randn(2,3) # 2*3的权重

# 2.x，W点积求得y
y = np.dot(x,W)
print("y：{}".format(y))

# 3.通过softmax函数处理y
y_sm = softmax(y)
print("y_sm：{}".format(y_sm))

# 4.求损失函数，交叉熵误差
loss = cross_entropy_error(y_sm,t)
print("loss：{}".format(loss))

# 5.定义匿名函数，将W变为损失函数的参数
f = lambda  W:cross_entropy_error(y_sm,t)
# 用匿名函数将W变为参数
print(f(W))
print(f(1))
print(W)

# 6.求损失函数关于W的梯度
dW = numerical_gradient_2d(f,W)
print(dW)