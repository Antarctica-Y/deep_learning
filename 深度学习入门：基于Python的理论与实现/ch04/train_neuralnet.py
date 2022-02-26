# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 读入数据
# x_train，x_test元素为0-1的数，n*784
# t_train，t_test  n*1*10
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 生成两层网络的实例
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 超参数
iters_num = 10000  # 60000里面每次选100个，共运行10000次
train_size = x_train.shape[0]
batch_size = 100 # 一批100个
learning_rate = 0.1 # 学习率

train_loss_list = [] # 每次更新权重后，计算的损失
train_acc_list = [] # 训练集对应的精度
test_acc_list = [] # 测试集对应的精度

# 平均每个epoch的重复次数
iter_per_epoch = max(train_size / batch_size, 1) # 60000 / 100 = 600次，每600次算一次精度

for i in range(iters_num): # [0,10000)
    # 获取mini-batch
    batch_mask = np.random.choice(train_size, batch_size) #60000里面随机生成索引，选100个
    x_batch = x_train[batch_mask] # 在60000个中取出这100个
    t_batch = t_train[batch_mask] # 把对应的标签也取出来
    
    # 计算梯度
    #grad = network.numerical_gradient(x_batch, t_batch) # 这种计算较慢
    grad = network.gradient(x_batch, t_batch) #这种运用了误差反向传播，计算快
    
    # 数据传进去，直接求梯度
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key] # 更新权重
    
    loss = network.loss(x_batch, t_batch) # 求损失函数
    train_loss_list.append(loss) # 把损失存下来

    # 计算每个epoch的识别精度，即循环每进行600次，就记录下精度
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 绘制图形
markers = {'train': 'o', 'test': 's'}
# https://blog.csdn.net/weixin_45630708/article/details/105894942
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc') # 训练精度曲线
plt.plot(x, test_acc_list, label='test acc', linestyle='--') # 测试精度曲线
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()