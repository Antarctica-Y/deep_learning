# coding: utf-8
try:
    import urllib.request # 向网络发出请求
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path # 获取文件路径
import gzip # 读取创建压缩文件，压缩现有文件等
import pickle # 二进制格式操作
import os # 文件路径，如显示当然目录下所有文件等
import numpy as np


url_base = 'http://yann.lecun.com/exdb/mnist/'
# 以一种字典的形式下载文件，方便后面的访问
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__)) # 当前运行文件的路径，也就是mnist.py所在的位置
#path1 = os.path.dirname(__file__)
#print(path1) # 获取当前运行脚本的绝对路径，返回一个目录

#path2 = os.path.dirname(os.path.dirname(__file__))
#print(path2) # 目录的目录（去掉最后一个路径）

#path3 = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
#print(path3) # 获取当前运行脚本的绝对路径（去掉最后2个路径）

#path4 = os.__file__
#print(path4) #获取os所在的目录

#path5 = os.path.abspath(__file__)
#print(path5) #当前.py文件的绝对路径（完整路径）

#path6 = os.path.dirname(os.path.abspath(__file__))
#print(path6) # 组合使用

#path7 = os.path.join(os.path.dirname(os.path.abspath(__file__)))
#print(path7) # os.path.join()拼接路径
save_file = dataset_dir + "/mnist.pkl" # 下载数据集的存储路径 + 文件名pkl

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def _download(file_name): # 下载文件
    file_path = dataset_dir + "/" + file_name # 存的位置及名字
    
    if os.path.exists(file_path): # 如果文件已经存在，则退出函数
        return
    # 如果没有退出函数，就会执行以下的代码
    print("Downloading " + file_name + " ... ") # 告诉用户你在下载了
    urllib.request.urlretrieve(url_base + file_name, file_path)
    # urlretrieve：url就是网址，retrieve得到
    # url_base存好的网址 + 存好的文件名，就是你要下载的文件
    print("Done") # 下载完了之后告诉你已经下载完毕
    
def download_mnist(): # 调用下载文件
    for v in key_file.values(): # 把存好的文件名一个个通过字典的形式下载
       _download(v) # 从网上下载四个压缩包
        
def _load_label(file_name): # 打开某一个压缩包
    file_path = dataset_dir + "/" + file_name # 存储路径
    
    print("Converting " + file_name + " to NumPy Array ...")
    # 输出：Converting：转化，正在把那个文件格式转变为numpy数组
    with gzip.open(file_path, 'rb') as f: # 打开这个压缩文件
            labels = np.frombuffer(f.read(), np.uint8, offset=8) # 将缓冲区解释为一维数组
    print("Done")
    
    return labels

def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")
    # 输出：Converting：转化，正在把哪个文件格式转变为numpy数组
    with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    # 一维数组变成n*784的数组。因为一张图片转成了1*784一行的数组，那比如6万个样本，就是6万*784的数组
    print("Done")
    
    return data
    
def _convert_numpy(): # 转换为numpy数组，利用上面定义的函数
    dataset = {}
    # 分别调用字典key_file中的训练样本、训练标签、测试样本、测试标签
    dataset['train_img'] =  _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])    
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])
    
    return dataset
    # 得到一个新的字典，图像值为 n*784，标签值为一维数组

def init_mnist(): # 数据初始化
    download_mnist() # 下载，四个压缩包
    dataset = _convert_numpy() # 得到一个字典，值为四个数组
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1) # 使用pickle写入文档，二进制编码
    print("Done!")

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
        
    return T
    

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """读入MNIST数据集
    
    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label : 
        one_hot_label为True的情况下，标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    flatten : 是否将图像展开为一维数组
    
    Returns
    -------
    (训练图像, 训练标签), (测试图像, 测试标签)
    """
    if not os.path.exists(save_file):
        init_mnist()
        
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
            
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])
    
    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label']) 


if __name__ == '__main__':
    init_mnist()
# __name__是当前模块名，当模块被直接运行时模块名为__main__
# 模块被直接运行时init_mnist()将被运行
# 当模块是被导入时init_mnist()不被执行
