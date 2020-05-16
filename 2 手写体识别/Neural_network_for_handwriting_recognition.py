import numpy as np
from os import listdir#使用listdir模块，用于访问本地文件
from sklearn.neural_network import MLPClassifier
#定义img2vector函数，将加载的32*32
# 的图片矩阵展开成一列向量
def img2vector(fileName):
    retMat = np.zeros([1024], int)
    fr = open(fileName)
    lines = fr.readlines()
    for i in range(32):
        for j in range(32):
            retMat[i*32 + j ] = lines[i][j]
    return retMat
# 在sklearnBP.py文件中定义加载训练数据的函数readDataSet，
# 并将样本标签转化为one-hot向量
def readDataSet(path):
    fileList = listdir(path)#获取文件夹下的所有文件
    numFiles = len(fileList)
    dataSet = np.zeros([numFiles, 1024], int)
    hwLabels = np.zeros([numFiles, 10])
    for i in range(numFiles):
        filePath = fileList[i]
        digit = int(filePath.split('_')[0])
        hwLabels[i][digit] = 1.0
        dataSet[i] = img2vector(path + '/' + filePath)
    return dataSet, hwLabels

train_dataSet , train_hwLabels = readDataSet('trainingDigits')
# 训练神经网络
# 构建神经网络：设置网络的隐藏层数、各隐
# 藏层神经元个数、激活函数、学习率、优化方法、最大迭代次数。
clf = MLPClassifier(hidden_layer_sizes=(50,), activation='logistic', solver='sgd', learning_rate_init=0.01, max_iter=2000)
# fit函数能够根据训练集及对应标签集自动设置多层感知机的输入与输
# 出层的神经元个数。例如train_dataSet为n*1024的矩阵，train_hwLabels为n*10的矩阵，
# 则fit函数将MLP的输入层神经元个数设为1024，输出层神经元个数为
# 10
clf.fit(train_dataSet,train_hwLabels)

# 测试集评价
# 加载测试集
dataSet, hwLabels = readDataSet('testDigits')
# 使用训练好的MLP对测试集进行预测，并计算错误率
res = clf.predict(dataSet)
error_num = 0
num = len(dataSet)
for i in range(num):
    if np.sum(res[i] == hwLabels[i]) < 10:
        error_num += 1
errorRate = error_num/ float(num)
rightRate = 1- errorRate
print("******************采用多层感知机构即一个全连接的神经网络建实现效果*******************")
print("Total num:", num, "Wrong num:", error_num, "WrongRate:",errorRate, "rightRate: ",rightRate)
print("****************************************************************************")
