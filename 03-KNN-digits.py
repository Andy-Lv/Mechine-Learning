import numpy as np
import operator
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as kNN

"""
函数说明:将32x32的二进制图像转换为1x1024向量。

Parameters:
    filename - 文件名
Returns:
    returnVect - 返回的二进制图像的1x1024向量
"""


def img2vector(filename):
    # 创建1X1024零向量
    returnVect = np.zeros((1, 1024))
    # 打开文件
    file = open(filename)
    # 按行读取
    for i in range(32):
        # 读一行数据
        lineStr = file.readline()
        # 每一行的前32个元素依次添加到returnVect中
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
        # 返回转换后的1x1024向量
    return returnVect


"""
参数解释
vectorUnderTest:测试数据
trainingMat:训练数据
hwLabels:标签
k:KNN的k
"""


def classify(vectorUnderTest, trainingMat, hwLabels, k):
    # 获得训练矩阵的行数
    m = trainingMat.shape[0]

    # 将测试数据推广后获取V
    diffMat = tile(vectorUnderTest, (m, 1)) - trainingMat

    sqDiffMat = diffMat ** 2

    Distances = (sqDiffMat.sum(axis=1)) ** 0.5

    # 返回distances中元素从小到大排序后的索引值,元素越靠前,则越匹配
    sortedDistIndices = Distances.argsort()

    # 定一个记录类别次数的字典
    classCount = {}

    # 开始匹配
    for i in range(k):
        # 获取当前匹配的标签
        voteIlabel = hwLabels[sortedDistIndices[i]]

        classCount[voteIlabel] = classCount.get(voteIlabel) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]


def handwritingClassTest():
    # 测试集的Lables
    hwLabels = []
    # 返回trainingDigits目录下的文件名
    trainingFileList = listdir('digits/trainingDigits')
    # 返回文件夹下文件的个数
    m = len(trainingFileList)
    # 初始化训练的Mat矩阵,测试集
    trainingMat = np.zeros((m, 1024))

    # 从文件名中解析出训练集的类别
    for i in range(m):
        # 获得文件的名字
        fileNameStr = trainingFileList[i]
        # 获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        # 将获得的类别添加到hwLabels中
        hwLabels.append(classNumber)
        # 将每一个文件的1x1024数据存储到trainingMat矩阵中
        trainingMat[i, :] = img2vector('digits/trainingDigits/%s' % (fileNameStr))

    # 构建KNN分类器
    neigh = kNN(n_neighbors=3, algorithm='auto')
    # 拟合模型, trainingMat为训练矩阵,hwLabels为对应的标签
    neigh.fit(trainingMat, hwLabels)

    # 返回testDigits目录下的文件列表
    testFileList = listdir('digits/testDigits')
    # 错误检测计数
    errorCount = 0.0
    # 测试数据的数量
    mTest = len(testFileList)

    # 从文件中解析出测试集的类别并进行分类测试
    for i in range(mTest):
        # 获得文件的名字
        fileNameStr = testFileList[i]
        # 获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        # 获得测试集的1x1024向量,用于测试
        vectorUnderTest = img2vector('digits/testDigits/%s' % (fileNameStr))
        # 获得预测结果
        # classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        classifierResult = neigh.predict(vectorUnderTest)
        print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        if (classifierResult != classNumber):
            errorCount += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount / mTest * 100))


if __name__ == '__main__':
    handwritingClassTest()
