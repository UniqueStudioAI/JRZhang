import pandas as pd
import numpy as np
import math
import operator


# 数据处理
df = pd.read_csv('泰坦尼克号数据2.csv')
df.Age = df.Age.fillna(df.Age.dropna().mean())
df.Embarked[df.Embarked.isnull()] = df.Embarked.dropna().mode().values
size_mapping = {"male": 1, "female": 0}
df.Sex = df.Sex.map(size_mapping)
df = df.drop('Name', axis=1)
df = df.drop('Ticket', axis=1)
df = df.drop('Cabin', axis=1)
df = df.drop('PassengerId', axis=1)
for i in range(len(df)):
    if df.Age[i] <= 18 and df.Age[i] > 0:
        df.Age[i] = 'Kid'
    elif df.Age[i] <= 60 and df.Age[i] > 18:
        df.Age[i] = 'Adult'
    elif df.Age[i] <= 80 and df.Age[i] > 60:
        df.Age[i] = 'Old'
for j in range(len(df)):
    if df.Fare[j] >= 0 and df.Fare[j] <=20:
        df.Fare[j] = 'Low'
    elif df.Fare[j] > 20 and df.Fare[j] <=100:
        df.Fare[j] = 'High'
    elif df.Fare[j] > 100:
        df.Fare[j] = 'Very High'
_data = pd.DataFrame(df.dropna())
_pre_test = df[df.isnull().T.any()]
_test = pd.DataFrame(_pre_test)
_index = list(_pre_test.index.values)
y_train = pd.DataFrame(_data, columns=['Pclass'])
x_train = pd.DataFrame(_data, columns=['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
x_test = np.array(pd.DataFrame(_test, columns=['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']))
x_data = np.array(pd.concat([x_train, y_train], axis=1)).tolist()
x_labels = ['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Pclass']


def loadTrainData():
    postingList = x_data
    property = ['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']  # 属性集
    classLabel = set([example[-1] for example in postingList])
    classLabel = list(classLabel)  # 类别集
    return postingList, property, classLabel


# 自助采样法采样生成T个训练集和测试集
def createSample(dataSet, T):
    trainSample = []  # 记录训练数据集，共T组
    testSample = []  # 记录分组测试集，共T组
    for i in range(T):
        temp = []
        # 在原数据集中随机采样
        for j in range(len(dataSet)):
            index = np.random.randint(0, len(dataSet))
            temp.append(dataSet[index])
        trainSample.append(temp)
        # 原数据集和训练数据集的差集为测试数据集集
        remainData = [item for item in dataSet if item not in temp]
        # 对于训练集与原始数据集的差集的长度小于2的偶然情况
        # 任意选择原数据集中的两个数据加入作为测试集
        if len(remainData) < 2:
            for k in range(2):
                extData = dataSet[np.random.randint(0, len(dataSet))]
                remainData.append(extData)
        testSample.append(remainData)
    return trainSample, testSample


# 计算数据集的信息熵
def calcInfoEntropy(dataSet):
    datalength = len(dataSet)
    labelCount = {}  # 字典用类别作为key，value为出现次数，统计类别概率
    for data in dataSet:
        label = data[-1]  # 获取该条数据的类别
        if label not in labelCount.keys():
            labelCount[label] = 0
        labelCount[label] += 1
    InfoEntropy = 0  # 记录信息熵
    for key in labelCount:
        pk = float(labelCount[key]) / datalength
        InfoEntropy -= pk * math.log(pk, 2)
    return InfoEntropy


# 获取按某属性值分类之后的数据集
def splitDataSet(dataSet, axis, value):
    new_DataSet = []  # 记录分类后的数据集
    for data in dataSet:
        if data[axis] == value:
            ret_data = data[:axis]
            ret_data.extend(data[axis + 1:])
            new_DataSet.append(ret_data)
    return new_DataSet


# 选择最优的划分属性，返回其下标
def chooseBestPropertyToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcInfoEntropy(dataSet)  # 计算分类前数据集的原始熵
    bestInfoGain = 0  # 记录最大的信息增益
    bestFeature = -1  # 记录最佳分类属性的下标
    choosedIndex = []  # 记录被选中的属性
    # 按随机森林的特性，在划分最佳划分属性之前，在原来数据属性集中随机选出候选的属性集
    # 选出一个数math.floor(math.log(numFeatures, 2)) + 1远小于numFeatures
    for i in range(math.floor(math.log(numFeatures, 2)) + 1):
        index = np.random.randint(0, numFeatures)
        while index in choosedIndex:
            index = np.random.randint(0, numFeatures)
        choosedIndex.append(index)
    # 计算每候选属性相应的信息增益
    for i in choosedIndex:
        featList = [example[i] for example in dataSet]  # 某属性的取值集合
        featLabels = set(featList)
        newEntropy = 0  # 记录分支结点的信息熵
        for value in featLabels:
            subDataSet = splitDataSet(dataSet, i, value)
            pb = len(subDataSet) / float(len(dataSet))
            newEntropy += pb * calcInfoEntropy(subDataSet)  # 按特征分类后的熵
        infoGain = baseEntropy - newEntropy  # 信息增益
        if (infoGain > bestInfoGain):  # 比较每个属性的信息增益，选择最大者
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 按分类列表中的值，取出现次数最多的那个值（少数服从多数）
def majorityCnt(classList):
    classCount = {}  # 记录每个值出现的次数
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 按字典中的value排序，降序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]  # 返回最大值的分类标签


# 递归构造决策树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestPropertyIndex = chooseBestPropertyToSplit(dataSet)  # 选择最优划分属性
    bestPropertyLabel = labels[bestPropertyIndex]
    myTree = {bestPropertyLabel: {}}  # 分类结果以字典形式保存
    del (labels[bestPropertyIndex])  # 更新属性集，去除最佳划分属性
    featValues = [example[bestPropertyIndex] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:  # 递归构造该属性结点的分支
        subLabels = labels[:]
        myTree[bestPropertyLabel][value] = createTree(splitDataSet(dataSet, bestPropertyIndex, value), subLabels)
    return myTree


# 递归查找决策树，返回查找结果
def seekTree(inputdata, Tree, classLabel):
    classLabel = list(classLabel)
    if len(Tree.keys()) != 1:  # 检查决策树构造
        return '决策树有误'
    for key in Tree.keys():
        property = key
    subTree = Tree[property]
    # 查找某个属性结点的分支子树
    for value in subTree.keys():
        if value in inputdata:  # 根据输入数据查找
            if subTree[value] in classLabel:  # 检查决策树是否到叶子结点
                return subTree[value]
            else:
                # 未到叶子结点则继续递归查找
                return seekTree(inputdata, subTree[value], classLabel)
    return '输入属性有误'


# 获取多个决策树的投票结果，规则是少数服从多数
def getVoteResult(inputdata, trees, classLabel):
    results = []  # 记录投票结果
    for tree in trees:
        results.append(seekTree(inputdata, tree, classLabel))
    voteResult = majorityCnt(results)
    return voteResult


# 输出各决策树和随机森林的测试准确率
def RFdect(T):
    dataSet, property, classLabel = loadTrainData()
    trainSample, testSample = createSample(dataSet, T)
    trees = []  # 记录决策树
    # 根据自助采样的训练样本训练决策树，每组样本对应一个决策树
    for train_x in trainSample:
        co_property = property[:]
        tree = createTree(train_x, co_property)
        trees.append(tree)
        # 计算各决策树的准确率
    for i in range(len(trees)):
        correct_numi = 0  # 记录决策树预测正确的次数
        for j in range(len(testSample[i])):
            # 预测结果
            pre_result = seekTree(testSample[i][j], trees[i], classLabel)
            # 真实结果
            real_result = testSample[i][j][-1]
            if pre_result == real_result:
                correct_numi += 1
        correct_ratei = correct_numi / len(testSample[i])  # 计算准确率
        print('决策树{num}测试准确率为：+ {correct_rate}'.format(num=i + 1, correct_rate=correct_ratei))
        # 计算随机森林的准确率
    correct_num = 0  # 记录RF预测正确的次数
    for i in range(len(dataSet)):
        pre_result = getVoteResult(dataSet[i], trees, classLabel)
        if pre_result == dataSet[i][-1]:
            correct_num += 1
    correct_rate = correct_num / len(dataSet)
    print('随机森林的测试准确率为%.2f' % correct_rate)
    return '测试样本的测试结果如上'


print(RFdect(200))
