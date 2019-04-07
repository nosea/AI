#coding:utf-8
"""
"""
from numpy import *
import operator

def createDataSet():
  group = array([[1.0, 1.1], [1.0, 1.0], [0,0], [0, 0.1]])
  labels = ['A', 'A', 'B', 'B']
  return group, labels

def classify0(inX, dataSet, labels, k):
  """
  @param inX: 用于分类的输入向量
  @param dataSet: 训练样本集
  @param labels: 标签向量
  @param k: 用于选择最近邻居的数目
  @summary: 本函数所用计算距离的公式为欧式距离公式
  """
  #array.shape 的值是一个元组, 用于获取数组的维度, 此处为获取行数 
  dataSetSize = dataSet.shape[0]

  # tile是一个numpy的函数，是用来生成一个新的数组, tile(A, (x,y)), 其实是生成一个有X行倍A, Y倍列A的新数组。
  # 这里inX是一条记录(向量)，因为要计算这个点到其他点得距离，所以要生成N个点（向量）来跟N个点来相减
  # 矩阵相减为新的矩阵
  diffMat = tile(inX, (dataSetSize, 1)) - dataSet
  
  # 矩阵取平方为里面各个值取平方
  sqDiffMat = diffMat ** 2

  # axis＝0表示按列相加，axis＝1表示按照行的方向相加
  sqDistances = sqDiffMat.sum(axis=1)
  # 取平方根
  distances = sqDistances ** 0.5

  # 按照从小到大对数组进行排序，返回的是不是值，而是下标
  sortedDistIndicies = distances.argsort()

  # 用一个字典存储结果，字典的key为标签，值为目标数据项跟训练集数据距离最近k个的标签的个数
  classCount = {}

  # 取k个
  for i in range(k):
    voteIlabel = labels[sortedDistIndicies[i]] # 返回标签值作为key，因为label跟dataSet是一一对应的
    classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 # 值为相同标签的累加和
  
  # sorted为内置排序函数，第一个参数为要排序的list或者iter对象，key为函数，用于指定待排序元素的哪一个进行排序
  # operator.itemgetter函数对于dict来说，就是取指定key的value，这里是要用于对比label的累加值

  # python2.7
  #sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
  # python3
  sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
  
  # sorted会返回一个list，里面是一个一个的tuple, 取第一个为guess的label
  return sortedClassCount[0][0]

def file2matrix(filename):
  """
  @summary: 将文本记录转换为Numpy可以解析的格式
  """
  fr = open(filename)
  arrayOLines = fr.readlines()
  numberOfLines = len(arrayOLines)
  # 生成一个numberOfLines 行， 3列的值全部为0矩阵
  returnMat = zeros((numberOfLines, 3))
  classLabelVector = []

  index = 0
  labels = {"largeDoses": 3, "smallDoses": 2, "didntLike": 1}
  for line in arrayOLines:
    line = line.strip()
    listFromLine = line.split('\t')
    # 每一行赋值，array[a, c:d], 代表获取第a行，slice c:d的数据
    returnMat[index, :] = listFromLine[0:3]
    classLabelVector.append(labels[listFromLine[-1]])
    index += 1

  return returnMat, classLabelVector


if __name__ == "__main__":
  datingDataMat, datingLabels = file2matrix('/Users/nosea/Work/AI/mla/ch2/datingTestSet.txt')
  import matplotlib
  import matplotlib.pyplot as plt
  fig = plt.figure()
  ax = fig.add_subplot(111)
  #ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
  ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
  15.0*array(datingLabels), 15.0*array(datingLabels))
  ax.set_title("Dating Data")
  ax.set_xlabel(u"玩视频游戏所耗时间百分比")
  ax.set_ylabel(u"每周消费的冰淇淋公升数")
  plt.show()