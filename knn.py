import numpy as np
import operator
import matplotlib.pyplot as plt
from os import listdir


def createDataset():
	group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	label = np.array(['A','A','B','B'])
	return group,label

def classfy(inx,group,label,k): 
	groupx = group.shape[0]
	ingroup = np.tile(inx,(groupx,1))
	diff = ingroup - group
	diff2 = diff**2
	dis = np.sum(diff2,axis=1)
	index = dis.argsort()
	dic = {}     # 字典,统计离得最近的k个元素的label的个数
	for i in range(k):
		lt = label[index[i]]
		dic[lt] = dic.get(lt,0) + 1
	st = sorted(dic.items(),key=operator.itemgetter(1),reverse=True)
	return st[0][0]

def filetomatrix(filename):
	fr = open(filename)
	lines = fr.readlines()
	length = len(lines)
	matrix = np.zeros([length,3])
	labelv = np.zeros(length)
	index = 0
	for line in lines:
		ls = line.strip()
		la = line.split('\t')
		matrix[index,:] = la[0:3]
		labelv[index] = la[-1]
		index = index+1
	return matrix,labelv

def normmat(dataset):
	minval = dataset.min(0)
	maxval = dataset.max(0)
	dif = maxval - minval
	m = dataset.shape[0]
	norm = np.zeros(np.shape(dataset))
	norm = (dataset - np.tile(minval,(m,1)))/(np.tile(dif,(m,1)))
	return norm

def datatest(ratio,k):
	datas,labels = filetomatrix('datingTestSet2.txt')
	normm = normmat(datas)
	mum = normm.shape[0]
	testnum = int(ratio*mum)
	count = 0
	for i in range(testnum):
		classresult = classfy(normm[i,:],normm[testnum:mum,:],labels[testnum:mum],k)
		# print ('result is:',classresult,'real is:',labels[i])
		if classresult != labels[i]:
			count +=1
	return float(count)/testnum

def imagetovector(filename):
	vector = np.zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		ls = fr.readline()
		for j in range(32):
			vector[0,32*i+j] = int(ls[j])
	fr.close()
	return vector


# group,label = createDataset()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datas[:,0],datas[:,1],15.0*np.array(labels),15.0*np.array(labels))
# plt.show()
# a = datatest(0.2,3)
# print(a)

def handwritting(k,port):
	namelist = listdir('trainingDigits')
	m = len(namelist)
	m1 = int(m*port)
	print(m,m1)
	trainmat = np.zeros((m1,1024))
	labellist = []
	for i1 in range(m1):
		i = int(i1/port)
		name0 = namelist[i]
		name1 = name0.split('.')[0]
		name2 = name1.split('_')[0]
		labellist.append(int(name2))
		name3 = str('trainingDigits/'+name0)
		vector = imagetovector(name3)
		trainmat[i1,:] = vector
	testlist = listdir('testDigits')
	n = len(testlist)
	print(n)
	error = 0
	for i in range(n):
		name0 = testlist[i]
		name1 = name0.split('.')[0]
		name2 = name1.split('_')[0]
		labeltest = int(name2)
		name3 = str('testDigits/'+name0)
		vector = imagetovector(name3)
		result = classfy(vector,trainmat,labellist,k)
		if result != labeltest:
			error +=1
	errorrate = error/float(n)
	return errorrate

ffr = open('knn_port.plt','w')
ffr.write('variables = port,error'+'\n')
k = 3
err = 0.01

port = 0.05
while port <= 1.0:
	err = handwritting(3,port)
	ffr.write(str(float(port))+','+str(float(err))+'\n')
	port += 0.05
