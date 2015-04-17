#-*-coding:utf-*-
'''
import sys
import os
import jieba
reload(sys)
sys.setdefaultencoding('utf8')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import string

print sys.argv #['tfidf.py', '/home/python/BigData/Data']
#获取文件列表（该目录下放着100份文档）
def getFilelist(argv):
	path = argv[1]
	filelist = []
	files = os.listdir(path)
	for f in files:
		if(f[0] == '.'):
			pass
		else:
			filelist.append(f)
	return filelist,path
#对文档进行分词处理
def fenci(argv,path):
	#保存分词结果的目录
	sFilePath = './segfile'
	if not os.path.exists(sFilePath):
		os.mkdir(sFilePath)
	#读取文档
	filename = argv #news_data1
	f = open(path+'/'+filename,'r+')
	file_list = f.read()
	f.close()
	#对文档进行分词处理，采用默认模式
	seg_list = jieba.cut(file_list, cut_all=True) #<generator object cut at 0x7facda2b4780>
	
	#对空格，换行府进行处理
	result = []
	for seg in seg_list:
		seg = ''.join(seg.split())
		if(seg != '' and seg != '\n' and seg != '\n\n'):
			result.append(seg)
	#将分词厚的结果用空格隔开，保存至本地。比如“我来到北京清华大学”，分词结果写入：“"我来到 北京 清华大学"
	f = open(sFilePath+"/"+filename+"-seg.txt","w+")#<open file './segfile/news_data-seg.txt', mode 'w+' at 0x7f49430f8ae0>
	f.write(' '.join(result))
	f.close()

#读取100份已分词好的文档，进行TF-IDF计算
def Tfidf(filelist):
	path = './segfile/'
	corpus = []#语料库
	for ff in filelist:
		fname = path + ff
		f = open(fname+"-seg.txt",'r+')
		content = f.read()
		f.close()
		corpus.append(content)

	vectorizer = CountVectorizer()
	transformer = TfidfTransformer()
	tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus)) 
	print vectorizer.fit_transform(corpus)
	print tfidf
	word = vectorizer.get_feature_names()#对所有文本的关键字
	weight = tfidf.toarray()

	sFilePath = './tfidffile'
	if not os.path.exists(sFilePath):
		os.mkdir(sFilePath)
	
	#这里将每份文档词语的TF-IDF写入tfidffile文件夹中保存
	for i in range(len(weight)):
		print u"------Writing all the tf-idf in the",i,u"file into",sFilePath+'/'+string.zfill(i,5) + '.txt','---------'
		f = open(sFilePath+'/'+string.zfill(i,5)+'.txt','w+')
		for j in range(len(word)):
			f.write(word[j]+"	"+str(weight[i][j])+'\n')
		f.close()
if __name__ == '__main__':
	(allfile,path) = getFilelist(sys.argv) #['news_data1', 'news_data3', 'news_data2', 'news_data0'] /home/python/BigData/Data
	for ff in allfile:
		print "Using jieba on "+ff
		fenci(ff,path)
	Tfidf(allfile)
'''

__author__ = "liuxuejiang"
import jieba
import jieba.posseg as pseg
import os
import sys
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == "__main__":
	corpus = ["我 来到 北京 清华大学",
			  "他 来到 了 网易 杭研 大厦",
			  "小明 硕士 毕业 于 中国 科学院",
			  "我 爱 北京 天安门"]
	vectorizer = CountVectorizer() #该类会将文本中的词语转换为词频矩阵
	transformer = TfidfTransformer() #该类会统计每个词语的tf-idf权值
	tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))#第一个fit_transform是计算tf-idf，第二个fit_transform是计算词频矩阵
	word = vectorizer.get_feature_names()#获取词袋模型中的所有词语
	weight = tfidf.toarray()
	print weight
	for i in range(len(weight)):
		print u"-----这里输出第",i,u"类文本的词语tf-idf权重------"
		for j in range(len(word)):
			print word[j],weight[i][j]

