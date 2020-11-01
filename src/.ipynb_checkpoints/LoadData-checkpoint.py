import numpy as np
import pandas as pd
import jieba
from sklearn.model_selection import train_test_split           #划分训练/测试集
from sklearn.feature_extraction.text import CountVectorizer    #抽取特征
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
#读取并清洗数据
#因为几个文档的编码不大一样，所以兼容了三种编码模式，根据经验，这三种是经常会遇到的
def get_txt_data(txt_file):
    mostwords=[]
    try:
        file=open(txt_file,'r',encoding='utf-8')
        for line in file.readlines():
            curline=line.strip().split("\t")
            mostwords.append(curline)
    except:
        try:
            file=open(txt_file,'r',encoding='gb2312')
            for line in file.readlines():
                curline=line.strip().split("\t")
                mostwords.append(curline)
        except:
            try:
                file=open(txt_file,'r',encoding='gbk')
                for line in file.readlines():
                    curline=line.strip().split("\t")
                    mostwords.append(curline)
            except:
                ''   
    return mostwords

neg_doc=get_txt_data(r'/Users/zhipeng/lesson/python/project/data/neg.txt')
pos_doc=get_txt_data(r'/Users/zhipeng/lesson/python/project/data/pos.txt')

def context_cut(sentence):
    words_list=[]
    #获取停用词
    stop=open(r'/Users/zhipeng/lesson/python/project/data/stopwords.txt','r+',encoding='utf-8')
    stopwords=stop.read().split('\n')
    cut_words=list(jieba.cut(sentence))
    words_str=""
    for word in cut_words:
        if not(word in stopwords):
            words_list.append(word)
        words_str=','.join(words_list)
    return words_str,words_list 

#合并两个数据集，并且打上标签，分成测试集和训练集
words=[]
word_list=[]
for i in neg_doc:
    cut_words_str,cut_words_list=context_cut(i[0])
    word_list.append((cut_words_str,-1))
    words.append(cut_words_list)
for j in pos_doc:
    cut_words_str2,cut_words_list2=context_cut(j[0])
    word_list.append((cut_words_str2,1))
    words.append(cut_words_list2)
#word_list=[('菜品,质量,好,味道,好,就是,百度,的,问题,总是,用,运力,原因,来,解释,我,也,不,懂,这,是,什么,原因,晚,了,三个,小时,呵呵,厉害,吧,反正,订,了,就,退,不了,只能,干,等', -1),...,...]
#将word_list中的值和标签分别赋予给x,y
x,y=zip(*word_list)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)