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
    stopwords=stop.read().split('\t')
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

##----------2.决策树-----##
#模型效果不好的时候（拟合不足），考虑换个更强大的模型，决策树
from sklearn.tree import DecisionTreeRegressor
tree_reg=DecisionTreeRegressor()

tree_reg.fit(vec.transform(x_train),y_train)

y_test_hat=tree_reg.predict(vec.transform(x_test))
y_train_hat=tree_reg.predict(vec.transform(x_train))
tree_mse1=mean_squared_error(y_test,y_test_hat)
tree_mse2=mean_squared_error(y_train,y_train_hat)
tree_rmse1=np.sqrt(tree_mse1)
tree_rmse2=np.sqrt(tree_mse2)
print ('测试集',tree_rmse1)
print ('训练集',tree_rmse2)
#使用tf-idf特征向量，使用决策树，
#如果tree_rmse接近0，可能模型师完美的，更可能的是数据严重过度拟合了
#在有信心启动模型之前，都不要触碰测试集，所以，建议拿训练集中的一部分用户训练，另一部分用于模型的验证
