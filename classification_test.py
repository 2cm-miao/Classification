# coding=utf-8
"""
@author: 2cm
@software: pycharm
@file: classification_test.py
@time: 2018/10/12 下午7:30
@desc:
"""
# import jieba
import logging
import gensim
import os
import nltk
from gensim.models import word2vec

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# with open('12487_7.txt', 'r') as fp:
#     print ("open 12487_7.txt sucessful!")
#     lines = fp.readline()        # 读取文件内容
#     print ("read 12487_7.txt sucessful!")
#     print ("lines: ", lines)
#     spplit = lines.split()       # 根据空格划分英语单词
#     print ("spplit: ", spplit)
#
# model = word2vec.Word2Vec([spplit], min_count=1, size=32)
# model.init_sims(replace=True)
#
# print (model["adorable"])   # 获取单词的词向量
#
# for e in model.most_similar(positive='adorable', topn=10):   # 输出与此单词最接近的10个单词
#    print(e[0], e[1])
#
# kk = model.similarity('adorable', 'life')     # 输出两个单词的相似度
# print ("kk: ", kk)
#
# # model.accuracy('12499_7.txt')
#
# # 保存模型
# model.save('first_train.model')

# num_features = 300    # Word vector dimensionality
# min_word_count = 10   # Minimum word count
# num_workers = 16       # Number of threads to run in parallel
# context = 10          # Context window size
# downsampling = 1e-3   # Downsample setting for frequent words
# sentences = word2vec.Text8Corpus("12487_7.txt")
#
# model = word2vec.Word2Vec(sentences)
# model.init_sims(replace=True)
# model.most_similar(positive=('plainly', 'the'), negative=['more'])  # 根据给定的条件推断相似词
# model.doesnt_match("breakfast cereal dinner lunch".split())  # 寻找离群词
# model.similarity('woman', 'man')  # 计算两个单词的相似度
# model['computer']  # 获取单词的词向量

##################################################################################################
# 将训练用文件分词后读取到一个文件中
# path1 = "/Users/cmzhang/Downloads/aclImdb_v1/aclImdb/train/pos"
# path2 = "/Users/cmzhang/Downloads/aclImdb_v1/aclImdb/train/neg"
#
# files1 = os.listdir(path1)
# files2 = os.listdir(path2)
#
# splist1 = []
# splist2 = []
#
# for file1 in files1:
#     if not os.path.isdir(file1):
#         f1 = open(path1 + "/" + file1)  # 打开文件
#         iter_f1 = iter(f1)  # 创建迭代器
#         strs = ""
#         for line in iter_f1:  # 遍历文件，一行行遍历，读取文本
#             strs = strs + line
#         line = strs
#         splist1 = line.split()
#         with open('file_text1.txt', 'a+') as ff1:    # 将英文分词存入到新的文件中
#             ff1.write(str(splist1))
#         ff1.close()
#         f1.close()
#
#
# for file2 in files2:
#     if not os.path.isdir(file2):
#         f2 = open(path2 + "/" + file2)  # 打开文件
#         iter_f2 = iter(f2)  # 创建迭代器
#         strs = ""
#         for line in iter_f2:  # 遍历文件，一行行遍历，读取文本
#             strs = strs + line
#         line = strs
#         splist2 = line.split()
#         with open('file_text2.txt', 'a+') as ff2:  # 将英文分词存入到新的文件中
#             ff2.write(str(splist2))

model = gensim.models.Word2Vec.load("first_train.model")
model.init_sims(replace=True)
# print (model["life"])

# result = model.most_similar('life')
# for n in result:
#     print n[0], n[1]


with open("file_text1.txt", "r") as fp1:
    line = fp1.readline()
sentences1 = line

with open("file_text2.txt", "r") as fp2:
    line = fp2.readline()
sentences2 = line

new_train = model.n_similarity(str(sentences1), str(sentences2))
print "new_train: ", new_train

