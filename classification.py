# coding=utf-8
"""
@author: 2cm
@software: pycharm
@file: classification.py
@time: 2018/10/13 下午8:40
@desc:对nltk语料集中关于电影品论的数据进行训练，以及保存训练模型
"""
#
# import os
# import nltk
#
# path1 = "/Users/cmzhang/Downloads/aclImdb_v1/aclImdb/train/pos"
# path2 = "/Users/cmzhang/Downloads/aclImdb_v1/aclImdb/train/neg"
# path3 = "/Users/cmzhang/Desktop/pytest/test_file"
#
# files1 = os.listdir(path3)
# files2 = os.listdir(path2)
#
# splist1 = []
# splist2 = []
#
# for file in files1:
#     if not os.path.isdir(file):
#         f1 = open(path3 + "/" + file)  # 打开文件
#         iter_f1 = iter(f1)  # 创建迭代器
#         strs = ""
#         for line in iter_f1:  # 遍历文件，一行行遍历，读取文本
#             strs = strs + line
#         line = strs
#         # for kk in line:
#         #     splist1.append(str(line.split()))
#         splist1.append(str(line.split()))
#
# # lines = ""
# # with open("file_text1.txt", "a+") as fp:
# #     fp.seek(0)
# #     for line in fp:
# #         print line
#
# # splist1 = []
# # with open("file_text1.txt", "r") as fp:
# #     line = fp.read()
# #     splist1.append(str(line))
# #     # print line
# #
# #     # line = fp.readline()
# #     # print line
# #     # splist1.append(str(line))
# #
# print splist1

# splist1 = ["abc", "aaa", "cccc", "bbbb", "ddddd"]

# splist1 = nltk.FreqDist(splist1)
# print(splist1.most_common(15))
# print(splist1['the'])

# import random
# import nltk
# from nltk.corpus import wordnet
# from nltk.corpus import movie_reviews

# synonyms = []
# antonyms = []
#
# for syn in wordnet.synsets("good"):
#     for l in syn.lemmas():
#         synonyms.append(l.name())
#         if l.antonyms():
#             antonyms.append(l.antonyms()[0].name())
#
# print(set(synonyms))
# print(set(antonyms))
#
# w1 = wordnet.synset('ship.n.01')
# w2 = wordnet.synset('boat.n.01')
# print(w1.wup_similarity(w2))
#
# # 0.9090909090909091
#
# w1 = wordnet.synset('ship.n.01')
# w2 = wordnet.synset('car.n.01')
# print(w1.wup_similarity(w2))
#
# # 0.6956521739130435
#
# w1 = wordnet.synset('ship.n.01')
# w2 = wordnet.synset('cat.n.01')
# print(w1.wup_similarity(w2))


import nltk
import random
import pickle

from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# 将文件打乱顺序
random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:3000]


def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

# 打印出特征集
print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

# 保存特征存在性布尔值，以及它们各自的正面或负面的类别
featuresets = [(find_features(rev), category) for (rev, category) in documents]

# 分割数据
training = featuresets[:1900]
testing = featuresets[1900:]

# 对分类器进行训练
classifier = nltk.NaiveBayesClassifier.train(training)

# 测试的准确率
print("Classifier accuracy percent:", (nltk.classify.accuracy(classifier, testing))*100)

# 每个词从负面到正面出现的概率
classifier.show_most_informative_features(15)

# 利用nltk保存分类器
save_classifier = open("naivebayes.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

# 使用分类器
classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

