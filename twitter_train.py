# coding=utf-8
"""
@author: 2cm
@software: pycharm
@file: twitter_train.py
@time: 2018/10/15 下午3:45
@desc:利用Twitter中的评论进行训练
"""

import nltk
import sys
import pickle
from nltk.tokenize import word_tokenize


def find_features(documents):
    words = word_tokenize(documents)
    features = {}
    for word in train_text_word_features:
        features[word] = (word in words)

    return features


if __name__ == "__main__":
    reload(sys)

    twitter_message_neg = open("neg.txt", "r").read()
    twitter_message_pos = open("pos.txt", "r").read()

    document = []

    for word in twitter_message_neg.split('\n'):
        document.append((word, "neg"))

    for word in twitter_message_pos.split('\n'):
        document.append((word, "pos"))

    train_text = []

    twitter_message_pos_words = word_tokenize(twitter_message_pos)
    twitter_message_neg_words = word_tokenize(twitter_message_neg)

    for word in twitter_message_neg_words:
        train_text.append(word.lower())

    for word in twitter_message_pos_words:
        train_text.append(word.lower())

    train_text = nltk.FreqDist(train_text)

    train_text_word_features = list(train_text.keys())[:5000]

    featuressets = [(find_features(rev), category) for (rev, category) in document]

    train_text_set = featuressets[:1900]
    test_text_set = featuressets[1900:]

    classifier_f = open("naivebayes.pickle", "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()

    # classifier = nltk.NaiveBayesClassifier.train(train_text_set)
    print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, test_text_set)) * 100)
    classifier.show_most_informative_features(15)
    #
    # save_classifier = open("naivebayes_twitter_text.pickle", "wb")
    # pickle.dump(classifier, save_classifier)
    # save_classifier.close()



