# coding=utf-8

# all word for feature
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import SVC, LinearSVC, NuSVC

def bag_of_words(words):
    """
    词袋
    :param words:
    :return:  dict, the normal data format of nltk
    """
    return dict([(word, True) for word in words])


# 双词搭配
import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


# 把所有词和双词搭配一起作为特征
def bigram(words, score_fn=BigramAssocMeasures.chi_sq, n=1000):
    bigram_finder = BigramCollocationFinder.from_words(words)  # 把文本变成双词搭配的形式
    bigrams = bigram_finder.nbest(score_fn, n)  # 使用了卡方统计的方法，选择排名前1000的双词
    return bag_of_words(bigrams)


# 把所有词和双词搭配一起作为特征
def bigram_words(words, score_fn=BigramAssocMeasures.chi_sq, n=1000):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bag_of_words(words + bigrams)  # 所有词和（信息量大的）双词搭配一起作为特征


# 整个语料里面每个词的信息量
def get_word_scores(file_name=None):
    word_scores = {}
    return word_scores


# 根据信息量进行倒序排序，选择排名靠前的信息量的词
def find_best_words(word_scores, number):
    # 把词按信息量倒序排序。number是特征的维度，是可以不断调整直至最优的
    best_vals = sorted(word_scores.iteritems(), key=lambda ( w, s): s, reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words


# 把选出的这些词作为特征（这就是选择了信息量丰富的特征）
def best_word_features(words, best_words):
    return dict([(word, True) for word in words if word in best_words])


# Load-data
import cPickle as pickle


def load_data(file_name=None):
    """
    data filter the stopWords and tokenizer
    data_format: [[word11, word12, ... word1n], [word21, word22, ... , word2n], ... , [wordn1, wordn2, ... , wordnn]]
    :param file_name:
    :return:
    """
    pos_review = pickle.load(open(file_name + 'pos_review.pkl', 'r'))
    neg_review = pickle.load(open(file_name + 'neg_review.pkl', 'r'))

    return pos_review, neg_review


# add label for the feature
def label_data(file_name=None):
    pos_label = pickle.load(open(file_name + 'pos_label.pkl', 'r'))
    neg_lable = pickle.load(open(file_name + 'neg_label.pkl', 'r'))
    return pos_label, neg_lable


# acc
def accuracy(data, label):
    return 1.0


# classifier
from nltk.probability import FreqDist, ConditionalFreqDist
import sklearn


def train(classifier, train_data, train_lable):
    """
    :param classifier: BernoulliNB(),LogisticRegression(),SVC()，LinearSVC()，NuSVC()
    :param train_data:
    :param train_lable:
    :return:
    """
    classifier.fit(train_data, train_lable)  # 训练分类器
    return classifier


if __name__ == '__main__':

    dimension = ['500', '1000', '1500', '2000', '2500', '3000']

    for d in dimension:
        word_scores = get_word_scores(file_name='')
        feature_number = d
        best_words = find_best_words(word_scores, int(feature_number))

        posFeatures, negFeatures = load_data(file_name='')
        pos_label, neg_label = label_data(file_name='')
        classifier = BernoulliNB()
        model = train(classifier, posFeatures + negFeatures, pos_label + neg_label)
        prob = model.predict_proba(posFeatures)