"""
Train Multinomial Logistic Regression Classifier for Stance Detection
"""
import json
import sys
import nltk
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from gensim.models import KeyedVectors


def read_file(filename):
    """
    Read labelled data from file
    :param filename: name of file
    :return: json
    """
    with open(filename, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except ValueError:
            print("UNABLE TO LOAD JSON")
            data = {}
    return data


def tokenize(raw_data):
    """
    Tokenize the raw data for model training
    :param raw_data: data from json file (IN JSON)
    :return: updated with tokenized representations
    """
    tknzer = TweetTokenizer(preserve_case=False, strip_handles=True)
    stop_words = stopwords.words('french')
    for _, el in enumerate(raw_data['tweets']):
        text_tkns_temp = tknzer.tokenize(el['text'])
        topic_tkns_temp = tknzer.tokenize(el['topic'])
        el["text tokens"] = [x for x in text_tkns_temp if x not in stop_words]
        el["topic tokens"] = [x for x in topic_tkns_temp if x not in stop_words]
    return raw_data


def compute_sim(tknzd_data):
    """
    Compute the consine vector similarity between the topic and the text
    :param tknzd_data:
    :return:
    """
    for ix, el in tknzd_data['tweets']:
        


def train_logit_reg(tknzd_data):
    """
    Train the logistic regression model on tokenized data
    :param tknzd_data:
    :return:
    """
    x_data, y_data = [], []
    for _, el in enumerate(tknzd_data):


if __name__ == "__main__":
    raw_data = read_file(sys.argv[1])
    tknzd_data = tokenize(raw_data)
    cmp_data = compute_sim(tknzd_data)
    model = train_logit_reg(cmp_data)