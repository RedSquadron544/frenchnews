"""
Tokenize the data from the logistic regession classifier
"""
import json
import sys
import pickle
import numpy as np
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


def read_model():
    """
    Write pickle file with model
    :param mode:
    :return:
    """
    with open("logreg_model.pickle", "rb") as f:
        model = pickle.load(f)
    return model


def tokenize(raw_data):
    """
    Tokenize the raw data for model training
    :param raw_data: data from json file (IN JSON)
    :return: updated with tokenized representations
    """
    tknzer = TweetTokenizer(preserve_case=False, strip_handles=True)
    stop_words = stopwords.words('french')
    for _, el in enumerate(raw_data['tweets']):
        #TODO does this strip urls???
        text_tkns_temp = tknzer.tokenize(el['text'])
        topic_tkns_temp = tknzer.tokenize(el['topic'])
        el["text tokens"] = [x for x in text_tkns_temp if x not in stop_words]
        el["topic tokens"] = [x for x in topic_tkns_temp if x not in stop_words]
    return raw_data


def compute_sim(tknzd_data, modelfile):
    """
    Compute the consine vector similarity between the topic and the text
    :param tknzd_data:
    :return:
    """
    model = KeyedVectors.load_word2vec_format(modelfile, binary=True)
    for ix, el in enumerate(tknzd_data['tweets']):
        #TODO alignment of vectors
        temp_test = []
        for word in el["text tokens"]:
            try:
                model[word]
                temp_test.append(word)
            except KeyError:
                continue
        temp_topic = []
        for word in el["topic tokens"]:
            try:
                model[word]
                temp_topic.append(word)
            except KeyError:
                continue
        try:
            el["similarity"] = model.n_similarity(temp_test, temp_topic)
        except ZeroDivisionError:
            el["similarity"] = 0.5
    return tknzd_data


def predict(model, data):
    """
    Perform prediction
    :param mode:
    :param data:
    :return:
    """
    x_data = []
    for _, el in enumerate(data['tweets']):
        x_data.append(el['similarity'])
    x_array = np.array(x_data)
    y_array = model.predict(x_array.reshape(len(x_data), 1))
    for el, y in zip(data['tweets'], y_array):
        el["label"] = y
    return data


def writeOutput(results):
  """
  Writes the results of the classification back to a file (id, gen/fake, pos/neg)
  :param results: dict
  :return: NONE
  """
  with open("base_output.json", "w", encoding='utf-8') as f:
    json.dump(results, f)


if __name__ == "__main__":
    raw_data = read_file(sys.argv[1])
    model = read_model()
    tknzd_data = tokenize(raw_data)
    cmp_data = compute_sim(tknzd_data, sys.argv[2])
    labeled_data = predict(model, tknzd_data)
    writeOutput(labeled_data)