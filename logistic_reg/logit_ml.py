"""
Train Multinomial Logistic Regression Classifier for Stance Detection
"""
import json
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from gensim.models import KeyedVectors
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score


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


def tokenize(x_vals):
    """
    Tokenize the raw data for model training
    :param x_vals: data minus labels (text, topic)
    :param y_vals: labeels
    :return:
    """
    tknzer = TweetTokenizer(preserve_case=False, strip_handles=True)
    stop_words = stopwords.words('french')
    return_tup = []
    for _, el in enumerate(x_vals):
        # Not always strip URL but omitted later on in code
        text_tkns_temp = tknzer.tokenize(el[0])
        topic_tkns_temp = tknzer.tokenize(el[1])
        txt_tokens = [x for x in text_tkns_temp if x not in stop_words]
        topic_tokens = [x for x in topic_tkns_temp if x not in stop_words]
        return_tup.append((txt_tokens, topic_tokens))
    return return_tup


def compute_sim(tknzd_data, modelfile):
    """
    Compute the consine vector similarity between the topic and the text
    :param tknzd_data: (text, topic)
    :param labels: labels
    :param modelfile: filepath to model file
    :return:
    """
    model = KeyedVectors.load_word2vec_format(modelfile, binary=True)
    sim_scores = []
    for ix, el in enumerate(tknzd_data):
        #TODO alignment of vectors
        temp_test = []
        for word in el[0]:
            try:
                model[word]
                temp_test.append(word)
            except KeyError:
                continue
        temp_topic = []
        for word in el[1]:
            try:
                model[word]
                temp_topic.append(word)
            except KeyError:
                continue
        try:
            sim_scores.append(model.n_similarity(temp_test, temp_topic))
        except ZeroDivisionError:
            sim_scores.append(0.5)
    return sim_scores


def train_logit_reg(x, y):
    """
    Train the logistic regression model on tokenized data
    :param tknzd_data:
    :return:
    """
    logreg = LogisticRegression(C=1e5)
    x_array = np.array(sim_scores_labeled)
    y_array = np.array(y)
    logreg.fit(x_array.reshape(len(sim_scores_labeled), 1), y_array)
    return logreg


def predict(model, sim_scores):
    """
    Perform prediction
    :param model:
    :param sim_scores:
    :return:
    """
    x_array = np.array(sim_scores)
    y_array = model.predict(x_array.reshape(len(sim_scores), 1))
    return y_array


def writeOutput(results):
  """
  Writes the results of the classification back to a file (id, gen/fake, pos/neg)
  :param results: dict
  :return: NONE
  """
  with open("base_output.json", "w", encoding='utf-8') as f:
    json.dump(results, f)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        # correct input args
        raw_data = read_file(sys.argv[1])
        cv_scores = []
        x = np.array([(x["text"], x["topic"]) for x in raw_data["tweets"]])
        y = np.array([y["label"] for y in raw_data["tweets"]])
        kfold = KFold(n_splits=10, shuffle=True)
        for train, test in kfold.split(x, y):
            # Train logit model
            tkzd_labeled = tokenize(x[train])
            sim_scores_labeled = compute_sim(tkzd_labeled, sys.argv[2])
            logit_mod = train_logit_reg(sim_scores_labeled, y[train])
            # Test logit model
            tknzd_unlabel = tokenize(x[test])
            sim_score_unlabeled = compute_sim(tknzd_unlabel, sys.argv[2])
            predictions = predict(logit_mod, sim_score_unlabeled)
            acc_s = accuracy_score(y[test], predictions)
            prc_val = precision_score(y[test], predictions, average='micro')
            rcl_val = recall_score(y[test], predictions, average='micro')
            f_b = fbeta_score(y[test], predictions, beta=1.0, average='micro')
            cv_scores.append(np.array([acc_s * 100, prc_val * 100, rcl_val * 100, f_b]))
            print("Accuracy: %.2f%%" % (acc_s * 100))
            print("Precision: %.2f%%" % (prc_val * 100))
            print("Recall: %.2f%%" % (rcl_val * 100))
            print("F1: %.2f" % (f_b,))
            print("=====================")
        avg = np.mean(cv_scores, axis=0)
        std_dev = np.std(cv_scores, axis=0)
        print('Overall:')
        print('Accuracy: %.2f%% (+/- %.2f%%)' % (avg[0], std_dev[0]))
        print('Precision: %.2f%% (+/- %.2f%%)' % (avg[1], std_dev[1]))
        print('Recall: %.2f%% (+/- %.2f%%)' % (avg[2], std_dev[2]))
        print('F1: %.2f (+/- %.2f)' % (avg[3], std_dev[3]))
    else:
        print("INVALID INPUT ARGS")