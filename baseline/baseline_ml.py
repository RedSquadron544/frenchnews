"""
simply calculates probability of occurance of each word in a class
then calculates the probability sum of each class
Build classifier that takes the model and unlabeled data
Produces json file with
"""

import sys
import json
import numpy as np
from math import log
from collections import defaultdict
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score


def read_file(filename):
  """
  Reads the labelled data in from file
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


def calc_prob(x_data, y_data):
  """
  Calculate the word probabilies for each class
  :param x_data: text from tweets
  :param y_data: labels from tweets
  :return: json
  """
  label_keys = ['agree', 'disagree', 'unrelated', 'neither']
  tknzer = TweetTokenizer(preserve_case=False, strip_handles=True)
  num_tweets = {"total": len(x_data), "agree": 0, "disagree": 0,
                "unrelated": 0, "neither": 0}
  probs = {"agree": defaultdict(lambda : 0), "disagree": defaultdict(lambda : 0),
           "unrelated": defaultdict(lambda : 0), "neither": defaultdict(lambda : 0)}
  #calculate the counts
  for x_val, y_val in zip(x_data, y_data):
      temp_tkns = tknzer.tokenize(x_val)
      for ix, el in enumerate(temp_tkns):
          probs[y_val][el] += 1
          num_tweets[y_val] += 1
  # make probabilities
  for label in label_keys:
      for word in probs[label].keys():
          probs[label][word] /= num_tweets[label]
  return {"counts": num_tweets, "probabilities": probs}


def write_model(calc_json):
  """
  Write model to file
  :param calc_json:
  :return: nada
  """
  with open("base_model.json", "w", encoding='utf-8')as f:
    json.dump(calc_json, f)

#
# def main():
#   data = read_file(sys.argv[1])
#   tweet_nums, tweet_probs = calc_prob(data)
#   model_data = {"counts": tweet_nums, "probabilities": tweet_probs}
#   write_model(model_data)


def calc_class(model, x_data):
  """
  Calculate the most probable class
  Write json with labels added
  :param model:
  :param x_data: x_values for the training set
  :return:
  """
  results = np.array([])
  label_keys = ['agree', 'disagree', 'unrelated', 'neither']
  tknzer = TweetTokenizer(preserve_case=False, strip_handles=True)
  for element in x_data:
    temp_tkns = tknzer.tokenize(element)
    temp_probs = [0,0,0,0]
    # iterate over all of the words
    for word in temp_tkns:
        #iterate over all possible labels
        for ix, label in enumerate(label_keys):
          if word in model["probabilities"].keys():
            prob = model['probabilities'][word]
          else:
            #approx laplace smoothing
            prob = 1/model['counts'][label]
          temp_probs[ix] += log(prob)
    det_class_ix = temp_probs.index(max(temp_probs))
    np.append(results, [label_keys[det_class_ix]])
  return results



def writeOutput(results):
  """
  Writes the results of the classification back to a file (id, gen/fake, pos/neg)
  :param results: dict
  :return: NONE
  """
  with open("base_output.json", "w", encoding='utf-8') as f:
    json.dump(results, f)


# def main():
#   model = read_file("base_model.json")
#   unlabel_data = read_file(sys.argv[2])
#   results = calc_calss(model, unlabel_data)
#   writeOutput(results)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        cv_scores = []
        data = read_file(sys.argv[1])
        x = np.array([x["text"] for x in data["tweets"]])
        y = np.array([y["label"] for y in data["tweets"]])
        kfold = KFold(n_splits=10, shuffle=True)
        for train, test in kfold.split(x, y):
            model = calc_prob(x[train], y[train])
            results = calc_class(model, x[test])
            acc_s = accuracy_score(y[test], results)
            prc_val = precision_score(y[test], results)
            rcl_val = recall_score(y[test], results)
            f_b = fbeta_score(y[test], results)
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
        print("INVALID ARGS")