"""
simply calculates probability of occurance of each word in a class
then calculates the probability sum of each class
"""

import sys
import json
from collections import defaultdict
from nltk.tokenize import TweetTokenizer


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


def calc_prob(data_json):
  """
  Calculate the word probabilies for each class
  :param data_json: data read in from file
  :return: json
  """
  label_keys = ['agree', 'disagree', 'unrelated', 'neither']
  tknzer = TweetTokenizer(preserve_case=False, strip_handles=True)
  num_tweets = {"total": len(data_json['tweets']), "agree": 0, "disagree": 0,
                "unrelated": 0, "neither": 0}
  probs = {"agree": defaultdict(lambda : 0), "disagree": defaultdict(lambda : 0),
           "unrelated": defaultdict(lambda : 0), "neither": defaultdict(lambda : 0)}
  #calculate the counts
  for ix, el in enumerate(data_json['tweets']):
    #iterate over the elements get text and tolkenize
    temp_tkns = tknzer.tokenize(el['text'])
    for word in temp_tkns:
      probs[el['label']][word] += 1
      num_tweets[el['label']] += 1
  #calculates probabilities from counts
  for label in label_keys:
    for word in probs[label].keys():
      probs[label][word] /= num_tweets[label]
  return num_tweets, probs


def write_model(calc_json):
  """
  Write model to file
  :param calc_json:
  :return: nada
  """
  with open("base_model.json", "w", encoding='utf-8')as f:
    json.dump(calc_json, f)


def main():
  data = read_file(sys.argv[1])
  tweet_nums, tweet_probs = calc_prob(data)
  model_data = {"counts": tweet_nums, "probabilities": tweet_probs}
  write_model(model_data)

main()