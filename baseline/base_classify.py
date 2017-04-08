"""
Build classifier that takes the model and unlabeled data
Produces json file with
"""

import json
import sys
from math import log
from nltk.tokenize import TweetTokenizer


def read_file(filename):
  """
  Read Unlabeled data from file
  :param filename: name of file with unlabeled data
  :return: data in json
  """
  with open(filename, "r", encoding="utf-8") as f:
    try:
      data = json.load(f)
    except ValueError:
      print("UNABLE TO LOAD JSON")
      data = {}
  return data


def calc_calss(model, unlabel_data):
  """
  Calculate the most probable class
  Write json with labels added
  :param model:
  :param unlabel_data:
  :return:
  """
  label_keys = ['agree', 'disagree', 'unrelated', 'neither']
  tknzer = TweetTokenizer(preserve_case=False, strip_handles=True)
  for element in unlabel_data['tweets']:
    temp_tkns = tknzer.tokenize(element['text'])
    temp_probs = [0,0,0,0]
    # iterate over all of the words
    for word in temp_tkns:
        #iterate over all possible labels
        for ix, label in enumerate(label_keys):
          try:
            prob = model['probabilities'][word]
          except (ValueError, KeyError) as err:
            #approx laplace smoothing
            prob = 1/model['counts'][label]
          temp_probs[ix] += log(prob)
    det_class_ix = temp_probs.index(max(temp_probs))
    element['label'] = label_keys[det_class_ix]
  return unlabel_data


def writeOutput(results):
  """
  Writes the results of the classification back to a file (id, gen/fake, pos/neg)
  :param results: dict
  :return: NONE
  """
  with open("base_output.json", "w", encoding='utf-8') as f:
    json.dump(results, f)


def main():
  model = read_file("base_model.json")
  unlabel_data = read_file(sys.argv[1])
  results = calc_calss(model, unlabel_data)
  writeOutput(results)

main()