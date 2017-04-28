import sys
import codecs
import json


with open("labeled_data.json","r",encoding="utf-8") as f1:
  agg = json.load(f1)
with open("fillon_labeled.json","r",encoding="utf-8") as f2:
  adding = json.load(f2)

new_additon = []
topic_fillon = "Je soutiens François Fillon. J’encourage François Fillon. François Fillon pour président."
for el in adding["tweets"]:
  if "label" in el.keys():
    if el["label"] in ["agree", "disagree", "unrelated", "neither"]:
      el["topic"] = topic_fillon
      new_additon.append(el)

new_json = {"tweets": [x for x in agg["tweets"]if "label" in x.keys() if x["label"] != "other"] + new_additon}

print(len(new_json["tweets"]))
print(len([x for x in new_json["tweets"] if "label" in x.keys()]))
print(len([x for x in new_json["tweets"] if "topic" in x.keys()]))
print(len([x for x in new_json["tweets"] if "text" in x.keys()]))

with open("combine.json","w",encoding="utf-8") as fO:
  json.dump(new_json, fO)