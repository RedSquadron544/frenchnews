"""
Format Data for each person
"""
import json
import codecs

news = []
for ix in range(1, 11):
  with open('..\\Data\\12_March_data_'+str(ix)+'.json', 'r', encoding='utf-8') as data:
    x = json.load(data)
  news.extend(x['posts'])

names = ["Michael", "Vinnie", "Ted", "Rachita"]
for ix, name in enumerate(names):
  with open("..\\Data\\data_to_label_" + name + ".json", 'w', encoding='utf-8') as fileOut:
    json.dump({'posts': news[(ix*len(news))//4:((ix+1)*len(news))//4]}, fileOut)
