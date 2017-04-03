"""
step through json and label data
"""
import json
import sys

def read_json(fileName):
  with open(fileName, 'r', encoding='utf-8') as fileIn:
    return json.load(fileIn)

def write_json(fileName, obj):
  with open(fileName, 'w', encoding='utf-8') as fileOut:
    json.dump(obj, fileOut)

def iter_objects(json_in):
  # Too many to do this
  # write_json('temp_label_redo.json', json_in)
  for element in json_in['tweets']:
    try:
      val = element['label']
    except KeyError:
      # then we want to label this article
      print('=========================================')
      print(element['text'])
      print("+++++++++++++++++++++++++++++++++++++++++\n", "No. of Retweets: ", element['retweets'],)
      # get input from user about label
      not_sure = False
      user_input, val = "", ""
      while user_input not in 'nadou' or not not_sure:
        user_input = input("enter label for sample ('n' 'a' 'd' 'u' or 'o'): ")
        if user_input == 'n':
          val = 'neither'
        elif user_input == 'a':
          val = 'agree'
        elif user_input == 'd':
          val = 'disagree'
        elif user_input == 'o':
          val = 'other'
        elif user_input =='u':
          val = 'unrelated'
        not_sure = bool(input('Is this value correct? true/empty ' + str(val)))
        if not_sure == False:
          val = "z"
        else:
          element['label'] = val
    write_json('temp_label_redo.json', json_in)
  write_json('all_label.json', json_in)


def main():
  og_json = read_json(sys.argv[1])
  iter_objects(og_json)
main()