import sys
import codecs
import json
import os

fo = codecs.open("output.txt","w","utf-8")
path = "./Combine/"
files = os.listdir(path)
for file in files:
	fi = codecs.open(path+file,"r","utf-8")
	data = fi.readline()
	js_data = json.loads(data)
	count=0
	for data in js_data["tweets"]:
		count+=1
		fo.write(data["text"]+"\n")
	data = fi.readline()
	fi.close()
fo.close()
print(count)