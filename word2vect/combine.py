import sys
import codecs
import json

fo = codecs.open("output.json","w","utf-8")
f1 = codecs.open("emarche.json","r","utf-8")
f2 = codecs.open("lapen_labeled.json","r","utf-8")
data = f1.readline()
js_data = json.loads(data)
count=0
for data in js_data["tweets"]:
	count+=1
	fo.write(data["text"]+"\n")
data = f2.readline()
js_data = json.loads(data)
for data in js_data["tweets"]:
	count+=1
	fo.write(data["text"]+"\n")
f1.close()
f2.close()
fo.close()
print(count)