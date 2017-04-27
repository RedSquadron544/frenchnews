import sys
import codecs
import json
import string
import re
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

def clean_word(word):
	
	# p=re.compile(r'\<http.+?\>', re.DOTALL)
	# tweet_clean = re.sub(p, '', tweet)
	return word


def main(inputfile, outputfile):
	domain_list = [".fr","www.","http","https",".com",".ly",".tt",".ch", "twitter.com" ]
	tknzer = TweetTokenizer(preserve_case=False, strip_handles=True)
	stop_words = stopwords.words('french')
	fi = codecs.open(inputfile,"r","utf-8")
	f_sent = codecs.open(outputfile,"w","utf-8")
	count = 1
	data = fi.readline().strip()
	while(data!=""):
		tweet = data.strip("\"")
		print(count)
		count+=1
		sent = ""
		if tweet[0:2] == 'RT':
			data = fi.readline().strip()
			continue
		tweet_words = tknzer.tokenize(tweet.lower())
		for t in tweet_words:
			flag = 0 
			# print(t)
			for d in domain_list:
				if re.search(d, t):
					flag = 1
			if flag:
				continue
			if t in string.punctuation:
				continue
			if re.search('[a-zA-Z]', t) == False:
			# 	print(t)
				continue
			sent += " " + t
		# print(sent)
		# print("--------------------------------------------------------------------------------------------------------------------------------")
		f_sent.write(sent+"\n")
		data = fi.readline().strip()
	fi.close()
	f_sent.close()

if __name__ == '__main__':
	main(sys.argv[1], sys.argv[2])