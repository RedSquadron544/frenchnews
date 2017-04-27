import gensim
# from gensim.models import word2vec
import os
import logging

class MySentences(object):
	def __init__(self, dirname):
		self.dirname = dirname
 
	def __iter__(self):
		for fname in os.listdir(self.dirname):
			for line in open(os.path.join(self.dirname, fname)):
				yield line.split()

if __name__ == '__main__':
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	sentences = MySentences('./data') # a memory-friendly iterator
	model = gensim.models.Word2Vec(sentences, min_count=2, size=700, window=4)
	model.save('/tmp/word2vec.model')
	model.wv.save_word2vec_format('/tmp/word2vec.model.bin', binary=True)
	print(model)
