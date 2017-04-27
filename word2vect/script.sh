#!/bin/bash

python3 word2vec_main.py output.json tweet_file.txt
mv tweet_file.txt data/tweet_file.txt
python3 word2vec.py
mv /tmp/word2vec.model.bin convertvec/
./convertvec/convertvec bin2txt ./convertvec/word2vec.model.bin model.txt
