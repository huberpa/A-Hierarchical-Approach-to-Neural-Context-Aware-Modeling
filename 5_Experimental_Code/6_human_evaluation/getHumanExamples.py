# Commandline Arguments
##############################################
import optparse
parser = optparse.OptionParser()
parser.add_option('--dataset', action="store", dest="dataset", help="Choose the dataset to sample (default: .)", default=".")

options, args = parser.parse_args()
dataset = options.dataset
##############################################

# Imports
##############################################
import random
import sys
import nltk
import re
import os
import datetime
import json
import itertools
##############################################

# Main
##############################################

# Open the training dataset
print "Reading training data..."
path  = open(dataset, "r")
text = path.read().decode('utf8')

print "Tokenizing file..."
tokens = []
for index, sentence in enumerate(nltk.sent_tokenize(text)): 
    tokens.append(nltk.word_tokenize(sentence))

only_sentences = []
context_sentences = []

print "Searching replacements.... Please hold on"
for idx, sentence in enumerate(tokens):
	for word in sentence:
		if word.find("___") > -1:
			only_sentences.append(sentence)
			context_sentences.append(tokens[idx-10:idx+1])

randomNumbers = []
for i in range(0,20):
	print "Random selection "+str(i)+" / 10"
	while True:
		newRandNumber = random.randint(0,len(only_sentences)-1)
		found = False
		for element in randomNumbers:
			if element == newRandNumber:
				found = True
		if found == False:
			randomNumbers.append(newRandNumber)
			break

with open("./full_human_examples.txt",'w') as f:
	with open("./sentence_human_examples.txt",'w') as g:
		for element in randomNumbers:
			context = ""
			for sent in context_sentences[element]:
				context = context +" ".join(sent)
			sentence = " ".join(only_sentences[element])

			f.write("{}\n".format(context.encode('utf8')))
			g.write("{}\n".format(sentence.encode('utf8')))

print "done"




