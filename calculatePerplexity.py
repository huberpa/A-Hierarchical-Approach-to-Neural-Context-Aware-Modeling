from keras.models import load_model
import numpy as np
import json
import math
import optparse
import nltk
import re

parser = optparse.OptionParser()
parser.add_option('-i', '--input', action="store", dest="input", help="relative path to the dataset to test", default="")
#parser.add_option('-t', '--test', action="store", dest="test", help="relative path to the modified dataset", default="")
parser.add_option('-m', '--model', action="store", dest="model", help="relative path to the Model to test", default="")
parser.add_option('-v', '--vocab', action="store", dest="vocab", help="relative path to the vocabulary folder", default="")

options, args = parser.parse_args()
print options
input_path = options.input
model_path = options.model
vocab_path = options.vocab
#test_path = options.test

model = load_model(model_path)
print model.summary() 

path  = open(input_path, "r")
text = path.read().decode('utf8').lower()

#path_modified  = open(test_path, "r")
#text_modified = path_modified.read().decode('utf8').lower()

word_to_index = []
with open(vocab_path + "/word_to_index.json") as f:    
	word_to_index = json.load(f)

index_to_word = []
with open(vocab_path + "/index_to_word.json") as f:    
	index_to_word = json.load(f)

tokens = []
for index, sentence in enumerate(nltk.sent_tokenize(text)): 
    tokens.append(nltk.word_tokenize(sentence))

'''
tokens_modified = []
for index, sentence in enumerate(nltk.sent_tokenize(text_modified)): 
    tokens_modified.append(nltk.word_tokenize(sentence))
'''

mean_perplexity_all = 1.
sentence_perplexity = []
sentence_probability = []

for i, sentence in enumerate(tokens):
	interchanged_words = []
	value_sentence = []
	for word in sentence:
		try:
			value_sentence.append(word_to_index[word])
		except Exception:
			value_sentence.append(word_to_index["<UNKWN>"])

	input_perplexity = value_sentence[:len(sentence)-1]
	output_perplexity = value_sentence[1:]

	testInput = np.zeros((1, len(input_perplexity)), dtype=np.int16)
	for index, idx in enumerate(input_perplexity):
		testInput[0, index] = idx

	prediction = model.predict(testInput, verbose=0)[0]

	#all_prob = 1.
	all_per = 1.
	#for index, word in enumerate(prediction):
		#all_prob *= word[output_perplexity[index]]
		#all_per *= 1/word[output_perplexity[index]]
	#sentence_perplexity.append(all_per**(1/float(len(input_perplexity))))
	print "Calculating sentence " + str(i) + " / " + str(len(tokens))

#mean_perplexity_all = np.mean(sentence_perplexity)
print ""
print ""
print "Sequence-wise perplexity: "
#print sentence_perplexity
print ""
print "Mean Perplexity of all Sequences: " + str(mean_perplexity_all)
print ""
print ""

with open("./perplexity.txt", "w") as f:
	f.write("{}\n".format(sentence_perplexity))
	f.write("{}\n".format(" "))
	f.write("{}\n".format(mean_perplexity_all))
	f.write("{}\n".format("*"*50))


