
# Commandline Arguments
##############################################
import optparse
parser = optparse.OptionParser()
# INPUT MUSS INDICATED FILE SEIN!!!!!
parser.add_option('-i', '--input', action="store", dest="input", help="relative path to the dataset to test", default="")
parser.add_option('-u', '--unigram', action="store", dest="unigram", help="relative path to the dataset that was trained on", default="")
parser.add_option('-m', '--model', action="store", dest="model", help="relative path to the Model to test", default="")
parser.add_option('-v', '--vocab', action="store", dest="vocab", help="relative path to the vocabulary folder", default="")
parser.add_option('-f', '--file', action="store", dest="file", help="relative path to the vocabulary folder", default="")

options, args = parser.parse_args()
print options
input_path = options.input
training_path = options.unigram
model_path = options.model
vocab_path = options.vocab
fileName = options.file
##############################################



# Imports
##############################################
from keras.models import load_model
from collections import Counter
import numpy as np
import json
import math
import os
import nltk
##############################################



# Main
##############################################

# Load the model to be tested against and output the structure
model = load_model(model_path)
print model.summary() 

# Load the file with the talks and tokenize
path  = open(input_path, "r")
text = path.read().decode('utf8')
tokens = []
for index, sentence in enumerate(nltk.sent_tokenize(text)): 
    tokens.append(nltk.word_tokenize(sentence))
tokens = [[word.lower() for word in sentence] for sentence in tokens]

# Load the index_to_word and word_to_index files to convert indizes into words
word_to_index = []
with open(vocab_path + "/word_to_index.json") as f:    
	word_to_index = json.load(f)

index_to_word = []
with open(vocab_path + "/index_to_word.json") as f:    
	index_to_word = json.load(f)

# Compute the perplexity for each sentence
words_not_in_vocab = 0
all_words_probability_with_modified = []
nb_changed_words = 0
nb_changed_words_high_prob = 0
nb_words_high_prob = 0
# Iterate through all the sentences in the data and compute the perplexity
for index, sentence in enumerate(tokens):
	value_sentence = []

	# DEBUGGING output
	if (index % 10000) == 0:
		print str(index) + " / " + str(len(tokens))

	# Adding start and end token to the sentence
	cut_sentence = ["<START>"] + sentence + ["<END>"]

	# Cutting sentences longer than 50 words
	if len(sentence) > 50:
		cut_sentence = sentence[:50]

	# Convert words into indexes and mark modified words
	changes = []
	for index, word in enumerate(cut_sentence):
		checkedWord = ""

		# Check if there is the "___" in the word --> the indicator for a modified word
		if word.find("___") == -1:
			checkedWord = word
		else:
			checkedWord = word[word.find("___")+3:word.rfind("___")]
			changes.append([index, word[word.find("___")+3:word.rfind("___")]])
			nb_changed_words += 1

		# Convert word into the according index
		try:
			value_sentence.append(word_to_index[checkedWord])
		except Exception:
			value_sentence.append(word_to_index["<UNKWN>"])

	# Create input and output vectors
	input_perplexity = value_sentence
	length = len(input_perplexity)

	# SAve input vector in numpy array
	testInput = np.zeros((1, length), dtype=np.int16)
	for index, idx in enumerate(input_perplexity):
		testInput[0, index] = idx

	# Predict the next words
	prediction = model.predict(testInput, verbose=0)[0]

	# Evaluate the prediction
	perplexity = 0.
	for index, word in enumerate(prediction):
		if word > 0.5:
			nb_words_high_prob += 1
		all_words_probability_with_modified.append([word, 0, "tbd", index_to_word[str(input_perplexity[index])]])

		#check if word got replaced and if so, save in special array
		for change in changes:
			if change[0] == index:
				try:	
					if word > 0.5:
						nb_changed_words_high_prob += 1

					all_words_probability_with_modified[len(all_words_probability_with_modified)-1][1] = 1
				except Exception:
					words_not_in_vocab += 1


#Sort all the probabilities to find modified words
all_words_probability_with_modified.sort(key=lambda row: row[0], reverse=True)

modifications_in_highest_100 = 0
modifications_in_highest_500 = 0
modifications_in_highest_1000 = 0
modifications_in_highest_4000 = 0
modifications_in_highest_10_percent = 0
modifications_in_highest_20_percent = 0
modifications_in_highest_30_percent = 0

for index, prob in enumerate(all_words_probability_with_modified):
	if index <= 100:
		if prob[1] == 1:
			modifications_in_highest_100 += 1
	if index <= 500:
		if prob[1] == 1:
			modifications_in_highest_500 += 1
	if index <= 1000:
		if prob[1] == 1:
			modifications_in_highest_1000 += 1
	if index <= 4000:
		if prob[1] == 1:
			modifications_in_highest_4000 += 1

for index, prob in enumerate(all_words_probability_with_modified[:(len(all_words_probability_with_modified)/100*30)]):
	if index <= (len(all_words_probability_with_modified)/100*10):
		if prob[1] == 1:
			modifications_in_highest_10_percent += 1
	if index <= (len(all_words_probability_with_modified)/100*20):
		if prob[1] == 1:
			modifications_in_highest_20_percent += 1
	if index <= (len(all_words_probability_with_modified)/100*30):
		if prob[1] == 1:
			modifications_in_highest_30_percent += 1

all_words_probability_with_modified.sort(key=lambda row: row[2], reverse=True)
unigram_modifications_in_highest_100 = 0
unigram_modifications_in_highest_500 = 0
unigram_modifications_in_highest_1000 = 0
unigram_modifications_in_highest_4000 = 0
unigram_modifications_in_highest_10_percent = 0
unigram_modifications_in_highest_20_percent = 0
unigram_modifications_in_highest_30_percent = 0

for index, prob in enumerate(all_words_probability_with_modified):
	if index <= 100:
		if prob[1] == 1:
			unigram_modifications_in_highest_100 += 1
	if index <= 500:
		if prob[1] == 1:
			unigram_modifications_in_highest_500 += 1
	if index <= 1000:
		if prob[1] == 1:
			unigram_modifications_in_highest_1000 += 1
	if index <= 4000:
		if prob[1] == 1:
			unigram_modifications_in_highest_4000 += 1

for index, prob in enumerate(all_words_probability_with_modified[:(len(all_words_probability_with_modified)/100*30)]):
	if index <= (len(all_words_probability_with_modified)/100*10):
		if prob[1] == 1:
			unigram_modifications_in_highest_10_percent += 1
	if index <= (len(all_words_probability_with_modified)/100*20):
		if prob[1] == 1:
			unigram_modifications_in_highest_20_percent += 1
	if index <= (len(all_words_probability_with_modified)/100*30):
		if prob[1] == 1:
			unigram_modifications_in_highest_30_percent += 1


print "Changed words high prob: "+ str(nb_changed_words_high_prob)
print "Words high prob: "+ str(nb_words_high_prob)
print "Changed words: "+ str(nb_changed_words)

#precision = nb_changed_words_high_prob / nb_words_high_prob
#recall = nb_changed_words_high_prob / nb_changed_words

# Output the results
print ""
print "Words not in vocabulary: " + str(words_not_in_vocab)
#print "System Precision: " + str(precision)
#print "System Recall: " + str(recall)
#print "System F1-Score: " + str(2*((precision*recall)/(precision+recall)))
print "Modified words within highest 100 words on dataset: " + str(modifications_in_highest_100)
print "Modified words within highest 500 words on dataset: " + str(modifications_in_highest_500)
print "Modified words within highest 1000 words on dataset: " + str(modifications_in_highest_1000)
print "Modified words within highest 4000 words on dataset: " + str(modifications_in_highest_4000)
print "Modified words within highest 10 percent on dataset (" + str(len(all_words_probability_with_modified)/100*10) + "): " + str(modifications_in_highest_10_percent)
print "Modified words within highest 20 percent on dataset (" + str(len(all_words_probability_with_modified)/100*20) + "): " + str(modifications_in_highest_20_percent)
print "Modified words within highest 30 percent on dataset (" + str(len(all_words_probability_with_modified)/100*30) + "): " + str(modifications_in_highest_30_percent)
print "Modified words/unigram probability within highest 100 words on dataset: " + str(unigram_modifications_in_highest_100)
print "Modified words/unigram probability within highest 500 words on dataset: " + str(unigram_modifications_in_highest_500)
print "Modified words/unigram probability within highest 1000 words on dataset: " + str(unigram_modifications_in_highest_1000)
print "Modified words/unigram probability within highest 4000 words on dataset: " + str(unigram_modifications_in_highest_4000)
print "Modified words/unigram probability within highest 10 percent on dataset (" + str(len(all_words_probability_with_modified)/100*10) + "): " + str(unigram_modifications_in_highest_10_percent)
print "Modified words/unigram probability within highest 20 percent on dataset (" + str(len(all_words_probability_with_modified)/100*20) + "): " + str(unigram_modifications_in_highest_20_percent)
print "Modified words/unigram probability within highest 30 percent on dataset (" + str(len(all_words_probability_with_modified)/100*30) + "): " + str(unigram_modifications_in_highest_30_percent)

with open("./perplexity_supervised.txt", "a") as f:
	content = fileName
	#content += "System Precision: " + str(precision)
	#content += "System Recall: " + str(recall)
	#content += "System F1-Score: " + str(2*((precision*recall)/(precision+recall)))
	content += ", Modified words within highest 100 words on dataset, " + str(modifications_in_highest_100)
	content += ", Modified words within highest 500 words on dataset, " + str(modifications_in_highest_500)
	content += ", Modified words within highest 1000 words on dataset, " + str(modifications_in_highest_1000)
	content += ", Modified words within highest 4000 words on dataset, " + str(modifications_in_highest_4000)
	content += ", Modified words within highest 10 percent on dataset (" + str(len(all_words_probability_with_modified)/100*10) + "), " + str(modifications_in_highest_10_percent)
	content += ", Modified words within highest 20 percent on dataset (" + str(len(all_words_probability_with_modified)/100*20) + "), " + str(modifications_in_highest_20_percent)
	content += ", Modified words within highest 30 percent on dataset (" + str(len(all_words_probability_with_modified)/100*30) + "), " + str(modifications_in_highest_30_percent)
	content += ", Modified words/unigram probability within highest 100 words on dataset, " + str(unigram_modifications_in_highest_100)
	content += ", Modified words/unigram probability within highest 500 words on dataset, " + str(unigram_modifications_in_highest_500)
	content += ", Modified words/unigram probability within highest 1000 words on dataset, " + str(unigram_modifications_in_highest_1000)
	content += ", Modified words/unigram probability within highest 4000 words on dataset, " + str(unigram_modifications_in_highest_4000)
	content += ", Modified words/unigram probability within highest 10 percent on dataset (" + str(len(all_words_probability_with_modified)/100*10) + "), " + str(unigram_modifications_in_highest_10_percent)
	content += ", Modified words/unigram probability within highest 20 percent on dataset (" + str(len(all_words_probability_with_modified)/100*20) + "), " + str(unigram_modifications_in_highest_20_percent)
	content += ", Modified words/unigram probability within highest 30 percent on dataset (" + str(len(all_words_probability_with_modified)/100*30) + "), " + str(unigram_modifications_in_highest_30_percent)
	f.write("{}\n".format(content))






