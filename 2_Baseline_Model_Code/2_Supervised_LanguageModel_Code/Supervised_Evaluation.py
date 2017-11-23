
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
parser.add_option('-g', '--generation', action="store", dest="generation", help="generation", default="")
parser.add_option('-l', '--lr', action="store", dest="lr", help="learning rate", default="")
parser.add_option('-s', '--calculate_or_display', action="store", dest="calculate_or_display", help="Calculate result or display Evaluation? (calc or disp) (default: calc)", default="calc")
parser.add_option('-d', '--disp_path', action="store", dest="disp_path", help="Path to the raw data file (default .)", default=".")
parser.add_option('--test_thres', action="store", dest="test_thres", help="Threshold for TEST set (default: INACTIVE)", default="INACTIVE")

options, args = parser.parse_args()
print options
input_path = options.input
unigram = options.unigram
model_path = options.model
vocab_path = options.vocab
fileName = options.file
generation = options.generation
lr = options.lr
calculate_or_display = options.calculate_or_display
disp_path = options.disp_path
test_thres = options.test_thres
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

if calculate_or_display == "calc":

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

	# Unigram probabilities from training
	unigram_path  = open(unigram, "r")
	unigram_text = unigram_path.read().decode('utf8')
	words = nltk.word_tokenize(unigram_text)
	words = [word.lower() for word in words]
	wordcount = Counter(words)
	wordUnigram = sorted(wordcount.items(), key=lambda item: item[1])

	unigrams = dict((word, count) for word, count in wordUnigram)

	# Compute the perplexity for each sentence
	words_not_in_vocab = 0
	all_words_probability_with_modified = []
	nb_changed_words = 0
	nb_changed_words_high_prob = 0
	nb_words_high_prob = 0
	unkwn_words = 0
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
			if index_to_word[str(input_perplexity[index])] != "<UNKWN>":

				count_o = 1
				try:
					count_o = unigrams[index_to_word[str(input_perplexity[index])]]
				except Exception:
					count_o = count_o

				all_words_probability_with_modified.append([str(word), 0, str((word*len(words))/(count_o)), index_to_word[str(input_perplexity[index])]])
			else:
				unkwn_words = unkwn_words + 1 
			#check if word got replaced and if so, save in special array
			for change in changes:
				if change[0] == index:
					try:	
						if word > 0.5:
							nb_changed_words_high_prob += 1

						all_words_probability_with_modified[len(all_words_probability_with_modified)-1][1] = 1
					except Exception:
						words_not_in_vocab += 1

	# JSON DUMP all_words_probability_with_modified
	with open("results/baseline_raw_supervised_"+lr+"_"+generation,'w') as f:
		json.dump(all_words_probability_with_modified, f)

	print "done calculation"


else:

	with open(disp_path,'r') as f:
		all_words_probability_with_modified = json.load(f)

	#Sort all the probabilities to find modified words
	all_words_probability_with_modified.sort(key=lambda row: row[0], reverse=True)
	all_modifications = 0

	# Get all replacements
	for idx, element in enumerate(all_words_probability_with_modified):
		if element[1] == 1:
			all_modifications += 1

	# Get replacements within thresholds
	f1_score_per_threshold = []

	if test_thres == "INACTIVE":
		# Thresholds 
		print "start threshold calculation"
		for endpoint in range(1, 200001,1000):
			
			print str(endpoint) + " / 200001"

			num_rep_words = 0
			num_all_words = 0
			f1 = 0
			prec = 0
			rec = 0

			# Replacements
			for idx, element in enumerate(all_words_probability_with_modified[:endpoint]):
				if element[1] == 1:
					num_rep_words += 1
				num_all_words += 1

			prec = float(num_rep_words) / float(num_all_words)
			rec = float(num_rep_words) / float(all_modifications)

			if (prec+rec) > 0:
				f1 = 2*((prec*rec)/(prec+rec))
			else:
				f1 = 0

			f1_score_per_threshold.append([num_all_words, num_rep_words, f1, prec, rec])


		#Sort all the probabilities to find modified words UNIGRANMS
		all_words_probability_with_modified.sort(key=lambda row: row[2], reverse=True)

		# Get replacements within thresholds
		f1_score_per_threshold_unigram = []

		# Thresholds 
		for endpoint in range(1, 200001,1000):
			
			print str(endpoint) + " / 200001 (unigram)"

			num_rep_words = 0
			num_all_words = 0
			f1 = 0
			prec = 0
			rec = 0

			# Replacements
			for idx, element in enumerate(all_words_probability_with_modified[:endpoint]):
				if element[1] == 1:
					num_rep_words += 1
				num_all_words += 1

			prec = float(num_rep_words) / float(num_all_words)
			rec = float(num_rep_words) / float(all_modifications)

			if (prec+rec) > 0:
				f1 = 2*((prec*rec)/(prec+rec))
			else:
				f1 = 0
			f1_score_per_threshold_unigram.append([num_all_words, num_rep_words, f1, prec, rec])

	else:
		# Thresholds 
		print "start threshold calculation"

		endpoint = int(test_thres)
		num_rep_words = 0
		num_all_words = 0
		f1 = 0
		prec = 0
		rec = 0

		# Replacements
		for idx, element in enumerate(all_words_probability_with_modified[:endpoint]):
			if element[1] == 1:
				num_rep_words += 1
			num_all_words += 1

		prec = float(num_rep_words) / float(num_all_words)
		rec = float(num_rep_words) / float(all_modifications)

		if (prec+rec) > 0:
			f1 = 2*((prec*rec)/(prec+rec))
		else:
			f1 = 0

		f1_score_per_threshold.append([num_all_words, num_rep_words, f1, prec, rec])


		#Sort all the probabilities to find modified words UNIGRANMS
		all_words_probability_with_modified.sort(key=lambda row: row[2], reverse=True)

		# Get replacements within thresholds
		f1_score_per_threshold_unigram = []

		# Thresholds 			
		num_rep_words = 0
		num_all_words = 0
		f1 = 0
		prec = 0
		rec = 0

		# Replacements
		for idx, element in enumerate(all_words_probability_with_modified[:endpoint]):
			if element[1] == 1:
				num_rep_words += 1
			num_all_words += 1

		prec = float(num_rep_words) / float(num_all_words)
		rec = float(num_rep_words) / float(all_modifications)

		if (prec+rec) > 0:
			f1 = 2*((prec*rec)/(prec+rec))
		else:
			f1 = 0
		f1_score_per_threshold_unigram.append([num_all_words, num_rep_words, f1, prec, rec])

	with open("results/baseline_threshold_evaluation_supervised_"+lr+"_"+generation,'w') as f:

		for element in f1_score_per_threshold:
			content = str(element[0]) + ";" + str(element[1]) + ";" + str(element[2]) + ";" + str(element[3]) + ";" + str(element[4])
			f.write("{}\n".format(content))

		f.write("{}\n".format(""))

		for element in f1_score_per_threshold_unigram:
			content = str(element[0]) + ";" + str(element[1]) + ";" + str(element[2]) + ";" + str(element[3]) + ";" + str(element[4])
			f.write("{}\n".format(content))




