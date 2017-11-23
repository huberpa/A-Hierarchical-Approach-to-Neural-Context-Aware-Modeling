
# Commandline Arguments
##############################################
import optparse
parser = optparse.OptionParser()
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
training_path = options.unigram
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
import numpy as np
import nltk
##############################################



# Main
##############################################

if calculate_or_display == "calc":

	# Load the model to be tested against and output the structure
	print "Load Model"
	model = load_model(model_path)
	print model.summary() 

	# Load the file with the talks and tokenize
	path  = open(input_path, "r")
	text = path.read().decode('utf8')
	tokens = []
	for index, sentence in enumerate(nltk.sent_tokenize(text)): 
	    tokens.append(nltk.word_tokenize(sentence))
	tokens = [[word.lower() for word in sentence] for sentence in tokens]


	# Unigram probabilities from training
	unigram_path  = open(training_path, "r")
	unigram_text = unigram_path.read().decode('utf8')
	words = nltk.word_tokenize(unigram_text)
	words = [word.lower() for word in words]
	wordcount = Counter(words)
	wordUnigram = sorted(wordcount.items(), key=lambda item: item[1])

	unigrams = dict((word, count) for word, count in wordUnigram)

	# Load the index_to_word and word_to_index files to convert indizes into words
	word_to_index = []
	with open(vocab_path + "/word_to_index.json") as f:    
		word_to_index = json.load(f)

	index_to_word = []
	with open(vocab_path + "/index_to_word.json") as f:    
		index_to_word = json.load(f)

	# Compute the perplexity for each sentence
	probability_orig_mods = []
	probability_with_unigram = []
	sentence_level_perplexity = []
	sentence_level_perplexity_filtered = []
	words_not_in_vocab = 0
	all_words_probability_with_modified = []

	perplexity = np.float128(0.0)
	perplexity_count = 0

	for sentence in tokens:
		# For the start token
		perplexity_count += 1
		for word in sentence:
			perplexity_count += 1

	# Iterate through all the sentences in the data and compute the perplexity
	for index, sentence in enumerate(tokens):
		value_sentence = []

		# Adding start and end token to the sentence
		cut_sentence = ["<START>"] + sentence + ["<END>"]

		# Cutting sentences longer than 50 words
		if len(cut_sentence) > 50:
			cut_sentence = cut_sentence[:50]

		# Convert words into indexes and mark modified words
		changes = []
		for index, word in enumerate(cut_sentence):
			checkedWord = ""

			# Check if there is the "___" in the word --> the indicator for a modified word
			if word.find("___") == -1:
				checkedWord = word
			else:
				checkedWord = word[:word.find("___")]
				changes.append([(index-1), word[word.find("___")+3:]]) # Index-1 because the output that predicts the input predicts that one timestep in advance.
			
			# Convert word into the according index
			try:
				value_sentence.append(word_to_index[checkedWord])
			except Exception:
				value_sentence.append(word_to_index["<UNKWN>"])

		# Create input and output vectors
		input_perplexity = value_sentence[:len(cut_sentence)-1]
		output_perplexity = value_sentence[1:]
		length = len(input_perplexity)

		# Save input vector in numpy array
		testInput = np.zeros((1, length), dtype=np.int16)
		for index, idx in enumerate(input_perplexity):
			testInput[0, index] = idx

		# Predict the next words
		prediction = model.predict(testInput, verbose=0)[0]

		# Evaluate the prediction
		for index, word in enumerate(prediction):

			# Add the probability of the correct output word to thw array (! This will be overwritten if the word is a modified one !)
			count_o = 1
			try:
				count_o = unigrams[index_to_word[str(output_perplexity[index])]]
			except Exception:
				count_o = count_o

			all_words_probability_with_modified.append([str(word[output_perplexity[index]]), 0, str((word[output_perplexity[index]]*len(words))/(count_o)), index_to_word[str(output_perplexity[index])]])

			#check if word got replaced and if so, save in special array
			for change in changes:
				if change[0] == index:
					try:	

						# Get rank of original and modified word			
						better_alternatives_original = 1
						better_alternatives_modified = 1

						for selection in word:
							if selection > word[word_to_index[change[1]]]:
								better_alternatives_modified += 1
							if selection > word[output_perplexity[index]]:
								better_alternatives_original += 1

						# Get unigram probability
						count_o = 1
						count_m = 1
						try:
							count_o = unigrams[index_to_word[str(output_perplexity[index])]]
							count_m = unigrams[change[1]]
						except Exception:
							count_o = count_o

						# Substitude the original word with the modified one -> try to find these later
						all_words_probability_with_modified[len(all_words_probability_with_modified)-1][0] = str(word[word_to_index[change[1]]])
						all_words_probability_with_modified[len(all_words_probability_with_modified)-1][1] = 1
						all_words_probability_with_modified[len(all_words_probability_with_modified)-1][2] = str((word[word_to_index[change[1]]]*len(words))/(count_m))
						all_words_probability_with_modified[len(all_words_probability_with_modified)-1][3] = change[1]

						print "replaced word"

						# Save analytics for changed words in list
						probability_orig_mods.append([[word[output_perplexity[index]], (word[output_perplexity[index]]*len(words))/(count_o), better_alternatives_original], [word[word_to_index[change[1]]], (word[word_to_index[change[1]]]*len(words))/(count_m), better_alternatives_modified]])

					except Exception:
						words_not_in_vocab += 1

			if word[output_perplexity[index]] > 0.0:
				perplexity += (math.log(word[output_perplexity[index]], 2))

	perplexity = -perplexity/perplexity_count
	sentence_level_perplexity.append(int(2**(perplexity)))

	# JSON DUMP all_words_probability_with_modified
	with open("results/baseline_raw_unsupervised_"+lr+"_"+generation,'w') as f:
		json.dump(all_words_probability_with_modified, f)

	# JSON DUMP perplexity
	with open("results/baseline_perplexity_unsupervised_"+lr+"_"+generation,'w') as f:
		json.dump(sentence_level_perplexity, f)

	print "done calculation"

else:

	with open(disp_path,'r') as f:
		all_words_probability_with_modified = json.load(f)

	#Sort all the probabilities to find modified words
	all_words_probability_with_modified.sort(key=lambda row: row[0])
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
		all_words_probability_with_modified.sort(key=lambda row: row[2])

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
		print "start threshold TEST calculation"

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
		all_words_probability_with_modified.sort(key=lambda row: row[2])

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
		

	with open("results/baseline_threshold_evaluation_unsupervised_"+lr+"_"+generation,'w') as f:

		for element in f1_score_per_threshold:
			content = str(element[0]) + ";" + str(element[1]) + ";" + str(element[2]) + ";" + str(element[3]) + ";" + str(element[4])
			f.write("{}\n".format(content))

		f.write("{}\n".format(""))

		for element in f1_score_per_threshold_unigram:
			content = str(element[0]) + ";" + str(element[1]) + ";" + str(element[2]) + ";" + str(element[3]) + ";" + str(element[4])
			f.write("{}\n".format(content))


