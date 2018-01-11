
# Commandline Arguments
##############################################
import optparse
parser = optparse.OptionParser()
parser.add_option('--model_path', action="store", dest="model_path", help="The path to the model file (default: '')", default="")
parser.add_option('--data_path', action="store", dest="data_path", help="The path to the dev or test data that should be tested on (default: .)", default=".")
parser.add_option('--unigram', action="store", dest="unigram", help="relative path to the dataset that was trained on (default: .)", default=".")
parser.add_option('-g', '--generation', action="store", dest="generation", help="generation", default="")
parser.add_option('-l', '--lr', action="store", dest="lr", help="learning rate", default="")
parser.add_option('-s', '--calculate_or_display', action="store", dest="calculate_or_display", help="Calculate result or display Evaluation? (calc or disp) (default: calc)", default="calc")
parser.add_option('-d', '--disp_path', action="store", dest="disp_path", help="Path to the raw data file (default .)", default=".")
parser.add_option('--test_thres', action="store", dest="test_thres", help="Threshold for TEST set (default: INACTIVE)", default="INACTIVE")
arser.add_option('--vocab', action="store", dest="vocab", help="relative path to the vocabulary folder", default="")


options, args = parser.parse_args()
model_path = options.model_path
data_path = options.data_path
training_path = options.unigram
batch_size = 100
vocab = options.vocab
generation = options.generation
lr = options.lr
calculate_or_display = options.calculate_or_display
disp_path = options.disp_path
test_thres = options.test_thres
##############################################


# Imports
##############################################
import tensorflow as tf 
import numpy as np
import json
import os
import nltk
from collections import Counter
import math
##############################################


# Helper functions
##############################################
def createBatch(listing, batchSize):
	length = len(listing)
	batchList = []
	for index in range(0, length, batchSize):
		if index + batchSize < length:
			batchList.append(listing[index:(index + batchSize)])
	return batchList

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
##############################################



# Main
##############################################

if calculate_or_display == "calc":

	# Load files
	path  = open(input_path, "r")
	text = path.read().decode('utf8')
	tokens = []
	for index, sentence in enumerate(nltk.sent_tokenize(text)): 
	    tokens.append(nltk.word_tokenize(sentence))
	tokens = [[word.lower() for word in sentence] for sentence in tokens]

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

	# Unigram probabilities from training
	unigram_path  = open(training_path, "r")
	unigram_text = unigram_path.read().decode('utf8')
	u_words = nltk.word_tokenize(unigram_text)
	u_words = [word.lower() for word in u_words]
	wordcount = Counter(u_words)
	wordUnigram = sorted(wordcount.items(), key=lambda item: item[1])
	unigrams = dict((word, count) for word, count in wordUnigram)

	print "Launch the graph..."
	session_config = tf.ConfigProto(allow_soft_placement=True)    
	session_config.gpu_options.per_process_gpu_memory_fraction = 0.90

	with tf.Session(config=session_config) as session:

		tf.train.import_meta_graph(model_path + ".meta").restore(session, model_path)
		variables = tf.get_collection('variables_to_store')

		unigrams_not_found = 0
		results = []
		perplexity_results = []
		perplexity = 0
		perplexity_count = 0

		for batch_index,_ in enumerate(encoder_input_data_batch):
			feed = {}
			feed["encoder_inputs"] = encoder_input_data_batch[batch_index]
			feed["encoder_length"] = encoder_input_length_batch[batch_index]
			feed["decoder_inputs"] = decoder_input_data_batch[batch_index]
			feed["decoder_length"] = decoder_input_length_batch[batch_index]

			result_raw = session.run(variables[0], feed_dict={variables[3]:feed["encoder_inputs"], variables[4]:feed["decoder_inputs"], variables[7]: feed["encoder_length"], variables[8]: feed["decoder_length"]})
			result_raw.tolist()

			for idx1, sentence in enumerate(result_raw):
				for idx2, word in enumerate(sentence):
					word_modified = 0
					word_probabilities = softmax(word)
					original_word = decoder_original_output_data_batch[batch_index][idx1][idx2]
					checkedword = original_word
					if original_word.find("___") != -1:
						word_modified = 1
						checkedword = original_word[original_word.find("___")+3:original_word.rfind("___")]

					count_unigram = 1
					try:
						count_unigram = unigrams[str(checkedword)]
					except Exception:
						unigrams_not_found += 1

					word_index = word_to_index[checkedword] if checkedword in word_to_index else word_to_index["<UNKWN>"]
					if original_word != "<PAD>":
						perplexity_count += 1
						if original_word != "<UNKWN>":
							if word_probabilities[word_index] > 0:
								perplexity += (math.log(word_probabilities[word_index], 2))
							results.append([str(word_probabilities[word_index]), str((word_probabilities[word_index]*len(u_words))/count_unigram), word_modified, original_word])
				

		perplexity = -perplexity/perplexity_count
		perplexity_results.append(int(2**(perplexity)))

		# JSON DUMP all_words_probability_with_modified
		with open("results/context_raw_unsupervised_"+attn+"_"+lr+"_"+generation,'w') as f:
			json.dump(results, f)

		# JSON DUMP perplexity
		with open("results/context_perplexity_unsupervised_"+attn+"_"+lr+"_"+generation,'w') as f:
			json.dump(perplexity_results, f)

		print "done calculation"

else:

	with open(disp_path,'r') as f:
		results =json.load(f)

	#Sort all the probabilities to find modified words
	results.sort(key=lambda row: row[0])
	all_modifications = 0

	# Get all replacements
	for idx, element in enumerate(results):
		if element[2] == 1:
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
			for idx, element in enumerate(results[:endpoint]):
				if element[2] == 1:
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
		results.sort(key=lambda row: row[1])

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
			for idx, element in enumerate(results[:endpoint]):
				if element[2] == 1:
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
		for idx, element in enumerate(results[:endpoint]):
			if element[2] == 1:
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
		results.sort(key=lambda row: row[1])

		# Get replacements within thresholds
		f1_score_per_threshold_unigram = []

		# Thresholds 
		num_rep_words = 0
		num_all_words = 0
		f1 = 0
		prec = 0
		rec = 0

		# Replacements
		for idx, element in enumerate(results[:endpoint]):
			if element[2] == 1:
				num_rep_words += 1
			num_all_words += 1

		prec = float(num_rep_words) / float(num_all_words)
		rec = float(num_rep_words) / float(all_modifications)

		if (prec+rec) > 0:
			f1 = 2*((prec*rec)/(prec+rec))
		else:
			f1 = 0
		f1_score_per_threshold_unigram.append([num_all_words, num_rep_words, f1, prec, rec])

	print "save: " + str(f1_score_per_threshold_unigram[0])

	with open("results/context_threshold_evaluation_unsupervised_"+attn+"_"+lr+"_"+generation,'w') as f:

		for element in f1_score_per_threshold:
			content = str(element[0]) + ";" + str(element[1]) + ";" + str(element[2]) + ";" + str(element[3]) + ";" + str(element[4])
			f.write("{}\n".format(content))

		f.write("{}\n".format(""))

		for element in f1_score_per_threshold_unigram:
			content = str(element[0]) + ";" + str(element[1]) + ";" + str(element[2]) + ";" + str(element[3]) + ";" + str(element[4])
			f.write("{}\n".format(content))

##############################################

