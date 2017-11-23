
# Commandline Arguments
##############################################
import optparse
parser = optparse.OptionParser()
parser.add_option('--model_path', action="store", dest="model_path", help="The path to the model file (default: '')", default="")
parser.add_option('--data_path', action="store", dest="data_path", help="The path to the dev or test data that should be tested on, including index_to_word and word_to_index (default: .)", default=".")
parser.add_option('--save_file', action="store", dest="save_file", help="path to the file where the results should be stored (default: .)", default=".")
parser.add_option('-g', '--generation', action="store", dest="generation", help="generation", default="")
parser.add_option('-l', '--lr', action="store", dest="lr", help="learning rate", default="")
parser.add_option('-s', '--calculate_or_display', action="store", dest="calculate_or_display", help="Calculate result or display Evaluation? (calc or disp) (default: calc)", default="calc")
parser.add_option('-d', '--disp_path', action="store", dest="disp_path", help="Path to the raw data file (default .)", default=".")
parser.add_option('--attn', action="store", dest="attn", help="Attention (Attn if so)", default="")
parser.add_option('--test_thres', action="store", dest="test_thres", help="Threshold for TEST set (default: INACTIVE)", default="INACTIVE")

options, args = parser.parse_args()
model_path = options.model_path
data_path = options.data_path
save_file = options.save_file
batch_size = 1
generation = options.generation
lr = options.lr
calculate_or_display = options.calculate_or_display
disp_path = options.disp_path
attn = options.attn
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

##############################################



# Main
##############################################

if calculate_or_display == "calc":

	# Load files
	print "Reading network input data..."
	with open (data_path+"/encoder_input_data.txt", 'r') as f:
	    encoder_input_data = json.load(f)
	with open (data_path+"/decoder_input_data.txt", 'r') as f:
	    decoder_input_data = json.load(f)
	with open (data_path+"/decoder_output_data.txt", 'r') as f:
	    decoder_output_data = json.load(f)
	with open (data_path+"/encoder_input_length.txt", 'r') as f:
	    encoder_input_length = json.load(f)
	with open (data_path+"/decoder_input_length.txt", 'r') as f:
	    decoder_input_length = json.load(f)
	with open (data_path+"/index_to_word.txt", 'r') as f:
	    index_to_word = json.load(f)
	with open (data_path+"/word_to_index.txt", 'r') as f:
	    word_to_index = json.load(f)

	# Retrieve input variables from files
	print "Retrieve input variables from files..."
	enc_input_dimension = len(encoder_input_data[0][0])
	enc_timesteps_max = len(encoder_input_data[0])
	dec_timesteps_max = len(decoder_input_data[0])
	vocab_size = len(index_to_word)

	# Split data into batches
	print "Split data into batches..."
	encoder_input_data_batch = createBatch(encoder_input_data, batch_size)
	decoder_input_data_batch = createBatch(decoder_input_data, batch_size)
	decoder_output_data_batch = createBatch(decoder_output_data, batch_size)
	encoder_input_length_batch = createBatch(encoder_input_length, batch_size)
	decoder_input_length_batch = createBatch(decoder_input_length, batch_size)

	# Launch the graph
	print "Launch the graph..."
	session_config = tf.ConfigProto(allow_soft_placement=True)    
	session_config.gpu_options.per_process_gpu_memory_fraction = 0.90

	with tf.Session(config=session_config) as session:

		tf.train.import_meta_graph(model_path + ".meta").restore(session, model_path)
		variables = tf.get_collection('variables_to_store')
		
		# Training
		print "Start testing..."

		results = []
		for batch_index,_ in enumerate(encoder_input_data_batch):
			feed = {}
			feed["encoder_inputs"] = encoder_input_data_batch[batch_index]
			feed["encoder_length"] = encoder_input_length_batch[batch_index]
			feed["decoder_inputs"] = decoder_input_data_batch[batch_index]
			feed["decoder_length"] = decoder_input_length_batch[batch_index]

			result_raw = session.run(variables[0], feed_dict={variables[3]:feed["encoder_inputs"], variables[4]:feed["decoder_inputs"], variables[7]: feed["encoder_length"], variables[8]: feed["decoder_length"]})
			result_raw.tolist()
			for idx1, sentence in enumerate(result_raw):
				for idx2, probability in enumerate(sentence):
					word_modified = decoder_output_data_batch[batch_index][idx1][idx2]
					if decoder_input_data_batch[batch_index][idx1][idx2] != word_to_index["<PAD>"]:
						if decoder_input_data_batch[batch_index][idx1][idx2] != word_to_index["<UNKWN>"]:
							results.append([str(probability[0]), str(probability[1]), word_modified, index_to_word[str(decoder_input_data_batch[batch_index][idx1][idx2])]])
				
		# JSON DUMP all_words_probability_with_modified
		with open("results/context_raw_supervised_"+attn+"_"+lr+"_"+generation,'w') as f:
			json.dump(results, f)

		print "done calculation"

else:

	with open(disp_path,'r') as f:
		results =json.load(f)

	trueResults = []
	for element in results:
		trueResults.append([float(element[0]), float(element[1]), element[2], element[3]])

	#Sort all the probabilities to find modified words
	trueResults.sort(key=lambda row: row[1], reverse=True)
	all_modifications = 0

	# Get all replacements
	for idx, element in enumerate(trueResults):
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
			for idx, element in enumerate(trueResults[:endpoint]):
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
		for idx, element in enumerate(trueResults[:endpoint]):
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

	with open("results/context_threshold_evaluation_supervised_"+attn+"_"+lr+"_"+generation,'w') as f:

		for element in f1_score_per_threshold:
			content = str(element[0]) + ";" + str(element[1]) + ";" + str(element[2]) + ";" + str(element[3]) + ";" + str(element[4])
			f.write("{}\n".format(content))
##############################################
