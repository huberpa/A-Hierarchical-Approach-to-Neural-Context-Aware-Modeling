
# Commandline Arguments
##############################################
import optparse
parser = optparse.OptionParser()
parser.add_option('--model_path', action="store", dest="model_path", help="The path to the model file (default: '')", default="")
parser.add_option('--data_path', action="store", dest="data_path", help="The path to the dev or test data that should be tested on, including index_to_word and word_to_index (default: .)", default=".")
parser.add_option('-u', '--unigram', action="store", dest="unigram", help="relative path to the dataset that was trained on (default: .)", default=".")
parser.add_option('--save_file', action="store", dest="save_file", help="path to the file where the results should be stored (default: .)", default=".")

options, args = parser.parse_args()
model_path = options.model_path
data_path = options.data_path
save_file = options.save_file
training_path = options.unigram
batch_size = 200
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

# Load files
print "Reading network input data..."
with open (data_path+"/encoder_input_data.txt", 'r') as f:
    encoder_input_data = json.load(f)
with open (data_path+"/decoder_input_data.txt", 'r') as f:
    decoder_input_data = json.load(f)
with open (data_path+"/decoder_original_output_data.txt", 'r') as f:
    decoder_original_output_data = json.load(f)
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
decoder_original_output_data_batch = createBatch(decoder_original_output_data, batch_size)
encoder_input_length_batch = createBatch(encoder_input_length, batch_size)
decoder_input_length_batch = createBatch(decoder_input_length, batch_size)

# Unigram probabilities from training
unigram_path  = open(training_path, "r")
unigram_text = unigram_path.read().decode('utf8')
u_words = nltk.word_tokenize(unigram_text)
u_words = [word.lower() for word in u_words]
wordcount = Counter(u_words)
wordUnigram = sorted(wordcount.items(), key=lambda item: item[1])
unigrams = dict((word, count) for word, count in wordUnigram)

# Launch the graph
print "Launch the graph..."
session_config = tf.ConfigProto(allow_soft_placement=True)    
session_config.gpu_options.per_process_gpu_memory_fraction = 0.90

with tf.Session(config=session_config) as session:

	tf.train.import_meta_graph(model_path + ".meta").restore(session, model_path)
	variables = tf.get_collection('variables_to_store')
	
	# Training
	print "Start testing..."

	unigrams_not_found = 0
	results = []
	perplexity_results = []
	for batch_index,_ in enumerate(encoder_input_data_batch):
		feed = {}
		feed["encoder_inputs"] = encoder_input_data_batch[batch_index]
		feed["encoder_length"] = encoder_input_length_batch[batch_index]
		feed["decoder_inputs"] = decoder_input_data_batch[batch_index]
		feed["decoder_length"] = decoder_input_length_batch[batch_index]

		result_raw = session.run(variables[0], feed_dict={variables[3]:feed["encoder_inputs"], variables[4]:feed["decoder_inputs"], variables[7]: feed["encoder_length"], variables[8]: feed["decoder_length"]})
		result_raw.tolist()

		for idx1, sentence in enumerate(result_raw):
			perplexity = 0
			perplexity_count = 0
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
					if original_word != "<UNKWN>":
						if word_probabilities[word_index] > 0:
							perplexity_count += 1
							perplexity += (math.log(word_probabilities[word_index], 2))
						results.append([str(word_probabilities[word_index]), str((word_probabilities[word_index]*len(u_words))/count_unigram), word_modified, original_word])
			
			perplexity = -perplexity/perplexity_count
			perplexity_results.append(int(2**(perplexity)))

	with open(data_path+"/tests/"+save_file+"_word_probability.txt",'a') as f:
		json.dump(results, f)
		
	with open(data_path+"/tests/"+save_file+"_sentence_perplexity.txt",'a') as f:
		json.dump(perplexity_results, f)

with open(data_path+"/tests/"+save_file+"_word_probability.txt",'r') as f:
	results =json.load(f)

with open(data_path+"/tests/"+save_file+"_sentence_perplexity.txt",'r') as f:
	perplexity_results =json.load(f)

trueResults = []
for element in results:
	trueResults.append([float(element[0]), float(element[1]), element[2], element[3]])

#Sort all the probabilities to find modified words
results.sort(key=lambda row: row[0])
results_modified = []
modifications_in_lowest_4000 = 0
unkwns_in_lowest_4000 = 0
for idx, element in enumerate(results):
	if element[3] == "<UNKWN>":
		unkwns_in_lowest_4000 += 1
	if element[2] == 1:
		results_modified.append([idx, element[3], element[1]])
		if idx <= 4000:
			modifications_in_lowest_4000 += 1

with open(data_path+"/tests/"+save_file+"_results.txt",'a') as f:
	f.write("{}\n".format("Number of words modified and in results_modified: "+str(len(results_modified))))
	f.write("{}\n".format("modifications_in_lowest_4000: "+str(modifications_in_lowest_4000)))
	f.write("{}\n".format("Number of <UNKWN> words: "+str(unkwns_in_lowest_4000)))

results.sort(key=lambda row: row[1])
results_modified = []
modifications_in_lowest_4000 = 0
unkwns_in_lowest_4000 = 0
for idx, element in enumerate(results):
	if element[3] == "<UNKWN>":
		unkwns_in_lowest_4000 += 1
	if element[2] == 1:
		results_modified.append([idx, element[3], element[1]])
		if idx <= 4000:
			modifications_in_lowest_4000 += 1

with open(data_path+"/tests/"+save_file+"_results.txt", "a") as f:	
	f.write("{}\n".format("Number of words modified and in results_modified / unigram: "+str(len(results_modified))))
	f.write("{}\n".format("modifications_in_lowest_4000 / unigram: "+str(modifications_in_lowest_4000)))
	f.write("{}\n".format("Number of <UNKWN> words: "+str(unkwns_in_lowest_4000)))
	f.write("{}\n".format("Average sentence perplexity: "+str(np.mean(perplexity_results))))
	f.write("{}\n".format("Median sentence perplexity: "+str(np.median(perplexity_results))))

##############################################

