
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
batch_size = 1
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
import sys
reload(sys)  
sys.setdefaultencoding('utf-8')
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

# Open files
print "Reading network input data..."
with open (data_path+"/input_data.txt", 'r') as f:
    encoder_data = json.load(f)
with open (data_path+"/output_data.txt", 'r') as f:
    decoder_data = json.load(f)
with open (data_path+"/index_to_word_eng.txt", 'r') as f:
    index_to_word_english = json.load(f)
with open (data_path+"/index_to_word_ger.txt", 'r') as f:
    index_to_word_german = json.load(f)	
with open (data_path+"/word_to_index_ger.txt", 'r') as f:
	word_to_index_german = json.load(f)	

# Variables for the network execution
print "Calculate dynamic lengths..."
encoder_length = []
decoder_length = []
decoder_input_data = []
decoder_output_data = []

for sentence in encoder_data:
	counter = 0
	for word in sentence:
		if word != 0:
			counter = counter + 1
	encoder_length.append(counter)

for idx, sentence in enumerate(decoder_data):
	decoder_output_data.append(sentence[1:len(sentence)])
	try:
		sentence.remove(word_to_index_german["<END>"])
	except ValueError:
		sentence = sentence[:len(sentence)-1]

	decoder_input_data.append(sentence)

for sentence in decoder_input_data:
	counter = 0
	for word in sentence:
		if word != 0:
			counter = counter + 1
	decoder_length.append(counter)


# Split data into batches
print "Split data into batches..."
encoder_input_data_batch = createBatch(encoder_data, batch_size)
decoder_input_data_batch = createBatch(decoder_input_data, batch_size)
decoder_output_data_batch = createBatch(decoder_output_data, batch_size)
encoder_length_batch = createBatch(encoder_length, batch_size)
decoder_length_batch = createBatch(decoder_length, batch_size)

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

	print len(encoder_input_data_batch)

	unigrams_not_found = 0
	results = []
	perplexity_results = []
	for batch_index,_ in enumerate(encoder_input_data_batch):
		feed = {}
		feed["encoder_inputs"] = encoder_input_data_batch[batch_index]
		feed["encoder_length"] = encoder_length_batch[batch_index]
		feed["decoder_inputs"] = decoder_input_data_batch[batch_index]
		feed["decoder_length"] = decoder_length_batch[batch_index]

		result_raw = session.run(variables[0], feed_dict={variables[3]:feed["encoder_inputs"], variables[4]:feed["decoder_inputs"], variables[7]: feed["encoder_length"], variables[8]: feed["decoder_length"]})
		result_raw.tolist()

		for idx1, sentence in enumerate(result_raw):
			perplexity = 0
			perplexity_count = 0
			for idx2, word in enumerate(sentence):
				word = softmax(word)

				count_unigram = 1
				try:
					count_unigram = unigrams[str(word)]
				except Exception:
					unigrams_not_found += 1

				if idx2 < len(sentence)-1:
					perplexity_count += 1
					perplexity += (math.log(word[decoder_output_data_batch[batch_index][idx1][idx2]], 2))

			perplexity = -perplexity/perplexity_count
			perplexity_results.append(int(2**(perplexity)))

with open(data_path+"/tests/"+save_file+"_results.txt", "a") as f:	
	f.write("{}\n".format("Average sentence perplexity: "+str(np.mean(perplexity_results))))
	f.write("{}\n".format("Median sentence perplexity: "+str(np.median(perplexity_results))))

##############################################

