
# Commandline Arguments
##############################################
import optparse
parser = optparse.OptionParser()
parser.add_option('--model_path', action="store", dest="model_path", help="The path to the model file (default: '')", default="")
parser.add_option('--data_path', action="store", dest="data_path", help="The path to the dev or test data that should be tested on, including index_to_word and word_to_index (default: .)", default=".")

options, args = parser.parse_args()
model_path = options.model_path
data_path = options.data_path
##############################################


# Imports
##############################################
import tensorflow as tf 
import numpy as np
import json
import os
import sys
import nltk
from collections import Counter
import math
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
##############################################



# Main
##############################################

# Load files
print "Reading network input data..."
#with open (data_path+"/input_data.txt", 'r') as f:
#	input_data = json.load(f)
with open (data_path+"/index_to_word_eng.txt", 'r') as f:
	index_to_word_eng = json.load(f)
with open (data_path+"/word_to_index_eng.txt", 'r') as f:
	word_to_index_eng = json.load(f)
with open (data_path+"/index_to_word_ger.txt", 'r') as f:
	index_to_word_ger = json.load(f)

final_input_data = []
input_data = "how much is the tomato? i want to play football. i'm really tall. which way is right? i'm really annoyed! so many possibilities. how can anyone say that they are really on the planet earth? last sentence."
tokenized_input = nltk.sent_tokenize(input_data)

print tokenized_input

for element in tokenized_input:
	final_input_data.append(["<START>"] + nltk.word_tokenize(element) + ["<END>"])


for index1, sentence in enumerate(final_input_data):
	for index2, word in enumerate(sentence):
		final_input_data[index1][index2] = word_to_index_eng[word] if word in word_to_index_eng else word_to_index_eng["<UNKWN>"]

print final_input_data

encoder_length = []
for sentence in final_input_data:
	counter = 0
	for word in sentence:
		if word != 0:
			counter = counter + 1
	encoder_length.append(counter)

# PADDING
for index, element in enumerate(final_input_data):
	final_input_data[index] = element + [0]*(100-len(final_input_data[index]))

print final_input_data

# Split data into batches
print "Split data into batches..."
batch_size = 1
input_data_batch = createBatch(final_input_data, batch_size)
encoder_length_batch = createBatch(encoder_length, batch_size)

print input_data_batch
print encoder_length_batch

# Launch the graph
print "Launch the graph..."
session_config = tf.ConfigProto(allow_soft_placement=True)    
session_config.gpu_options.per_process_gpu_memory_fraction = 0.90

with tf.Session(config=session_config) as session:
	tf.train.import_meta_graph(model_path + ".meta").restore(session, model_path)
	variables = tf.get_collection('variables_to_store')
	
	print "Start testing..."
	for batch_index,_ in enumerate(input_data_batch):
		
		feed = {}
		feed["encoder_inputs"] = input_data_batch[batch_index]
		feed["encoder_length"] = encoder_length_batch[batch_index]
		feed["decoder_length"] = [99]*batch_size
		feed["start_token_infer"] = [word_to_index_eng["<START>"]]*batch_size

		test_output = session.run(variables[9], feed_dict={variables[3]:feed["encoder_inputs"], variables[7]: feed["encoder_length"], variables[8]: feed["decoder_length"], variables[10]: feed["start_token_infer"]})
		print test_output
		
		for index, batch in enumerate(test_output.tolist()):
			orig_sentence = ""
			test_sentence = ""
			for word in input_data_batch[batch_index][index]:
				if word != 0:
					orig_sentence = orig_sentence + " " + index_to_word_eng[str(word)]
			for word in batch:
				if word != word_to_index_eng["<END>"]:
					test_sentence = test_sentence + " " + index_to_word_ger[str(word)]
			print "The english sentence " + orig_sentence
			print "is translated into " + test_sentence

##############################################

