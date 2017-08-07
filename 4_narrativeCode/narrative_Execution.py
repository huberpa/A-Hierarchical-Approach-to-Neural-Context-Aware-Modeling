
# Commandline Arguments
##############################################
import optparse
parser = optparse.OptionParser()
parser.add_option('--model_path', action="store", dest="model_path", help="The path to the model checkpoint file (default: '')", default="")
parser.add_option('--data_path', action="store", dest="data_path", help="The path to the dev or test data that should be tested on, including index_to_word and word_to_index (default: .)", default=".")
options, args = parser.parse_args()
model_path = options.model_path
data_path = options.data_path
batch_size = 1
##############################################


# Imports
##############################################
import tensorflow as tf 
import numpy as np
import json
import os
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
with open (data_path+"/decoder_output_data.txt", 'r') as f:
    decoder_output_data = json.load(f)
with open (data_path+"/decoder_original_output_data.txt", 'r') as f:
    decoder_original_output_data = json.load(f)
with open (data_path+"/encoder_input_length.txt", 'r') as f:
    encoder_input_length = json.load(f)
with open (data_path+"/decoder_input_length.txt", 'r') as f:
    decoder_input_length = json.load(f)
with open (data_path+"/decoder_mask.txt", 'r') as f:
    decoder_mask = json.load(f)
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
decoder_original_output_data_batch = createBatch(decoder_original_output_data, batch_size)
encoder_input_length_batch = createBatch(encoder_input_length, batch_size)
decoder_input_length_batch = createBatch(decoder_input_length, batch_size)
decoder_mask_batch = createBatch(decoder_mask, batch_size)

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
	for batch_index,_ in enumerate(encoder_input_data_batch[:2]):
		feed = {}

		# Inputs
		feed["encoder_inputs"] = encoder_input_data_batch[batch_index]
		feed["encoder_length"] = encoder_input_length_batch[batch_index]
		feed["decoder_inputs"] = decoder_input_data_batch[batch_index]
		feed["decoder_length"] = decoder_input_length_batch[batch_index]
		
		# For checking if it's a modified value
		feed["decoder_outputs"] = decoder_output_data_batch[batch_index]
		feed["mask"] = decoder_mask_batch[batch_index]

		result_raw = session.run(variables[0], feed_dict={variables[3]:feed["encoder_inputs"], variables[4]:feed["decoder_inputs"], variables[7]: feed["encoder_length"], variables[8]: feed["decoder_length"]})
		results.append(result_raw)

	# Have all the data for the corpus
	# Make it probabilities
	todo

	# get values for the real/modified words (indicate modified ones with (value, 1))
	for idx1, batch in enumerate(results):
		for idx2, sentence in enumerate(batch):
			for idx3, word in enumerate(single_batch):
				# Check for the score of the real word
				


	#print results[0][0]
	#print decoder_original_output_data_batch[0][0]



	# Sort the words
	# Get the most unlikely 4000result_probability = []
		

##############################################
