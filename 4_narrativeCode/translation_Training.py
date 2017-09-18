
# Commandline Arguments
##############################################
import optparse
parser = optparse.OptionParser()
parser.add_option('--embedding_size', action="store", dest="embedding_size", help="The number of hidden layers in the model (default: 256)", default=256)
parser.add_option('--layer_number', action="store", dest="layer_number", help="The number of hidden layers in the model (default: 1)", default=1)
parser.add_option('--layer_dimension', action="store", dest="layer_dimension", help="The number of neurons in the hidden layer(s)  (default: 512)", default=512)
parser.add_option('--batch_size', action="store", dest="batch_size", help="The batch size of the model (default: 100)", default=100)
parser.add_option('--epochs', action="store", dest="epochs", help="The number of training epochs (default: 25)", default=25)
parser.add_option('--pre_path', action="store", dest="pre_path", help="The path to the preprocessed data folder (default: .)", default=".")
parser.add_option('--save_path', action="store", dest="save_path", help="The path to the monolingual data folder (default: .)", default=".")
parser.add_option('--name', action="store", dest="name", help="The name of the model (default: model)", default="model")

options, args = parser.parse_args()
embedding_size = int(options.embedding_size)
nb_hidden_layers = int(options.layer_number)
hidden_dimensions= int(options.layer_dimension)
batch_size = int(options.batch_size)
epochs = int(options.epochs)
data_path = options.pre_path
model_name = options.name
save_path = options.save_path
batch_size_inference = 1
##############################################


# Imports
##############################################
import tensorflow as tf 
from tensorflow.python.layers import core as layers_core
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variable_scope
import numpy as np
from xml.dom.minidom import parse, parseString
import json
import os
from collections import Counter
import datetime
import pickle
import sys
import copy
reload(sys)  
sys.setdefaultencoding('utf-8')
##############################################


# Helper functions
##############################################
def seq2seq(enc_input_dimension,enc_timesteps_max,dec_input_dimension, dec_timesteps_max,hidden_units,hidden_layers,embedding_size, start_of_sequence_id, end_of_sequence_id):

	# Inputs / Outputs / Cells
	encoder_inputs = tf.placeholder(dtypes.int64, shape=[None, enc_timesteps_max], name="encoder_inputs")
	encoder_lengths = tf.placeholder(dtypes.int32, shape=[None], name="encoder_lengths")
	decoder_inputs = tf.placeholder(dtypes.int64, shape=[None, dec_timesteps_max], name="decoder_inputs")
	decoder_lengths = tf.placeholder(dtypes.int32, shape=[None], name="decoder_lengths")
	decoder_outputs = tf.placeholder(dtypes.int64, shape=[None, dec_timesteps_max], name="decoder_outputs")
	masking = tf.placeholder(dtypes.float32, shape=[None, dec_timesteps_max], name="loss_masking")
	start_token_infer = tf.placeholder(dtypes.int32, shape=[None], name="start_token_infer")

	# Cells
	encoder_cell = single_cell_enc = tf.contrib.rnn.LSTMCell(hidden_units)
	if hidden_layers > 1:
		encoder_cell = tf.contrib.rnn.MultiRNNCell([single_cell_enc] * hidden_layers)
	decoder_cell = single_cell_dec = tf.contrib.rnn.LSTMCell(hidden_units)
	if hidden_layers > 1:
		decoder_cell = tf.contrib.rnn.MultiRNNCell([single_cell_dec] * hidden_layers)

	# Seq2Seq		
	# Encoder
	with variable_scope.variable_scope("Encoder_embedding_layer"):
		embeddings_enc = tf.Variable(tf.random_uniform([enc_input_dimension, embedding_size], -1.0, 1.0), dtype=tf.float32)
		encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings_enc, encoder_inputs)

	_, initial_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs_embedded, sequence_length=encoder_lengths, dtype=tf.float32, scope = "RNN_encoder")

	# Decoder
	with variable_scope.variable_scope("Decoder_embedding_layer"):
		embeddings_dec = tf.Variable(tf.random_uniform([dec_input_dimension, embedding_size], -1.0, 1.0), dtype=tf.float32)
		decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings_dec, decoder_inputs)

	final_layer = layers_core.Dense(units=dec_input_dimension, name="Output_layer")
	with variable_scope.variable_scope("RNN_decoder"):
		helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs_embedded, decoder_lengths)
		decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, initial_state, output_layer=final_layer)
		outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=dec_timesteps_max)
		training_output = outputs.rnn_output
		print training_output.shape

		helper_infer = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings_dec, start_token_infer, end_of_sequence_id)
		decoder_infer = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper_infer, initial_state, output_layer=final_layer)
		outputs_infer, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder_infer, maximum_iterations=dec_timesteps_max)
		infer_output = outputs_infer.sample_id
		print infer_output.shape

	# Training
	with variable_scope.variable_scope("Backpropagation"):
		loss = tf.contrib.seq2seq.sequence_loss(targets=decoder_outputs, logits=training_output, weights=masking)
		updates = tf.train.AdamOptimizer(5e-4).minimize(loss)

	# Store variables for further training or execution
	tf.add_to_collection('variables_to_store', training_output)
	tf.add_to_collection('variables_to_store', updates)
	tf.add_to_collection('variables_to_store', loss)
	tf.add_to_collection('variables_to_store', encoder_inputs)
	tf.add_to_collection('variables_to_store', decoder_inputs)
	tf.add_to_collection('variables_to_store', decoder_outputs)
	tf.add_to_collection('variables_to_store', masking)
	tf.add_to_collection('variables_to_store', encoder_lengths)
	tf.add_to_collection('variables_to_store', decoder_lengths)
	tf.add_to_collection('variables_to_store', infer_output)
	tf.add_to_collection('variables_to_store', start_token_infer)
	tf.add_to_collection('variables_to_store', initial_state)

	return (training_output, updates, loss, encoder_inputs, decoder_inputs, decoder_outputs, masking, encoder_lengths, decoder_lengths, infer_output, start_token_infer)

def softmax(x):
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()

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

# Open files
print "Reading network input data..."
with open (data_path+"/input_data.txt", 'r') as f:
    encoder_input_data = json.load(f)
with open (data_path+"/output_data.txt", 'r') as f:
    decoder_data = json.load(f)
with open (data_path+"/word_to_index_eng.txt", 'r') as f:
    word_to_index_english = json.load(f)
with open (data_path+"/index_to_word_eng.txt", 'r') as f:
    index_to_word_english = json.load(f)
with open (data_path+"/word_to_index_ger.txt", 'r') as f:
    word_to_index_german = json.load(f)
with open (data_path+"/index_to_word_ger.txt", 'r') as f:
    index_to_word_german = json.load(f)


# Variables for the network definition
print "Retrieve input variables from files..."
enc_dimension = len(index_to_word_english)
dec_dimension = len(index_to_word_german)
enc_timesteps_max = len(encoder_input_data[0])
dec_timesteps_max = len(decoder_data[0])-1 # Because the output needs to be shifted by 1
start_of_sequence_id = word_to_index_english["<START>"]
end_of_sequence_id = word_to_index_english["<END>"]
print enc_dimension
print dec_dimension
print enc_timesteps_max
print dec_timesteps_max
print "Start Token = " + str(start_of_sequence_id)
print "End Token = " + str(end_of_sequence_id)


# Variables for the network execution
print "Calculate dynamic lengths..."
encoder_length = []
decoder_length = []
decoder_mask = []
decoder_input_data = []
decoder_output_data = []

for sentence in encoder_input_data:
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
	decoder_mask.append([])
	for word in decoder_output_data[-1]:
		if word != 0:
			decoder_mask[idx].append(1.0)
		else:
			decoder_mask[idx].append(0.0)
	decoder_length.append(len(decoder_output_data[-1]))

# Split data into batches
print "Split data into batches..."
encoder_input_data_batch = createBatch(encoder_input_data, batch_size)
decoder_input_data_batch = createBatch(decoder_input_data, batch_size)
decoder_output_data_batch = createBatch(decoder_output_data, batch_size)
encoder_input_length_batch = createBatch(encoder_length, batch_size)
decoder_input_length_batch = createBatch(decoder_length, batch_size)
decoder_mask_batch = createBatch(decoder_mask, batch_size)

# Create computational graph
print "Create computational graph..."
train_out, updates, loss, enc_in, dec_in, dec_out, mask, enc_len, dec_len, inf_out, start_token_infer = seq2seq(enc_input_dimension=enc_dimension, enc_timesteps_max=enc_timesteps_max, dec_timesteps_max=dec_timesteps_max, hidden_units=hidden_dimensions, hidden_layers=nb_hidden_layers, embedding_size=embedding_size, dec_input_dimension=dec_dimension, start_of_sequence_id=start_of_sequence_id, end_of_sequence_id=end_of_sequence_id)

# Launch the graph
print "Launch the graph..."
session_config = tf.ConfigProto(allow_soft_placement=True)    
session_config.gpu_options.per_process_gpu_memory_fraction = 0.90

with tf.Session(config=session_config) as session:
	session.run(tf.global_variables_initializer())
	saver = tf.train.Saver(max_to_keep=None)
	writer = tf.summary.FileWriter(".", graph=tf.get_default_graph())
	
	print "Start training..."
	if not os.path.exists(save_path+'/models/'):
		os.makedirs(save_path+'/models/')

	if not os.path.exists(save_path+'/models/'+model_name):
		os.makedirs(save_path+'/models/'+model_name)

	with open(save_path+'/models/'+model_name+"/log.txt",'a') as f:
		f.write("{}\n".format(""))
		f.write("{}\n".format("Training started with: " + str(options)))

	for epoch in range(epochs):
		print "epoch " + str(epoch+1) + " / " + str(epochs)
		with open(save_path+'/models/'+model_name+"/log.txt",'a') as f:
			f.write("{}\n".format("epoch " + str(epoch+1) + " / " + str(epochs) + " " + str(datetime.datetime.now())))

		for batch_index,_ in enumerate(encoder_input_data_batch):

			print "----batch " + str(batch_index+1) + " / " + str(len(encoder_input_data_batch))
			with open(save_path+'/models/'+model_name+"/log.txt",'a') as f:
				f.write("{}\n".format("batch " + str(batch_index+1) + " / " + str(len(encoder_input_data_batch)) + " " + str(datetime.datetime.now())))

			feed = {}
			feed["encoder_inputs"] = encoder_input_data_batch[batch_index]
			feed["encoder_length"] = encoder_input_length_batch[batch_index]
			feed["decoder_inputs"] = decoder_input_data_batch[batch_index]
			feed["decoder_length"] = decoder_input_length_batch[batch_index]
			feed["decoder_outputs"] = decoder_output_data_batch[batch_index]
			feed["mask"] = decoder_mask_batch[batch_index]

			training_output = session.run([updates, loss], feed_dict={enc_in:feed["encoder_inputs"], dec_in:feed["decoder_inputs"], dec_out: feed["decoder_outputs"], mask: feed["mask"], enc_len: feed["encoder_length"], dec_len: feed["decoder_length"]})

		print "Saving epoch..."
		saver.save(session, save_path+'/models/'+model_name+"/model", global_step = epoch+1)

	print "Training finished..."
##############################################
'''
	encoder_input_data_batch_infer = createBatch(encoder_input_data, batch_size_inference)
	encoder_input_length_batch_infer = createBatch(encoder_length, batch_size_inference)
	decoder_input_length_batch_infer = createBatch(decoder_length, batch_size_inference)

	feed = {}
	feed["encoder_inputs"] = encoder_input_data_batch_infer[5]
	feed["encoder_length"] = encoder_input_length_batch_infer[5]
	feed["decoder_length"] = decoder_input_length_batch_infer[5]
	feed["start_token_infer"] = [start_of_sequence_id]*batch_size_inference

	test_output = session.run(inf_out, feed_dict={enc_in:feed["encoder_inputs"], enc_len: feed["encoder_length"], dec_len: feed["decoder_length"], start_token_infer: feed["start_token_infer"]})
	print test_output
	test_sentence = ""
	for batch in test_output.tolist():
		for word in batch:
			test_sentence = test_sentence + index_to_word_german[str(word)]
	print test_sentence
	'''
# END
