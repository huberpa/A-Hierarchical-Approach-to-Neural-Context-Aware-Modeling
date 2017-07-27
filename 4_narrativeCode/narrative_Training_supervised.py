
# Commandline Arguments
##############################################
import optparse
parser = optparse.OptionParser()
parser.add_option('--embedding_size', action="store", dest="embedding_size", help="The number of hidden layers in the model (default: 256)", default=256)
parser.add_option('--layer_number', action="store", dest="layer_number", help="The number of hidden layers in the model (default: 1)", default=1)
parser.add_option('--layer_dimension', action="store", dest="layer_dimension", help="The number of neurons in the hidden layer(s)  (default: 512)", default=512)
parser.add_option('--batch_size', action="store", dest="batch_size", help="The batch size of the model (default: 100)", default=100)
parser.add_option('--epochs', action="store", dest="epochs", help="The number of training epochs (default: 15)", default=15)
parser.add_option('--data_path', action="store", dest="data_path", help="The path to the data folder (default: .)", default=".")
parser.add_option('--name', action="store", dest="name", help="The name of the model (default: model)", default="model")
options, args = parser.parse_args()
embedding_size = int(options.embedding_size)
nb_hidden_layers = int(options.layer_number)
hidden_dimensions= int(options.layer_dimension)
batch_size = int(options.batch_size)
epochs = int(options.epochs)
data_path = options.data_path
model_name = options.name
##############################################



# Imports
##############################################
import tensorflow as tf 
import numpy as np
import json
import copy
import os
import datetime
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variable_scope
##############################################



# Helper functions
##############################################
def seq2seq(batch_size,enc_input_dimension,enc_timesteps_max,dec_timesteps_max,hidden_units,hidden_layers,input_embedding_size,vocab_size):

	# Inputs / Outputs / Cells
	encoder_inputs = tf.placeholder(dtypes.float32, shape=[batch_size, enc_timesteps_max, enc_input_dimension], name="encoder_inputs")
	encoder_lengths = tf.placeholder(dtypes.int32, shape=[batch_size], name="encoder_lengths")
	decoder_inputs = tf.placeholder(dtypes.int64, shape=[batch_size, dec_timesteps_max], name="decoder_inputs")
	decoder_lengths = tf.placeholder(dtypes.int32, shape=[batch_size], name="decoder_lengths")
	decoder_outputs = tf.placeholder(dtypes.int64, shape=[batch_size, dec_timesteps_max], name="decoder_outputs")
	masking = tf.placeholder(dtypes.float32, shape=[batch_size, dec_timesteps_max], name="loss_masking")

	# Cells
	encoder_cell = single_cell_enc = tf.contrib.rnn.LSTMCell(hidden_units)
	if hidden_layers > 1:
		encoder_cell = tf.contrib.rnn.MultiRNNCell([single_cell_enc] * hidden_layers)
	decoder_cell = single_cell_dec = tf.contrib.rnn.LSTMCell(hidden_units)
	if hidden_layers > 1:
		decoder_cell = tf.contrib.rnn.MultiRNNCell([single_cell_dec] * hidden_layers)

	# Seq2Seq		
	# Encoder
	_, initial_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs, sequence_length=encoder_lengths, dtype=tf.float32, scope = "RNN_encoder")

	# Decoder
	outputs = []

	with variable_scope.variable_scope("Embedding_layer"):
		embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
		decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

	lstm_output,state = tf.nn.dynamic_rnn(decoder_cell, decoder_inputs_embedded, sequence_length=decoder_lengths, initial_state=initial_state, scope = "RNN_decoder")
	
	with variable_scope.variable_scope("Output_layer"):
		transp = tf.transpose(lstm_output, [1, 0, 2])
		lstm_output_unpacked = tf.unstack(transp)
		for index, item in enumerate(lstm_output_unpacked):
			if index == 0:
				logits = tf.layers.dense(inputs=item, units=1, name="output_dense")
			if index > 0:
				logits = tf.layers.dense(inputs=item, units=1, name="output_dense", reuse=True)
			outputs.append(logits)
		tensor_output = tf.stack(values=outputs, axis=0)
		forward = tf.transpose(tensor_output, [1, 0, 2])

	# Training
	with variable_scope.variable_scope("Backpropagation"):
		loss = tf.contrib.seq2seq.sequence_loss(targets=decoder_outputs, logits=forward, weights=masking, average_across_timesteps=True )# Change to False?
		updates = tf.train.AdamOptimizer().minimize(loss)

	# Store variables for further training or execution
	tf.add_to_collection('variables_to_store', forward)
	tf.add_to_collection('variables_to_store', updates)
	tf.add_to_collection('variables_to_store', loss)
	tf.add_to_collection('variables_to_store', encoder_inputs)
	tf.add_to_collection('variables_to_store', decoder_inputs)
	tf.add_to_collection('variables_to_store', decoder_outputs)
	tf.add_to_collection('variables_to_store', masking)
	tf.add_to_collection('variables_to_store', encoder_lengths)
	tf.add_to_collection('variables_to_store', decoder_lengths)

	return (forward, updates, loss, encoder_inputs, decoder_inputs, decoder_outputs, masking, encoder_lengths, decoder_lengths)

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
encoder_input_length_batch = createBatch(encoder_input_length, batch_size)
decoder_input_length_batch = createBatch(decoder_input_length, batch_size)
decoder_mask_batch = createBatch(decoder_mask, batch_size)

# Create computational graph
print "Create computational graph..."
network, updates, loss, enc_in, dec_in, dec_out, mask, enc_len, dec_len = seq2seq(batch_size=batch_size, enc_input_dimension=enc_input_dimension, enc_timesteps_max=enc_timesteps_max, dec_timesteps_max=dec_timesteps_max, hidden_units=hidden_dimensions, hidden_layers=nb_hidden_layers, input_embedding_size=embedding_size, vocab_size=vocab_size)

# Launch the graph
print "Launch the graph..."
with tf.Session() as session:
	session.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	writer = tf.summary.FileWriter(".", graph=tf.get_default_graph())
	
	# Training
	print "Start training..."

	if not os.path.exists(data_path+'/models/'):
		os.makedirs(data_path+'/models/')

	if not os.path.exists(data_path+'/models/'+model_name):
		os.makedirs(data_path+'/models/'+model_name)

	with open(data_path+'/models/'+model_name+"/log.txt",'a') as f:
		f.write("{}\n".format(""))
		f.write("{}\n".format("Training started with: " + str(options)))

	for epoch in range(epochs):
		print "epoch " + str(epoch+1) + " / " + str(epochs)
		with open(data_path+'/models/'+model_name+"/log.txt",'a') as f:
			f.write("{}\n".format("epoch " + str(epoch+1) + " / " + str(epochs) + " " + str(datetime.datetime.now())))

		for batch_index,_ in enumerate(encoder_input_data_batch):

			print "----batch " + str(batch_index+1) + " / " + str(len(encoder_input_data_batch))
			with open(data_path+'/models/'+model_name+"/log.txt",'a') as f:
				f.write("{}\n".format("batch " + str(batch_index+1) + " / " + str(len(encoder_input_data_batch)) + " " + str(datetime.datetime.now())))

			feed = {}
			feed["encoder_inputs"] = encoder_input_data_batch[batch_index]
			feed["encoder_length"] = encoder_input_length_batch[batch_index]
			feed["decoder_inputs"] = decoder_input_data_batch[batch_index]
			feed["decoder_length"] = decoder_input_length_batch[batch_index]
			feed["decoder_outputs"] = decoder_output_data_batch[batch_index]
			feed["mask"] = decoder_mask_batch[batch_index]

			'''
			shape1 = copy.deepcopy(feed["encoder_inputs"])
			shape2 = copy.deepcopy(feed["decoder_inputs"])
			shape3 = copy.deepcopy(feed["decoder_outputs"])
			shape4 = copy.deepcopy(feed["mask"])


			if len(np.asarray(shape1).shape)<3:
				print ("*"*100)
				print "encoder_inputs"
				print np.asarray(shape1).shape

			if len(np.asarray(shape2).shape)<2:
				print ("*"*100)
				print "decoder_inputs"
				print np.asarray(shape2).shape

			if len(np.asarray(shape3).shape)<2:
				print ("*"*100)
				print "decoder_outputs"
				print np.asarray(shape3).shape
				print feed["decoder_outputs"]

			if len(np.asarray(shape4).shape)<2:
				print ("*"*100)
				print "mask"
				print np.asarray(shape4).shape
				'''

			training_output = session.run([updates, loss], feed_dict={enc_in:feed["encoder_inputs"], dec_in:feed["decoder_inputs"], dec_out: feed["decoder_outputs"], mask: feed["mask"], enc_len: feed["encoder_length"], dec_len: feed["decoder_length"]})

		print "Saving epoch..."
		saver.save(session, data_path+'/models/'+model_name+"/"+model_name, global_step = epoch+1)

print "Training finished..."
##############################################

# END
