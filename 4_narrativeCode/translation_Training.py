
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
#import tensorflow as tf 
import numpy as np
from xml.dom.minidom import parse, parseString
import json
import os
import datetime
import sys
import nltk
reload(sys)  
sys.setdefaultencoding('utf-8')
#from tensorflow.python.framework import dtypes
#from tensorflow.python.ops import variable_scope
##############################################
'''
# Helper functions
##############################################
def seq2seq(enc_input_dimension,enc_timesteps_max,dec_input_dimension, dec_timesteps_max,hidden_units,hidden_layers,embedding_size, start_of_sequence_id, end_of_sequence_id):

	# Inputs / Outputs / Cells
	encoder_inputs = tf.placeholder(dtypes.float32, shape=[None, enc_timesteps_max], name="encoder_inputs")
	encoder_lengths = tf.placeholder(dtypes.int32, shape=[None], name="encoder_lengths")
	decoder_inputs = tf.placeholder(dtypes.int64, shape=[None, dec_timesteps_max], name="decoder_inputs")
	decoder_lengths = tf.placeholder(dtypes.int32, shape=[None], name="decoder_lengths")
	decoder_outputs = tf.placeholder(dtypes.int64, shape=[None, dec_timesteps_max], name="decoder_outputs")
	masking = tf.placeholder(dtypes.float32, shape=[None, dec_timesteps_max], name="loss_masking")

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

	# For training
	decoder_train = tf.contrib.seq2seq.simple_decoder_fn_train(initial_state)
	decoder_output_train, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(cell=decoder_cell, decoder_fn=decoder_train, inputs=decoder_inputs_embedded, sequence_length=decoder_lengths, scope = "RNN_decoder")
	
	# For generating
	decoder_infer = tf.contrib.seq2seq.simple_decoder_fn_inference(encoder_state=initial_state, embeddings=embeddings_dec, start_of_sequence_id=start_of_sequence_id, end_of_sequence_id=end_of_sequence_id, maximum_length= dec_timesteps_max, num_decoder_symbols=dec_input_dimension)
    inference_output, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(cell=decoder_cell, decoder_fn=decoder_infer, scope = "RNN_decoder")

	with variable_scope.variable_scope("Output_layer"):
		outputs = []
		transp = tf.transpose(decoder_output_train, [1, 0, 2])
		decoder_output_unpacked = tf.unstack(transp)
		for index, item in enumerate(decoder_output_unpacked):
			if index == 0:
				logits = tf.layers.dense(inputs=item, units=dec_input_dimension, name="output_dense")
			if index > 0:
				logits = tf.layers.dense(inputs=item, units=dec_input_dimension, name="output_dense", reuse=True)
			outputs.append(logits)

		tensor_output = tf.stack(values=outputs, axis=0)
		forward_train = tf.transpose(tensor_output, [1, 0, 2])

	# Training
	with variable_scope.variable_scope("Backpropagation"):
		loss = tf.contrib.seq2seq.sequence_loss(targets=decoder_outputs, logits=forward_train, weights=masking)
		updates = tf.train.AdamOptimizer(1e-4).minimize(loss)

	# Store variables for further training or execution
	tf.add_to_collection('variables_to_store', forward_train)
	tf.add_to_collection('variables_to_store', updates)
	tf.add_to_collection('variables_to_store', loss)
	tf.add_to_collection('variables_to_store', encoder_inputs)
	tf.add_to_collection('variables_to_store', decoder_inputs)
	tf.add_to_collection('variables_to_store', decoder_outputs)
	tf.add_to_collection('variables_to_store', masking)
	tf.add_to_collection('variables_to_store', encoder_lengths)
	tf.add_to_collection('variables_to_store', decoder_lengths)
	tf.add_to_collection('variables_to_store', inference_output)

	return (forward_train, updates, loss, encoder_inputs, decoder_inputs, decoder_outputs, masking, encoder_lengths, decoder_lengths, inference_output)

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
'''


# Main
##############################################

# Load files
english_talks = []
english_words = []
english_path  = open(data_path+"/train.tags.en-de.en", "r")
english_text = english_path.read()
englishFile = parseString(english_text)
english_documents=englishFile.getElementsByTagName('doc')
for document in english_documents:
	content=document.getElementsByTagName('content')
	for talk in content:
		node_value = talk.childNodes[0].nodeValue.replace(';', '.')
		splitted_talk = node_value.split('\n')
		tokens = []
		for line in splitted_talk:
			if len(line) > 1:
				tokens.append([])
				for index, sentence in enumerate(nltk.sent_tokenize(line)): 
					tokens[-1].append(nltk.word_tokenize(sentence))
		english_talks = english_talks + tokens
		#english_words = english_words + nltk.word_tokenize(node_value)

german_talks = []
german_words = []
german_path  = open(data_path+"/train.tags.en-de.de", "r")
german_text = german_path.read()
germanFile = parseString(german_text)
german_documents=germanFile.getElementsByTagName('doc')
for document in german_documents:
	content=document.getElementsByTagName('content')
	for talk in content:
		node_value = talk.childNodes[0].nodeValue.replace(';', '.')
		splitted_talk = node_value.split('\n')
		tokens = []
		for line in splitted_talk:
			if len(line) > 1:
				tokens.append([])
				for index, sentence in enumerate(nltk.sent_tokenize(line)): 
					tokens[-1].append(nltk.word_tokenize(sentence))
		german_talks = german_talks + tokens		
		#german_words = german_words + nltk.word_tokenize(node_value)

#english_words = [word.lower() for word in english_words]
#german_words = [word.lower() for word in german_words]
#english_talks = [[word.lower() for word in sentence] for sentence in english_talks]
#german_talks = [[word.lower() for word in sentence] for sentence in german_talks]

for index, talk in enumerate(english_talks):
	print len(talk)
	print len(german_talks[index])
	print ('*'*50)
	print ""

#PROBLEM::: Satze nicht sync!!!!!

'''
print "Creating vocabulary..."
allVocab = [word[0] for word in Counter(words).most_common()]
vocab = ["<PAD>"]+allVocab[:vocabulary_size]
vocab.append(unknown_token)
vocab.append(start_token)
vocab.append(end_token)


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
network_train, updates, loss, enc_in, dec_in, dec_out, mask, enc_len, dec_len, inf_out = seq2seq(enc_input_dimension=enc_input_dimension, enc_timesteps_max=enc_timesteps_max, dec_timesteps_max=dec_timesteps_max, hidden_units=hidden_dimensions, hidden_layers=nb_hidden_layers, embedding_size=embedding_size, dec_input_dimension=dec_input_dimension, start_of_sequence_id=, end_of_sequence_id=)

# Launch the graph
print "Launch the graph..."
session_config = tf.ConfigProto(allow_soft_placement=True)    
session_config.gpu_options.per_process_gpu_memory_fraction = 0.90

with tf.Session(config=session_config) as session:
	session.run(tf.global_variables_initializer())
	saver = tf.train.Saver(max_to_keep=None)
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

			#print np.asarray(feed["encoder_inputs"]).shape
			#print np.asarray(feed["decoder_inputs"]).shape
			#print np.asarray(feed["decoder_outputs"]).shape

			if len(np.asarray(feed["decoder_inputs"]).shape) < 2:
				for p in feed["decoder_inputs"]:
					print p
					print len(p)

			print ('*'*50)


			training_output = session.run([updates, loss], feed_dict={enc_in:feed["encoder_inputs"], dec_in:feed["decoder_inputs"], dec_out: feed["decoder_outputs"], mask: feed["mask"], enc_len: feed["encoder_length"], dec_len: feed["decoder_length"]})

		print "Saving epoch..."
		saver.save(session, data_path+'/models/'+model_name+"/model", global_step = epoch+1)

print "Training finished..."
##############################################
'''
# END
