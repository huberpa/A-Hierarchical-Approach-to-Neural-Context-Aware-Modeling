
# Commandline Arguments
##############################################
import optparse
parser = optparse.OptionParser()
parser.add_option('--dataset', action="store", dest="dataset", help="Choose the dataset to train on [PRD,DEV,SMALL] (default: PRD)", default="PRD")
parser.add_option('--vocabulary_size', action="store", dest="vocabulary_size", help="Size of the vocabulary (default: 30000)", default=30000)
parser.add_option('--save_path', action="store", dest="save_path", help="The path to save the folders (logging, models etc.)  (default: .)", default=".")
parser.add_option('--max_context_sentences', action="store", dest="max_context_sentences", help="The number of context sentences to take into account (default: 10)", default=10)
parser.add_option('--max_encoder_words', action="store", dest="max_encoder_words", help="The number of words in an encoder sentence (default: 50)", default=50)
parser.add_option('--max_decoder_words', action="store", dest="max_decoder_words", help="The number of words in an decoder sentence (default: 50)", default=50)
parser.add_option('--embedding_size', action="store", dest="embedding_size", help="The number of hidden layers in the model (default: 256)", default=256)
parser.add_option('--max_sentence', action="store", dest="max_sentence", help="The maximal sentence length (default: 50)", default=50)
parser.add_option('--layer_number', action="store", dest="layer_number", help="The number of hidden layers in the model (default: 1)", default=1)
parser.add_option('--layer_dimension', action="store", dest="layer_dimension", help="The number of neurons in the hidden layer(s)  (default: 512)", default=512)
parser.add_option('--batch_size', action="store", dest="batch_size", help="The batch size of the model (default: 100)", default=100)
parser.add_option('--epochs', action="store", dest="epochs", help="The number of training epochs (default: 25)", default=25)
parser.add_option('--name', action="store", dest="name", help="The name of the model (default: model)", default="model")
parser.add_option('--lr', action="store", dest="lr", help="The model's learning rate (default: 1e-3)", default="1e-3")

options, args = parser.parse_args()
max_context_sentences = options.max_context_sentences
max_encoder_words = options.max_encoder_words
max_decoder_words = options.max_decoder_words
embedding_size = int(options.embedding_size)
nb_hidden_layers = int(options.layer_number)
hidden_dimensions= int(options.layer_dimension)
batch_size = int(options.batch_size)
epochs = int(options.epochs)
learning_rate = float(options.lr)
model_name = options.name
dataset = options.dataset
path_to_folders = options.save_path
vocabulary_size = int(options.vocabulary_size)
max_sentence = options.max_sentence
if dataset == "PRD":
    training_data = "./../tedData/sets/training/original_training_texts.txt"
if dataset == "DEV":
    training_data = "./../tedData/sets/development/original_development_texts.txt"
if dataset == "TEST":
    training_data = "./../tedData/sets/test/original_test_texts.txt"

unknown_token = "<UNK>"
start_token = "<S>"
end_token = "<E>"
pad_token = "<P>"
##############################################





# Imports
##############################################
import tensorflow as tf 
from random import shuffle 
import numpy as np
import json
import os
import datetime
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variable_scope
##############################################



# Helper functions
##############################################
def seq2seq(enc_sentence_max,enc_words_max,dec_words_max,hidden_units,hidden_layers,input_embedding_size,vocab_size):

	# Inputs / Outputs / Cells
	encoder_inputs = tf.placeholder(dtypes.int64, shape=[None, enc_sentence_max, enc_words_max], name="encoder_inputs")
	encoder_word_lengths = tf.placeholder(dtypes.int32, shape=[None, enc_sentence_max], name="encoder_word_lengths")
	encoder_sentence_lengths = tf.placeholder(dtypes.int32, shape=[None], name="encoder_sentence_lengths")
	decoder_inputs = tf.placeholder(dtypes.int64, shape=[None, dec_words_max], name="decoder_inputs")
	decoder_lengths = tf.placeholder(dtypes.int32, shape=[None], name="decoder_lengths")
	decoder_outputs = tf.placeholder(dtypes.int64, shape=[None, dec_words_max], name="decoder_outputs")
	masking = tf.placeholder(dtypes.float32, shape=[None, dec_words_max], name="loss_masking")

	# Cells
	encoder_cell = tf.contrib.rnn.LSTMCell(hidden_units, reuse=tf.get_variable_scope().reuse)
	if hidden_layers > 1:
		encoder_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(hidden_units) for _ in range(hidden_layers)])
	decoder_cell = tf.contrib.rnn.LSTMCell(hidden_units)
	if hidden_layers > 1:
		decoder_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(hidden_units) for _ in range(hidden_layers)])

	# Seq2Seq		
	# Encoder
	with variable_scope.variable_scope("Embedding_layer"):
		embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
	
	sentence_states = []
	transp = tf.transpose(encoder_inputs, [1, 0, 2])
	input_sentences = tf.unstack(transp)
	for index, sentence in enumerate(input_sentences):
		encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, sentence)
	#	if index == 0:
		_, final_sentence_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs_embedded, sequence_length=encoder_word_lengths[index], dtype=tf.float32)
		#if index > 0:
		#	_, final_sentence_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs_embedded, sequence_length=encoder_word_lengths[index], dtype=tf.float32)
		
		sentence_states.append(final_sentence_state.c)

		sentence_output = tf.stack(values=sentence_states)
		sentence_output = tf.transpose(sentence_output, [1, 0, 2])

		_, final_context_state = tf.nn.dynamic_rnn(encoder_cell2, sentence_output, sequence_length=encoder_sentence_lengths, dtype=tf.float32)

	# Decoder
	decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)
	
	lstm_output,state = tf.nn.dynamic_rnn(decoder_cell, decoder_inputs_embedded, sequence_length=decoder_lengths, initial_state=final_context_state, scope = "RNN_decoder")
	
	with variable_scope.variable_scope("Output_layer"):
		transp = tf.transpose(lstm_output, [1, 0, 2])
		lstm_output_unpacked = tf.unstack(transp)
		for index, item in enumerate(lstm_output_unpacked):
			if index == 0:
				logits = tf.layers.dense(inputs=item, units=vocab_size, name="output_dense")
			if index > 0:
				logits = tf.layers.dense(inputs=item, units=vocab_size, name="output_dense", reuse=True)
			outputs.append(logits)
		tensor_output = tf.stack(values=outputs, axis=0)
		forward = tf.transpose(tensor_output, [1, 0, 2])

	# Training
	with variable_scope.variable_scope("Backpropagation"):
		loss = tf.contrib.seq2seq.sequence_loss(targets=decoder_outputs, logits=forward, weights=masking)
		updates = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	# Store variables for further training or execution
	tf.add_to_collection('variables_to_store', forward)
	tf.add_to_collection('variables_to_store', updates)
	tf.add_to_collection('variables_to_store', loss)
	tf.add_to_collection('variables_to_store', encoder_inputs)
	tf.add_to_collection('variables_to_store', decoder_inputs)
	tf.add_to_collection('variables_to_store', decoder_outputs)
	tf.add_to_collection('variables_to_store', masking)
	tf.add_to_collection('variables_to_store', encoder_word_lengths)
	tf.add_to_collection('variables_to_store', encoder_sentence_lengths)
	tf.add_to_collection('variables_to_store', decoder_lengths)

	return (forward, updates, loss, encoder_inputs, decoder_inputs, decoder_outputs, masking, encoder_word_lengths, encoder_sentence_lengths, decoder_lengths)

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
'''
#TODO!
# Open the training dataset
print "Reading training data..."
path  = open(training_data, "r")
text = path.read().decode('utf8')

# Split the text in sentences and words --> tokens[#sentence][#word]
print "Tokenizing file..."
words = nltk.word_tokenize(text)
tokens = []
for index, sentence in enumerate(nltk.sent_tokenize(text)): 
    tokens.append(nltk.word_tokenize(sentence))
words = [word.lower() for word in words]
tokens = [[word.lower() for word in sentence] for sentence in tokens]

# All words that occur in the text and cut the first vocabulary_size
print "Creating vocabulary..."
allVocab = [word[0] for word in Counter(words).most_common()]
vocab = [pad_token]+allVocab[:vocabulary_size]
vocab.append(unknown_token)
vocab.append(start_token)
vocab.append(end_token)

# Mapping from word to Index and vice versa
print "Creating mappings..."
index_to_word = dict((index, word) for index, word in enumerate(vocab))
word_to_index = dict((word, index) for index, word in enumerate(vocab))

# Adding the unknown_token and transforming the text into indexes
print "Creating integer array..."
for index1, sentence in enumerate(tokens):
    for index2, word in enumerate(sentence):
        tokens[index1][index2] = word_to_index[word] if word in word_to_index else word_to_index[unknown_token]

# Adding the start- and end-tokens to the sentences
print "Adding start and end tokens..."
for index, sentence in enumerate(tokens):
    tokens[index] = [word_to_index[start_token]] + tokens[index] + [word_to_index[end_token]]

# Cut all sentences that are longer than 50 words
for index, sentence in enumerate(tokens):
    if len(sentence) > max_sentence:
        tokens[index] = tokens[index][:max_sentence]




### SO FAR!!!!! ###




# Split the sentences into input and output by shifting them by one
input_Words = []
output_Words = []
for sentence in tokens:
    input_Words.append(sentence[:len(sentence)-1])
    output_Words.append(sentence[1:len(sentence)])

# Pad the sequences with zeros
print "Padding sequences with zeros..."
#TODO


# Split data into batches
print "Split data into batches..."
encoder_input_data_batch = createBatch(encoder_input_data, batch_size)
decoder_input_data_batch = createBatch(decoder_input_data, batch_size)
decoder_output_data_batch = createBatch(decoder_output_data, batch_size)
encoder_input_length_batch = createBatch(encoder_input_length, batch_size)
decoder_input_length_batch = createBatch(decoder_input_length, batch_size)
decoder_mask_batch = createBatch(decoder_mask, batch_size)
'''

# Create computational graph
print "Create computational graph..."
network, updates, loss, enc_in, dec_in, dec_out, mask, enc_word_len, enc_sent_len, dec_len = seq2seq(enc_sentence_max=max_context_sentences, enc_words_max=max_encoder_words, dec_words_max=max_decoder_words, hidden_units=hidden_dimensions, hidden_layers=nb_hidden_layers, input_embedding_size=embedding_size, vocab_size=vocabulary_size)

'''
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

	# Put all inputs in one datastructure to be able to shuffle
	data = zip(encoder_input_data_batch, encoder_input_length_batch, decoder_input_data_batch, decoder_input_length_batch, decoder_output_data_batch, decoder_mask_batch)

	for epoch in range(epochs):
		print "epoch " + str(epoch+1) + " / " + str(epochs)
		with open(data_path+'/models/'+model_name+"/log.txt",'a') as f:
			f.write("{}\n".format("epoch " + str(epoch+1) + " / " + str(epochs) + " " + str(datetime.datetime.now())))

		shuffle(data)

		for batch_index,_ in enumerate(encoder_input_data_batch):

			print "----batch " + str(batch_index+1) + " / " + str(len(encoder_input_data_batch))
			with open(data_path+'/models/'+model_name+"/log.txt",'a') as f:
				f.write("{}\n".format("batch " + str(batch_index+1) + " / " + str(len(encoder_input_data_batch)) + " " + str(datetime.datetime.now())))

			feed = {}
			feed["encoder_inputs"] = data[batch_index][0]
			feed["encoder_length"] = data[batch_index][1]
			feed["decoder_inputs"] = data[batch_index][2]
			feed["decoder_length"] = data[batch_index][3]
			feed["decoder_outputs"] = data[batch_index][4]
			feed["mask"] = data[batch_index][5]

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
