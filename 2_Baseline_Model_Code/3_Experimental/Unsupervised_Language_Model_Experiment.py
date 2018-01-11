
# Commandline Arguments
##############################################
import optparse
parser = optparse.OptionParser()
parser.add_option('--dataset', action="store", dest="dataset", help="Choose the dataset to train on [PRD,DEV,TEST] (default: PRD)", default="PRD")
parser.add_option('--layers', action="store", dest="layers", help="The number of hidden layers in the model (default: 1)", default=1)
parser.add_option('--layer_dim', action="store", dest="layer_dim", help="The number of neurons in the hidden layer(s)  (default: 512)", default=512)
parser.add_option('--embedding_dim', action="store", dest="embedding_dim", help="The number of dimensions the embedding has  (default: 256)", default=256)
parser.add_option('--batch_size', action="store", dest="batch_size", help="The batch size of the model (default: 100)", default=100)
parser.add_option('--epochs', action="store", dest="epochs", help="The number of training epochs (default: 20)", default=20)
parser.add_option('--vocabulary_size', action="store", dest="vocabulary_size", help="Size of the vocabulary (default: 30000)", default=30000)
parser.add_option('--save_path', action="store", dest="save_path", help="The path to save the folders (default: .)", default=".")
parser.add_option('--save_name', action="store", dest="save_name", help="The save name  (default: model)", default="model")
parser.add_option('--lr', action="store", dest="lr", help="Learning rate (default: 1e-3)", default=1e-3)
parser.add_option('--timesteps', action="store", dest="timesteps", help="Timesteps (default: 50)", default=50)

options, args = parser.parse_args()
nb_hidden_layers = int(options.layers)
hidden_dimensions= int(options.layer_dim)
embedding_size = int(options.embedding_dim)
batch_size = int(options.batch_size)
trainingIterations = int(options.epochs)
path_to_folders = options.save_path
save_name = options.save_name
vocabulary_size = int(options.vocabulary_size)
unknown_token = "<UNK>"
start_token = "<S>"
end_token = "<E>"
lr = float(options.lr)
longestSentence = int(options.timesteps)
batch_size = 100
dataset = options.dataset
if dataset == "PRD":
    training_data = "./../tedData/sets/training/original_training_texts.txt"
if dataset == "DEV":
    training_data = "./../tedData/sets/development/original_development_texts.txt"
if dataset == "TEST":
    training_data = "./../tedData/sets/test/original_test_texts.txt"
##############################################

# Imports
##############################################
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variable_scope
from collections import Counter
from random import shuffle 
import numpy as np
import random
import sys
import nltk
import re
import os
import datetime
import json
##############################################

# Helper functions
##############################################
def lm(timesteps_max, hidden_units,hidden_layers,embedding_size,vocab_size, learning_rate):

	# Inputs / Outputs / Cells
	network_inputs = tf.placeholder(dtypes.int64, shape=[None, timesteps_max], name="inputs")
	network_lengths = tf.placeholder(dtypes.int32, shape=[None], name="lengths")
	network_outputs = tf.placeholder(dtypes.int64, shape=[None, timesteps_max], name="outputs")
	network_masking = tf.placeholder(dtypes.float32, shape=[None, timesteps_max], name="loss_masking")

	cell = tf.contrib.rnn.LSTMCell(hidden_units)
	if hidden_layers > 1:
		cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(hidden_units) for _ in range(hidden_layers)])

	embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), dtype=tf.float32)
	inputs_embedded = tf.nn.embedding_lookup(embeddings, network_inputs)

	lstm_output,_ = tf.nn.dynamic_rnn(cell, inputs_embedded, sequence_length=network_lengths, dtype=tf.float32, scope = "RNN")
	
	transp = tf.transpose(lstm_output, [1, 0, 2])
	lstm_output_unpacked = tf.unstack(transp)
	outputs = []
	for index, item in enumerate(lstm_output_unpacked):
		if index == 0:
			logits = tf.layers.dense(inputs=item, units=vocab_size, name="output_dense")
		if index > 0:
			logits = tf.layers.dense(inputs=item, units=vocab_size, name="output_dense", reuse=True)
		outputs.append(logits)
	tensor_output = tf.stack(values=outputs, axis=0)
	forward = tf.transpose(tensor_output, [1, 0, 2])

	print network_outputs.shape
	print forward.shape

	loss = tf.contrib.seq2seq.sequence_loss(targets=network_outputs, logits=forward, weights=network_masking)
	updates = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	# Store variables for further training or execution
	tf.add_to_collection('variables_to_store', forward)
	tf.add_to_collection('variables_to_store', updates)
	tf.add_to_collection('variables_to_store', loss)
	tf.add_to_collection('variables_to_store', network_inputs)
	tf.add_to_collection('variables_to_store', network_lengths)
	tf.add_to_collection('variables_to_store', network_outputs)
	tf.add_to_collection('variables_to_store', network_masking)

	return (forward, updates, loss, network_inputs, network_lengths, network_outputs, network_masking)

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

print "Creating vocabulary..."
allVocab = [word[0] for word in Counter(words).most_common()]
vocab = ["<PAD>"]+allVocab[:vocabulary_size]
vocab.append(unknown_token)
vocab.append(start_token)
vocab.append(end_token)

print "Creating mappings..."
index_to_word = dict((index, word) for index, word in enumerate(vocab))
word_to_index = dict((word, index) for index, word in enumerate(vocab))

print "Creating integer array..."
for index1, sentence in enumerate(tokens):
	for index2, word in enumerate(sentence):
		tokens[index1][index2] = word_to_index[word] if word in word_to_index else word_to_index[unknown_token]

print "Adding start and end tokens..."
for index, sentence in enumerate(tokens):
	tokens[index] = [word_to_index[start_token]] + tokens[index] + [word_to_index[end_token]]

for index, sentence in enumerate(tokens):
	if len(sentence) > longestSentence:
		tokens[index] = tokens[index][:longestSentence]

input_Words = []
output_Words = []
lengths = []
masking = []
for sentence in tokens:
	input_Words.append(sentence[:len(sentence)-1])
	output_Words.append(sentence[1:len(sentence)])
	lengths.append(len(sentence)-1)
	masking.append([1.0]*(len(sentence)-1) + [0.0]*(longestSentence-(len(sentence)-1)))

print "Padding sequences with zeros..."
for idx,_ in enumerate(input_Words):
	input_padding = [0]*(longestSentence-len(input_Words[idx]))
	output_padding = [0]*(longestSentence-len(output_Words[idx]))
	input_Words[idx] = input_Words[idx] + input_padding
	output_Words[idx] = output_Words[idx] + output_padding

if not os.path.exists(path_to_folders + "/model_"+str(save_name)):
	os.makedirs(path_to_folders + "/model_"+str(save_name))
with open(path_to_folders + "/model_"+str(save_name)+"/index_to_word.json", "w") as f:
	json.dump(index_to_word, f)
with open(path_to_folders + "/model_"+str(save_name)+"/word_to_index.json", "w") as f:
	json.dump(word_to_index, f)

input_batch = createBatch(input_Words, batch_size)
output_batch = createBatch(output_Words, batch_size)
length_batch = createBatch(lengths, batch_size)
mask_batch = createBatch(masking, batch_size)

print np.asarray(input_batch).shape

print np.asarray(input_batch[0]).shape
print np.asarray(input_batch[1]).shape
print np.asarray(input_batch[2]).shape

print np.asarray(input_batch[0][0]).shape
print np.asarray(input_batch[0][1]).shape
print np.asarray(input_batch[0][2]).shape

print np.asarray(output_batch).shape
print np.asarray(length_batch).shape
print np.asarray(mask_batch).shape


network, updates, loss, inp, length_inp, out, mask = lm(
	timesteps_max=longestSentence, 
	hidden_units=hidden_dimensions, 
	hidden_layers=nb_hidden_layers, 
	embedding_size=embedding_size, 
	vocab_size=vocabulary_size, 
	learning_rate=lr)

session_config = tf.ConfigProto(allow_soft_placement=True)    
session_config.gpu_options.per_process_gpu_memory_fraction = 0.90

with tf.Session(config=session_config) as session:
	session.run(tf.global_variables_initializer())
	saver = tf.train.Saver(max_to_keep=None)
	with open(path_to_folders + "/model_"+str(save_name)+"/log.txt", "w") as f:
		f.write("{}\n".format("Training parameters: " + str(options)))
	data = zip(input_batch, output_batch, length_batch, mask_batch)

	for epoch in range(trainingIterations):
		print "epoch " + str(epoch+1) + " / " + str(trainingIterations)
		with open(path_to_folders + "/model_"+str(save_name)+"/log.txt",'a') as f:
			f.write("{}\n".format("epoch " + str(epoch+1) + " / " + str(trainingIterations) + " " + str(datetime.datetime.now())))

		shuffle(data)

		for batch_index,_ in enumerate(input_batch):
			print "----batch " + str(batch_index+1) + " / " + str(len(input_batch))
			with open(path_to_folders + "/model_"+str(save_name)+"/log.txt",'a') as f:
				f.write("{}\n".format("batch " + str(batch_index+1) + " / " + str(len(input_batch)) + " " + str(datetime.datetime.now())))

			feed = {}
			feed["input"] = data[batch_index][0]
			feed["output"] = data[batch_index][1]
			feed["length"] = data[batch_index][2]
			feed["mask"] = data[batch_index][3]

			session.run([updates, loss], feed_dict={inp:feed["input"], out:feed["output"], length_inp: feed["length"], mask: feed["mask"]})

		saver.save(session, path_to_folders + "/model_"+str(save_name), global_step = epoch+1)

##############################################
# END