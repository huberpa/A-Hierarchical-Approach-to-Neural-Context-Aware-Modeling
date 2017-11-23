
# Commandline Arguments
##############################################
import optparse
parser = optparse.OptionParser()
parser.add_option('--dataset', action="store", dest="dataset", help="Choose the dataset to train on [PRD,DEV,SMALL] (default: PRD)", default="PRD")
parser.add_option('--layers', action="store", dest="layers", help="The number of hidden layers in the model (default: 1)", default=1)
parser.add_option('--layer_dim', action="store", dest="layer_dim", help="The number of neurons in the hidden layer(s)  (default: 512)", default=512)
parser.add_option('--embedding_dim', action="store", dest="embedding_dim", help="The number of dimensions the embedding has  (default: 256)", default=256)
parser.add_option('--batch_size', action="store", dest="batch_size", help="The batch size of the model (default: 100)", default=100)
parser.add_option('--epochs', action="store", dest="epochs", help="The number of training epochs (default: 20)", default=20)
parser.add_option('--vocabulary_size', action="store", dest="vocabulary_size", help="Size of the vocabulary (default: 30000)", default=30000)
parser.add_option('--save_path', action="store", dest="save_path", help="The path to save the folders (logging, models etc.)  (default: .)", default=".")

options, args = parser.parse_args()
nb_hidden_layers = int(options.layers)
hidden_dimensions= int(options.layer_dim)
embedding_size = int(options.embedding_dim)
batch_size = int(options.batch_size)
trainingIterations = int(options.epochs)
path_to_folders = options.save_path
vocabulary_size = int(options.vocabulary_size)
unknown_token = "<UNK>"
start_token = "<S>"
end_token = "<E>"
longestSentence = 50
dataset = options.dataset
if dataset == "PRD":
    training_data = "./../tedData/sets/training/original_training_texts.txt"
if dataset == "DEV":
    training_data = "./../tedData/sets/development/original_development_texts.txt"
if dataset == "SMALL":
    training_data = "./../tedData/Manual_change/original_text.txt"
##############################################

# Imports
##############################################
import tensorflow as tf
from collections import Counter
import numpy as np
import random
import sys
import nltk
import re
import os
import datetime
import json
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
for sentence in tokens:
	input_Words.append(sentence[:len(sentence)-1])
	output_Words.append(sentence[1:len(sentence)])

print "Padding sequences with zeros..."
input_padding = [0]*(longestSentence-input_Words)
output_padding = [0]*(longestSentence-output_Words)
pad_input_Words = input_Words + input_padding
pad_output_Words = output_Words + output_padding

idx = 1
loop = 1
while loop:
	if not os.path.exists(path_to_folders + "/files_"+str(idx)):
		os.makedirs(path_to_folders + "/files_"+str(idx))
		with open(path_to_folders + "/files_"+str(idx)+"/index_to_word.json", "w") as f:
			json.dump(index_to_word, f)
		with open(path_to_folders + "/files_"+str(idx)+"/word_to_index.json", "w") as f:
			json.dump(word_to_index, f)
		logFile.close()
	if not os.path.exists(path_to_folders + "/models_"+str(idx)):
		os.makedirs(path_to_folders + "/models_"+str(idx))
		loop = 0
	if loop == 1:
		idx += 1

# Build the model


# Iterate trough the set
for iteration in range(starting_epoch, trainingIterations):

    print ""
    print('#' * 50)
    print ""
    print'Iteration: ' + str(iteration)

    per batch:
    
    	iterationResult = model.fit(trainingInput, trainingOutput, batch_size=int(batch_size), epochs=1)

    

    # Save the model at the end of the iteration
    model.save(path_to_folders + "/models_" + str(idx) + "/LM_model_Iteration_" + str(iteration) + '.h5')

