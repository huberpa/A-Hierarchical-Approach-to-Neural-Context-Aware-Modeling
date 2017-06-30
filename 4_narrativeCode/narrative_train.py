
# Commandline Arguments
##############################################
import optparse
parser = optparse.OptionParser()
parser.add_option('--dataset', action="store", dest="dataset", help="Choose the dataset to train on [PRD,DEV] (default: PRD)", default="PRD")
parser.add_option('--layers', action="store", dest="layers", help="The number of hidden layers in the model (default: 1)", default=1)
parser.add_option('--layer_dim', action="store", dest="layer_dim", help="The number of neurons in the hidden layer(s)  (default: 128)", default=128)
parser.add_option('--embedding_dim', action="store", dest="embedding_dim", help="The number of dimensions the embedding has  (default: 300)", default=300)
parser.add_option('--batch_size', action="store", dest="batch_size", help="The batch size of the model (default: 30)", default=30)
parser.add_option('--epochs', action="store", dest="epochs", help="The number of training epochs (default: 10)", default=10)
parser.add_option('--vocabulary_size', action="store", dest="vocabulary_size", help="Size of the vocabulary (default: 12000)", default=12000)
parser.add_option('--unknown_token', action="store", dest="unknown_token", help="Token for words that are not in the vacabulary (default: <UNKWN>)", default="<UNKWN>")
parser.add_option('--start_token', action="store", dest="start_token", help="Token for start (default: <START>)", default="<START>")
parser.add_option('--end_token', action="store", dest="end_token", help="Token for end (default: <END>)", default="<END>")
parser.add_option('--save_path', action="store", dest="save_path", help="The path to save the folders (logging, models etc.)  (default: .)", default=".")
parser.add_option('--continue_version', action="store", dest="continue_version", help="Continue to learn a already started model (default: 0)", default=0)
parser.add_option('--starting_epoch', action="store", dest="starting_epoch", help="Continue learning with this epoch (default: 1)", default=1)
options, args = parser.parse_args()
nb_hidden_layers = int(options.layers)
hidden_dimensions= int(options.layer_dim)
embedding_size = int(options.embedding_dim)
batch_size = int(options.batch_size)
trainingIterations = int(options.epochs)
starting_epoch = int(options.starting_epoch)
continue_version = int(options.continue_version)
continue_learning = 0
if(int(starting_epoch) > 1):
    continue_learning = 1
path_to_folders = options.save_path
vocabulary_size = int(options.vocabulary_size)
unknown_token = options.unknown_token
start_token = options.start_token
end_token = options.end_token
dataset = options.dataset
if dataset == "DEV":
    training_data = "./../tedData/sets/development/talkseperated_original_development_texts.txt"
if dataset == "PRD":
    training_data = "./../tedData/sets/training/talkseperated_original_training_texts.txt"
##############################################



# Imports
##############################################
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, LSTM
from keras.preprocessing import sequence as kerasSequence
from keras.layers.wrappers import TimeDistributed
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

# Split the text in talks, sentences and words --> tokens[#talks][#sentence][#word]
# words are needed for the vocabulary calculation further down the road
print "Tokenizing file..."
all_words = nltk.word_tokenize(text)

# Split the text at the talks
plain_talks = []
walkthrough_text = text
start_current_talk = 0
end_current_talk = 0
while walkthrough_text.find("</TALK>") > -1:
	start_current_talk = walkthrough_text.find("<TALK>")
	end_current_talk = walkthrough_text.find("</TALK>")
	plain_talks.append(walkthrough_text[start_current_talk+6:end_current_talk])
	walkthrough_text = walkthrough_text[end_current_talk+7:]

# Create 3D array with [#talks][#sentence][#word]
talks = [0]*len(plain_talks)
for index, talk in enumerate(plain_talks):
	sentences = nltk.sent_tokenize(talk)
	talks[index] =[0]*len(sentences)
	for idx, sentence in enumerate(sentences): 
	    talks[index][idx] = nltk.word_tokenize(sentence)



