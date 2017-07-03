
# Commandline Arguments
##############################################
import optparse
parser = optparse.OptionParser()
parser.add_option('--layers', action="store", dest="layers", help="The number of hidden layers in the model (default: 1)", default=1)
parser.add_option('--sentence_model', action="store", dest="sentence_model", help="The path to the sentence embeddings model(defaule: .)", default=".")
parser.add_option('--sentence_vocab', action="store", dest="sentence_vocab", help="The path to the sentence embeddings vocabulary(defaule: .)", default=".")
parser.add_option('--sentence_log', action="store", dest="sentence_log", help="The path to the sentence embeddings logfiles(defaule: .)", default=".")
parser.add_option('--layer_dim', action="store", dest="layer_dim", help="The number of neurons in the hidden layer(s)  (default: 128)", default=128)
parser.add_option('--batch_size', action="store", dest="batch_size", help="The batch size of the model (default: 30)", default=30)
parser.add_option('--epochs', action="store", dest="epochs", help="The number of training epochs (default: 10)", default=10)
parser.add_option('--save_path', action="store", dest="save_path", help="The path to save the folders (logging, models etc.)  (default: .)", default=".")
parser.add_option('--continue_version', action="store", dest="continue_version", help="Continue to learn a already started model (default: 0)", default=0)
parser.add_option('--starting_epoch', action="store", dest="starting_epoch", help="Continue learning with this epoch (default: 1)", default=1)
options, args = parser.parse_args()
nb_hidden_layers = int(options.layers)
hidden_dimensions= int(options.layer_dim)
batch_size = int(options.batch_size)
trainingIterations = int(options.epochs)
starting_epoch = int(options.starting_epoch)
continue_version = int(options.continue_version)
continue_learning = 0
if(int(starting_epoch) > 1):
    continue_learning = 1

sentence_model = options.sentence_model
sentence_vocab = options.sentence_vocab
sentence_log = options.sentence_log
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
from scipy import spatial
##############################################



# Main
##############################################

# Open the sentence log to find out about the used parameters
print "Reading sentence log data..."
path  = open(sentence_log, "r")
sentence_logging_text = path.read().decode('utf8')

# Find corpus and save
parameters = json.loads(sentence_logging_text[:sentence_logging_text.find("}")+1].replace("'","\""))
corpus = parameters['dataset']
if corpus == "DEV":
    training_data = "./../tedData/sets/development/talkseperated_original_development_texts.txt"
if corpus == "PRD":
    training_data = "./../tedData/sets/training/talkseperated_original_development_texts.txt"
embedding_size = int(parameters['embedding_dim'])
nb_hidden_layers = int(parameters['layers'])
hidden_dimensions = int(parameters['layer_dim'])

# Open the sentence index_to_word and word_to_index
print "Reading sentence vocab data..."
word_to_index = []
with open(sentence_vocab + "/word_to_index.json") as f:    
	word_to_index = json.load(f)
index_to_word = []
with open(sentence_vocab + "/index_to_word.json") as f:    
	index_to_word = json.load(f)

# Open the model to create the sentence embeddings
sentenceModel = load_model(sentence_model)
newSentenceModel = Sequential()
newSentenceModel.add(Embedding(input_dim=len(word_to_index), output_dim=embedding_size, mask_zero=True, weights=sentenceModel.layers[0].get_weights()))
for layerNumber in range(0, nb_hidden_layers):
    print "add LSTM layer...."
    newSentenceModel.add(LSTM(units=hidden_dimensions, return_sequences=True, weights=sentenceModel.layers[layerNumber+1].get_weights()))
newSentenceModel.compile(loss='sparse_categorical_crossentropy', optimizer="adam")

# Test for word embeddings
'''
test = Sequential()
test.add(Embedding(input_dim=len(word_to_index), output_dim=embedding_size, mask_zero=True, weights=sentenceModel.layers[0].get_weights()))
test.compile(loss='sparse_categorical_crossentropy', optimizer="adam")
input_sentence= ["dog"]
testInput = np.zeros((1, len(input_sentence)), dtype=np.int16)
for index, idx in enumerate(input_sentence):
	testInput[0, index] = word_to_index[idx.lower()]
prediction1 = test.predict(testInput)
print prediction1[0][-1]
input_sentence= ["car"]
testInput = np.zeros((1, len(input_sentence)), dtype=np.int16)
for index, idx in enumerate(input_sentence):
	testInput[0, index] = word_to_index[idx.lower()]
prediction2 = test.predict(testInput)
print prediction2[0][-1]
input_sentence= ["cat"]
testInput = np.zeros((1, len(input_sentence)), dtype=np.int16)
for index, idx in enumerate(input_sentence):
	testInput[0, index] = word_to_index[idx.lower()]
prediction3 = test.predict(testInput)
print prediction3[0][-1]
result1 = spatial.distance.cosine(prediction1[0][-1], prediction2[0][-1])
result2 = spatial.distance.cosine(prediction2[0][-1], prediction3[0][-1])
result3 = spatial.distance.cosine(prediction1[0][-1], prediction3[0][-1])
print result1
print result2
print result3
'''

# Split the text in talks, sentences and words --> tokens[#talks][#sentence][#word]
# words are needed for the vocabulary calculation further down the road
print "Tokenizing file..."
all_words = nltk.word_tokenize(corpus)

# Split the text at the talks
plain_talks = []
walkthrough_text = corpus
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


#TODO
#LM sentence embedding for former sentence in dataset
#seq to seq model for word prediction

