
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
import seq2seq
from seq2seq.models import Seq2Seq
##############################################



# Main
##############################################

# Open the sentence log to find out about the used parameters
print "Reading sentence log data..."
path  = open(sentence_log, "r")
sentence_logging_text = path.read().decode('utf8')
parameters = json.loads(sentence_logging_text[:sentence_logging_text.find("}")+1].replace("'","\""))
corpus = parameters['dataset']
if corpus == "DEV":
    training_data = "./../tedData/sets/development/talkseperated_original_development_texts.txt"
if corpus == "PRD":
    training_data = "./../tedData/sets/training/talkseperated_original_training_texts.txt"
embedding_size = int(parameters['embedding_dim'])
nb_hidden_layers = int(parameters['layers'])
hidden_dimensions = int(parameters['layer_dim'])
start_token = parameters['start_token']
end_token = parameters['end_token']
unknown_token = parameters['unknown_token']

# Open the training dataset
print "Reading training data..."
path  = open(training_data, "r")
train = path.read().decode('utf8')

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

# Split the text in talks, sentences and words --> tokens[#talks][#sentence][#word]
print "Tokenizing file..."
plain_talks = []
walkthrough_text = train
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

# Transform words into numbers and get talk maxLength
talks_numerified = []
talk_sentence_embedding = []
talk_maxLength = 0 # 518 for PRD
for index0, talk in enumerate(talks[:1]):
	if len(talk) > talk_maxLength:
		talk_maxLength = len(talk)
	talks_numerified.append([])
	for index1, sentence in enumerate(talk[:3]):
		talks_numerified[index0].append([])
		for index2, word in enumerate(sentence):
			talks_numerified[index0][index1].append(word_to_index[word] if word in word_to_index else word_to_index[unknown_token])
for index0, talk in enumerate(talks[:1]):
	for index1, sentence in enumerate(talk[:3]):
		talks_numerified[index0][index1] = [word_to_index[start_token]] + talks_numerified[index0][index1] + [word_to_index[end_token]]


for index0, talk in enumerate(talks_numerified):
	talk_sentence_embedding.append([])
	for index1, sentence in enumerate(talk):
		trainInput = np.zeros((1, len(sentence)), dtype=np.int16)
		for index, idx in enumerate(sentence):
			trainInput[0, index] = idx
		# Has shape (#nb_talks, #talk_length(nb_sentences), #hidden_state_neurons)
		talk_sentence_embedding[index0].append(newSentenceModel.predict(trainInput, verbose=0)[0][-1])

trainInput = np.zeros((len(talk_sentence_embedding), len(talk_sentence_embedding[0]), len(talk_sentence_embedding[0][0])), dtype=np.int16)
for index1, talk in enumerate(talk_sentence_embedding):
	for index2, sentence in enumerate(talk):
		for index3, value in enumerate(sentence):
			trainInput[index1, index2, index3] = value

trainOutput = np.zeros((len(talks_numerified), len(talks_numerified[0]), len(talks_numerified[0][0])), dtype=np.int16)
for index1, talk in enumerate(talks_numerified):
	for index2, sentence in enumerate(talk):
		for index3, value in enumerate(sentence):
			trainOutput[index1, index2, index3] = value

print trainInput[1][2]
print trainOutput[1][2]


model = Seq2Seq(batch_input_shape=(len(talk_sentence_embedding), len(talk_sentence_embedding[0]), len(talk_sentence_embedding[0][0])), hidden_dim=256, output_length=50, output_dim=30000, depth=2)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
print model.summary() 

#model.fit(trainInput, trainingOutput, batch_size=int(batch_size), epochs=1)


#print len(talk_sentence_embedding) # all
#print len(talk_sentence_embedding[0]) # The talk
#print len(talk_sentence_embedding[0][0]) # The last hidden state


#TODO
#LM sentence embedding for former sentence in dataset
#seq to seq model for word prediction

