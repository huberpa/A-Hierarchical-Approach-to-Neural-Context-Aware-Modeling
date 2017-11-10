# Commandline Arguments
##############################################
import optparse
parser = optparse.OptionParser()
parser.add_option('--sentence_model', action="store", dest="sentence_model", help="The path to the sentence embeddings model (default: .)", default=".")
parser.add_option('--sentence_vocab', action="store", dest="sentence_vocab", help="The path to the sentence embeddings vocabulary (default: .)", default=".")
parser.add_option('--sentence_log', action="store", dest="sentence_log", help="The path to the sentence embeddings logfiles (default: .)", default=".")
parser.add_option('--save_path', action="store", dest="save_path", help="The path to save the files (default: .)", default=".")

options, args = parser.parse_args()
sentence_model = options.sentence_model
sentence_vocab = options.sentence_vocab
sentence_log = options.sentence_log
save_path = options.save_path
training_data = "./../tedData/sets/training/original_training_texts.txt"
##############################################

# Imports
##############################################
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, LSTM
import numpy as np
import nltk
import os
import json
import copy
import pickle
#import pylab as Plot
##############################################

# Main
##############################################

# Open the sentence log to find out about the used parameters
print "Reading sentence log data..."
path  = open(sentence_log, "r")
sentence_logging_text = path.read().decode('utf8')
parameters = json.loads(sentence_logging_text[:sentence_logging_text.find("}")+1].replace("'","\""))
embedding_size = int(parameters['embedding_dim'])
nb_hidden_layers = int(parameters['layers'])
hidden_dimensions = int(parameters['layer_dim'])
start_token = parameters['start_token']
end_token = parameters['end_token']
unknown_token = parameters['unknown_token']

# Open the sentence index_to_word and word_to_index
print "Reading sentence vocab data..."
word_to_index = []
with open(sentence_vocab + "/word_to_index.json") as f:    
	word_to_index = json.load(f)
index_to_word = []
with open(sentence_vocab + "/index_to_word.json") as f:    
	index_to_word = json.load(f)

# Open the model to create the sentence embeddings
m = load_model(sentence_model)
model = Sequential()
model.add(Embedding(input_dim=len(word_to_index), output_dim=embedding_size, mask_zero=True, weights=m.layers[0].get_weights()))
for layerNumber in range(0, nb_hidden_layers):
    print "add LSTM layer...."
    model.add(LSTM(units=hidden_dimensions, return_sequences=True, weights=m.layers[layerNumber+1].get_weights()))
model.compile(loss='sparse_categorical_crossentropy', optimizer="adam")



# Open the training dataset
print "Reading training data..."
path  = open(training_data, "r")
train = path.read().decode('utf8')

output = []
for idx, sentence in enumerate(nltk.sent_tokenize(train)): 
	word_list = nltk.word_tokenize(sentence.lower())
	trainInput = np.zeros((1, len(word_list)), dtype=np.int16)
	for index, idx in enumerate(word_list):
			index_idx = word_to_index[idx] if idx in word_to_index else word_to_index[unknown_token]
			trainInput[0, index] = index_idx
	output.append([sentence, model.predict(trainInput, verbose=0)[0][-1].tolist()])

print output
with open(save_path+"/SentenceEmbeddings.txt",'w') as f:
	json.dump(output, f)



