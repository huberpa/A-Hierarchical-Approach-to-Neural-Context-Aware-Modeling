
# Commandline Arguments
##############################################
import optparse
parser = optparse.OptionParser()
parser.add_option('--dataset_path', action="store", dest="dataset", help="Path to the dataset to train on", default=".")
parser.add_option('--layers', action="store", dest="layers", help="The number of hidden layers in the model (default: 1)", default=1)
parser.add_option('--layer_dim', action="store", dest="layer_dim", help="The number of neurons in the hidden layer(s)  (default: 512)", default=512)
parser.add_option('--embedding_dim', action="store", dest="embedding_dim", help="The number of dimensions the embedding has  (default: 256)", default=256)
parser.add_option('--batch_size', action="store", dest="batch_size", help="The batch size of the model (default: 100)", default=100)
parser.add_option('--epochs', action="store", dest="epochs", help="The number of training epochs (default: 20)", default=20)
parser.add_option('--vocabulary_path', action="store", dest="vocabulary", help="Path to the vocabulary to train on", default=".")
parser.add_option('--unknown_token', action="store", dest="unknown_token", help="Token for words that are not in the vacabulary (default: <UNKWN>)", default="<UNKWN>")
parser.add_option('--start_token', action="store", dest="start_token", help="Token for start (default: <START>)", default="<START>")
parser.add_option('--end_token', action="store", dest="end_token", help="Token for end (default: <END>)", default="<END>")
parser.add_option('--save_path', action="store", dest="save_path", help="The path to save the folders (logging, models etc.)  (default: .)", default=".")
options, args = parser.parse_args()
nb_hidden_layers = int(options.layers)
hidden_dimensions= int(options.layer_dim)
embedding_size = int(options.embedding_dim)
batch_size = int(options.batch_size)
trainingIterations = int(options.epochs)
path_to_folders = options.save_path
vocabulary = options.vocabulary
unknown_token = options.unknown_token
start_token = options.start_token
end_token = options.end_token
dataset = options.dataset
##############################################

# Imports
##############################################
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, LSTM
from keras.preprocessing import sequence as kerasSequence
from keras.layers.wrappers import TimeDistributed
from keras import optimizers
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
path  = open(dataset, "r")
text = path.read().decode('utf8')

print "Tokenizing file..."
words = nltk.word_tokenize(text)
tokens = []
for index, sentence in enumerate(nltk.sent_tokenize(text)): 
    tokens.append(nltk.word_tokenize(sentence))

print "Reading vocabulary data..."
path  = open(vocabulary, "r")
vocab_file = path.read().decode('utf8')

print "Creating vocabulary..."
vocab = ["<PAD>"] + vocab_file.splitlines()
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

longestSentence = 50
for index, sentence in enumerate(tokens):
    if len(sentence) > longestSentence:
        tokens[index] = tokens[index][:50]

input_Words = []
output_Words = []
for sentence in tokens:
    input_Words.append(sentence[:len(sentence)-1])
    output_Words.append(sentence[1:len(sentence)])

print "Padding sequences with zeros..."
pad_input_Words = kerasSequence.pad_sequences(input_Words, maxlen=longestSentence, padding='pre', value=0)
pad_output_Words = kerasSequence.pad_sequences(output_Words, maxlen=longestSentence, padding='pre', value=0)

print 'Finished initialization with......'
print 'Number of sequences:' + str(len(input_Words))
print 'Number of words (vocabulary):' + str(len(vocab))

print('Vectorization...')
trainingInput = np.zeros((len(pad_input_Words), longestSentence), dtype=np.int16)
trainingOutput = np.zeros((len(pad_output_Words), longestSentence), dtype=np.int16)

for index1, sequence in enumerate(pad_input_Words):
    for index2, word in enumerate(sequence):
        trainingInput[index1, index2] = word

for index1, sequence in enumerate(pad_output_Words):
    for index2, word in enumerate(sequence):        
        trainingOutput[index1, index2] = word

trainingOutput = np.expand_dims(trainingOutput, -1)

idx = 1
loop = 1
while loop:
    if not os.path.exists(path_to_folders + "/files_"+str(idx)):
        os.makedirs(path_to_folders + "/files_"+str(idx))
        with open(path_to_folders + "/files_"+str(idx)+"/index_to_word.json", "w") as f:
            json.dump(index_to_word, f)
        with open(path_to_folders + "/files_"+str(idx)+"/word_to_index.json", "w") as f:
            json.dump(word_to_index, f)
    if not os.path.exists(path_to_folders + "/log_"+str(idx)):
        os.makedirs(path_to_folders + "/log_"+str(idx))
        logFile = open(path_to_folders + "/log_"+str(idx)+"/logging.log","a")
        logFile.close()
    if not os.path.exists(path_to_folders + "/models_"+str(idx)):
        os.makedirs(path_to_folders + "/models_"+str(idx))
        loop = 0
    if loop == 1:
        idx += 1

print('Building the Model...')
model = Sequential()
model.add(Embedding(input_dim=len(vocab), output_dim=embedding_size, mask_zero=True))
for layer in range(0, nb_hidden_layers):
    model.add(LSTM(units=hidden_dimensions, return_sequences=True))
model.add(TimeDistributed(Dense(len(vocab), activation='softmax')))
optimizer = optimizers.Adam(lr=0.0001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)

print model.summary() 

for iteration in range(0, trainingIterations):

    print ""
    print('#' * 50)
    print ""
    print'Iteration: ' + str(iteration)

    iterationResult = model.fit(trainingInput, trainingOutput, batch_size=int(batch_size), epochs=1)

    with open(path_to_folders + "/log_" + str(idx) + "/logging.log", "a") as logFile:
        if iteration == 1:
            logFile.write("{}\n".format(options))

        logData = training_data
        logDate = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logIteration = " --- Iteration: " + str(iteration)
        logLoss = " --- Loss: " + str(iterationResult.history.get('loss')[0])
    
        logFile.write("{}\n".format(""))
        logFile.write("{}\n".format(logData + " --- " + logDate+logIteration+logLoss))
        logFile.write("{}\n".format(""))
        logFile.write("{}\n".format("+"*50))

    model.save(path_to_folders + "/models_" + str(idx) + "/LM_model_Iteration_" + str(iteration) + '.h5')