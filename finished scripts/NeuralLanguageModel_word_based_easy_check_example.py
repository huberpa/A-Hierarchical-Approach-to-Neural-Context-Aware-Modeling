
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



# Variable definition
##############################################
vocabulary_size = 10000
unknown_token = "<UNKWN>"
start_token = "<START>"
end_token = "<END>"
batch_size = 20
trainingIterations = 30
##############################################



# Main
##############################################

# Open the training dataset
print "Reading training data..."
path  = open("./../tedData/Manual_change/original_text.txt", "r")
#path  = open("./../tedData/allTEDTalkContent.txt", "r")
#path  = open("./../tedData/sets/training/original_training_texts.txt", "r")
text = path.read().decode('utf8').lower()

# Split the text in sentences and words --> tokens[#sentence][#word]
print "Tokenizing file..."
words = nltk.word_tokenize(text)
tokens = []
for index, sentence in enumerate(nltk.sent_tokenize(text)): 
    tokens.append(nltk.word_tokenize(sentence))

# All words that occur in the text and cut the first vocabulary_size
print "Creating vocabulary..."
vocab = ["<PAD>"]+[word[0] for word in Counter(words).most_common()][:vocabulary_size]
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

print "Adding start and end tokens..."
for index, sentence in enumerate(tokens):
    tokens[index] = [word_to_index[start_token]] + tokens[index] + [word_to_index[end_token]]

longestSentence = 0
for sentence in tokens:
    if len(sentence) > longestSentence:
        longestSentence = len(sentence)

input_Words = []
output_Words = []
for sentence in tokens:
    input_Words.append(sentence[:len(sentence)-1])
    output_Words.append(sentence[1:len(sentence)])

print "Padding sequences with zeros..."
pad_input_Words = kerasSequence.pad_sequences(input_Words, maxlen=longestSentence, padding='pre', value=0)
pad_output_Words = kerasSequence.pad_sequences(output_Words, maxlen=longestSentence, padding='pre', value=0)

# Print information
print 'Finished initialization with......'
print 'Number of sequences:' + str(len(input_Words))
print 'Number of words (vocabulary):' + str(len(vocab))
print 'Length of Input (Longest sentence):' + str(longestSentence)

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


trainingInput = [[0, 1], [1, 2], [2, 3]]
trainingOutput = [[1, 2], [2, 3], [3, 4]]
trainingOutput = np.expand_dims(trainingOutput, -1)


# Create folder for logging
if not os.path.exists("./files"):
    os.makedirs("./files")
if not os.path.exists("./log"):
    os.makedirs("./log")
if not os.path.exists("./models"):
    os.makedirs("./models")
logFile = open("./log/logging.log","a")
logFile.close()

with open("./files/index_to_word.json", "w") as f:
    json.dump(index_to_word, f)

with open("./files/word_to_index.json", "w") as f:
    json.dump(word_to_index, f)

# Build the model
print('Building the Model...')
model = Sequential()
model.add(Embedding(input_dim=5, output_dim=128, mask_zero=True))
model.add(LSTM(units=128, return_sequences=True))
model.add(TimeDistributed(Dense(5, activation='softmax')))
model.compile(loss='sparse_categorical_crossentropy', optimizer="Adam")

print model.summary() 

#srun -hostname
#+ extra bash
#nvidia-smi

# Iterate trough the trainingsset
for iteration in range(1, trainingIterations+1):
    print ""
    print('#' * 50)
    print ""
    print'Iteration: ' + str(iteration)

    iterationResult = model.fit(trainingInput, trainingOutput, batch_size=batch_size, epochs=1)

'''
    # Log the iteration results
    with open("./log/logging.log", "a") as logFile:
        logDate = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logIteration = " --- Iteration: " + str(iteration)
        logLoss = " --- Loss: " + str(iterationResult.history.get('loss')[0])
    
        logFile.write("{}\n".format(""))
        logFile.write("{}\n".format(logDate+logIteration+logLoss))
        logFile.write("{}\n".format(""))
        logFile.write("{}\n".format("+"*50))

    model.save('./models/LM_model_Iteration_' + str(iteration) + '.h5')
'''

test = np.zeros((1, 1), dtype=np.int16)
test[0][0] = 2
#test[0][1] = 3

prediction = model.predict(test, verbose=0)[0]

outputs = []
for index1, word in enumerate(prediction):
    maxValue = 0
    maxIndex = 0
    for index2, wordProbability in enumerate(word):
        if wordProbability > maxValue:
            maxValue = wordProbability
            maxIndex = index2
    outputs.append(maxIndex)

print test
print outputs
