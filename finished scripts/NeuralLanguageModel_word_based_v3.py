
# Imports
##############################################
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, LSTM
from keras.preprocessing import sequence as kerasSequence

from collections import Counter
import numpy as np
import random
import sys
import nltk
import re
import os
import datetime
##############################################



vocabulary_size = 10000
unknown_token = "<UNKWN>"
start_token = "<START>"
end_token = "<END>"
lookahead = 20

print "Reading file..."
path  = open("./../tedData/Manual_change/original_text.txt", "r")
#path  = open("./../tedData/allTEDTalkContent.txt", "r")
#path  = open("./../tedData/sets/training/original_training_texts.txt", "r")
text = path.read().decode('utf8').lower()


print "Tokenizing file..."
# Split the text in sentences and words --> tokens[#sentence][#word]
words = nltk.word_tokenize(text)
tokens = []
for index, sentence in enumerate(nltk.sent_tokenize(text)): 
    tokens.append(nltk.word_tokenize(sentence))


print "Creating vocabulary..."
# All words that occur in the text and cut the first vocabulary_size
vocab = ["<PAD>"]+[word[0] for word in Counter(words).most_common()][:vocabulary_size]
vocab.append(unknown_token)
vocab.append(start_token)
vocab.append(end_token)


print "Creating mappings..."
# Mapping from word to Index and vice versa
index_to_word = dict((index, word) for index, word in enumerate(vocab))
word_to_index = dict((word, index) for index, word in enumerate(vocab))


print "Creating integer array..."
# Adding the unknown_token and transforming the text into indexes
for index1, sentence in enumerate(tokens):
    for index2, word in enumerate(sentence):
        tokens[index1][index2] = word_to_index[word] if word in word_to_index else word_to_index[unknown_token]


print "Adding start and end tokens..."
# Pad the sentences to be the same length in the back and add #lookahead starttokens in the beginning
for index, sentence in enumerate(tokens):
    tokens[index] = [word_to_index[start_token]] + tokens[index] + [word_to_index[end_token]]


longestSentence = 0
for sentence in tokens:
    if len(sentence) > longestSentence:
        longestSentence = len(sentence)



# Create folder for logging
if not os.path.exists("./log"):
    os.makedirs("./log")
if not os.path.exists("./models"):
    os.makedirs("./models")
logFile = open("./log/logging.log","a")
logFile.close()



# Build the model
print('Building the Model...')
model = Sequential()
model.add(Embedding(input_dim=len(vocab)+1, output_dim=128, input_length=longestSentence, mask_zero=True))
model.add(LSTM(units=128, return_sequences=True))
model.add(Dense(len(vocab)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=['accuracy'])



# Iterate trough the trainingsset
for iteration in range(1, 10):
    print ""
    print('#' * 50)
    print ""
    print'Iteration: ' + str(iteration)

    print "Create mini batch..."
    # Cut the sentences into input and output sequences for training

    np.random.shuffle(tokens)
    print tokens[0]
    print tokens[10]

    batchSize = 50
    for index in range(0, len(tokens)-batchSize, batchSize):
        input_Words = []
        output_Words = []
        for sentence in tokens[index:index+batchSize]:
            input_Words.append(sentence[:len(sentence)-1])
            output_Words.append(sentence[1:len(sentence)])

        print "Padding sequences with zeros..."
        pad_input_Words = kerasSequence.pad_sequences(input_Words, maxlen=longestSentence, padding='pre', value=0)
        pad_output_Words = kerasSequence.pad_sequences(output_Words, maxlen=longestSentence, padding='pre', value=0)

        #print pad_input_Words[0]
        #print pad_output_Words[0]

        # Inforamtion
        print 'Information...'
        print 'Number of sequences:' + str(len(input_Words))
        print 'Number of words (vocabulary):' + str(len(vocab))
        print 'Length of Input (Longest sentence):' + str(longestSentence)

        print('Vectorization...')
        trainingInput = np.zeros((len(pad_input_Words), longestSentence), dtype=np.int16)
        trainingOutput = np.zeros((len(pad_output_Words), longestSentence, len(vocab)), dtype=np.int8)

        for index1, sequence in enumerate(pad_input_Words):
            for index2, word in enumerate(sequence):
                trainingInput[index1, index2] = word

        for index1, sequence in enumerate(pad_output_Words):
            for index2, word in enumerate(sequence):        
                trainingOutput[index1, index2, word] = 1

        iterationResult = model.fit(trainingInput, trainingOutput, batch_size=10, epochs=1, validation_split=0.1)

        # Log the iteration results
        with open("./log/logging.log", "a") as logFile:
            logDate = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logIteration = " --- Iteration: " + str(iteration)
            logAccurancy = " --- Accurancy: " + str(iterationResult.history.get('acc')[0])
            logLoss = " --- Loss: " + str(iterationResult.history.get('loss')[0])
        
            logFile.write("{}\n".format(""))
            logFile.write("{}\n".format(logDate+logIteration+logAccurancy+logLoss))
            logFile.write("{}\n".format(""))
            logFile.write("{}\n".format("+"*50))

    model.save('./models/LM_model_Iteration_' + str(iteration) + '.h5')


# Test
'''
testValues = [807, 26, 73, 83, 6, 236, 8, 456]
pad_input_Words = kerasSequence.pad_sequences([testValues], maxlen=longestSentence, padding='pre', value=0)
testInput = np.asarray(pad_input_Words).astype('int32')
prediction = model.predict(testInput, verbose=0)

testOutput = []
for index, value in enumerate(np.asarray(prediction).astype('float64')[0]):
    testOutput.append(np.argmax(value))
print "prediction:"
print np.asarray(testOutput).astype('int32')
'''