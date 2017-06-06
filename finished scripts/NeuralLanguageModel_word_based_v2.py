
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



# Helper functions
##############################################
def untokenize(words):
    text = ' '.join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
         "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()
##############################################


#vocabulary_size = "ALL USED"
unknown_token = "<UNKWN>"
start_token = "<START>"
end_token = "<END>"
step = 1
lookahead = 20

print "Reading file..."
path  = open("./../tedData/Manual_change/original_text.txt", "r")
#path  = open("./../tedData/allTEDTalkContent.txt", "r")
text = path.read().decode('unicode-escape').lower()


print "Tokenizing file..."
# Split the text in sentences and words --> tokens[#sentence][#word]
words = nltk.word_tokenize(text)
tokens = []
for index, sentence in enumerate(nltk.sent_tokenize(text)): 
    tokens.append(nltk.word_tokenize(sentence))


print "Creating vocabulary..."
# All words that occur in the text and cut the first vocabulary_size
vocab = ["<PAD>"]+[word[0] for word in Counter(words).most_common()]
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
        tokens[index1][index2] = word_to_index[word] if word in word_to_index else unknown_token


print "Adding start and end tokens..."
# Pad the sentences to be the same length in the back and add #lookahead starttokens in the beginning
for index, sentence in enumerate(tokens):
    tokens[index] = [word_to_index[start_token]] + tokens[index] + [word_to_index[end_token]]

maxLength = 0
for sentence in tokens:
    if len(sentence) > maxLength:
        maxLength = len(sentence)


print "Padding sequences with zeros..."
padTokens = kerasSequence.pad_sequences(tokens, maxlen=maxLength, padding='pre', value=0)

print padTokens[0]

print "Creating training sequences..."
# Cut the sentences into sequences for training
given_Sequences = []
following_Words = []
for sentence in padTokens:
    for index in range(0, len(sentence) - lookahead, step):
        given_Sequences.append(sentence[index:index + lookahead])
        following_Words.append(sentence[index + lookahead])



# Inforamtion
print('Information...')
print('Number of sequences:', len(given_Sequences))
print('Number of words (vocabulary):', len(vocab))



print('Vectorization...')
trainingInput = np.zeros((len(given_Sequences), lookahead), dtype=np.int32)
trainingOutput = np.zeros((len(given_Sequences), len(vocab)), dtype=np.int32)
for index1, sequence in enumerate(given_Sequences):
    for index2, word in enumerate(sequence):
        trainingInput[index1, index2] = word
    trainingOutput[index, following_Words[index1]] = 1



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
model.add(Embedding(input_dim=len(given_Sequences), output_dim=128, input_length=lookahead, mask_zero=True))
model.add(LSTM(units=128))
model.add(Dense(len(vocab)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=['accuracy'])



# Iterate trough the trainingsset
for iteration in range(1, 10):
    print ""
    print('#' * 50)
    print ""
    print'Iteration: ' + str(iteration)    
    iterationResult = model.fit(trainingInput, trainingOutput, batch_size=128, epochs=1, validation_split=0.2)

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








'''
    # Show current outcome for random sentence part
    randomSentence = padTokens[random.randint(0, len(padTokens) - 1)]
    start_index = random.randint(lookahead/2, len(randomSentence)/2 - 1)
    fullCheckSentence = [index_to_word[word] for word in randomSentence]
    checkSentence = fullCheckSentence[start_index: start_index + lookahead]
    generated = untokenize(checkSentence) + " /// "
    testInput = np.zeros((1, lookahead))
    for index, word in enumerate(checkSentence):
        testInput[0, index] = word_to_index[word]
    prediction = model.predict(testInput, verbose=0)
    values = np.asarray(prediction).astype('float64')
    maxValue = 0
    maxIndex = 0
    for index, value in enumerate(values[0]):
        if value > maxValue:
            maxValue = value
            maxIndex = index

    next_index = maxIndex
    next_word = index_to_word[next_index]
    
    generated += next_word
    print generated
    print ""
    print('#' * 50)
    print ""
'''  