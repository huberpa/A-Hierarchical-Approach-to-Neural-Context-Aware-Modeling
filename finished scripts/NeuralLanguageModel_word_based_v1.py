
# Imports
##############################################
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import nltk
from collections import Counter
import re
from keras import layers
from keras.models import load_model
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

# TODOs:
# Sentence level
# Logging
# Save after few Training steps


vocabulary_size = 799 # More!!!
unknown_token = "<UNKWN>"
maxlen = 40 # Sentence Level
step = 1

path  = open("./../tedData/Manual_change/original_text.txt", "r")
text = path.read().decode('unicode-escape').lower()

tokens = nltk.word_tokenize(text)

# All words that occur in the text and cut the first vocabulary_size
vocab = [word[0] for word in Counter(tokens).most_common()][:vocabulary_size]
vocab.append(unknown_token)

# Mapping from word to Index and vice versa
index_to_word = dict((index, word) for index, word in enumerate(vocab))
word_to_index = dict((word, index) for index, word in enumerate(vocab))

for index, word in enumerate(tokens):
    tokens[index] = word if word in word_to_index else unknown_token

# cut the text in sequences of maxlen characters
sentences = []
next_words = []
for i in range(0, len(tokens) - maxlen, step):
    sentences.append(tokens[i: i + maxlen])
    next_words.append(tokens[i + maxlen])

print ('*'*20)
print sentences[0]
print next_words[0]
print ('*'*20)
print sentences[1]
print next_words[1]
print ('*'*20)
print sentences[2]
print next_words[2]
print ('*'*20)
print sentences[3]
print next_words[3]
print ('*'*20)

print('number of sequences:', len(sentences))
print('number of words (vocabulary):', len(vocab))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen), dtype=np.int32)
y = np.zeros((len(sentences), maxlen, len(vocab)), dtype=np.int32)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        X[i, t] = word_to_index[word]
    y[i, 0, word_to_index[next_words[i]]] = 1

#print X
#print ""
#print y

# build the model
print('Building the Model...')
model = Sequential()
model.add(Embedding(input_dim=len(sentences), output_dim=128, input_length=maxlen))
model.add(LSTM(return_sequences=True, units=128))
model.add(Dense(len(vocab)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer="Adam")


for iteration in range(1, 10):
    print ""
    print ""
    print ""
    print('#' * 50)
    print ""
    print'Iteration: ' + str(iteration)

    # Train    
    model.fit(X, y, batch_size=128, epochs=1)

    # Show current outcome for random substring
    start_index = random.randint(0, len(tokens) - maxlen - 1)

    generated = ''
    sentence = tokens[start_index: start_index + maxlen]
    generated += untokenize(sentence)
    generated += " /// "
    print ""
    print ""
    addedWords = []
    for i in range(1):
        x = np.zeros((1, maxlen))
        for t, word in enumerate(sentence):
            x[0, t] = word_to_index[word]

        preds = model.predict(x, verbose=0)[0]
        values = np.asarray(preds).astype('float64')
        maxValue = 0
        maxIndex = 0
        #print values[0]
        for index, value in enumerate(values[0]):
            #print value
            if value > maxValue:
                maxValue = value
                maxIndex = index

        next_index = maxIndex
        next_word = index_to_word[next_index]
        addedWords.append(next_word)
        sentence = sentence[1:]
        sentence.append(next_word)
    generated += untokenize(addedWords)
    print generated
    print ""
    print('#' * 50)
    print ""
model.save('testModel.h5')
  