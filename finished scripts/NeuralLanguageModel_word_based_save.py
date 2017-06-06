
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

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
##############################################



vocabulary_size = 799
unknown_token = "UNKNOWN_TOKEN"
maxlen = 40
step = 3

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
print('number of sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(vocab)), dtype=np.bool)
y = np.zeros((len(sentences), len(vocab)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        X[i, t, word_to_index[word]] = 1
    y[i, word_to_index[next_words[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
#model.add(Embedding(input_dim=maxlen, output_dim=128, input_length=len(vocab)))
#model.add(LSTM(return_sequences=True, units=128))
model.add(LSTM(128, input_shape=(maxlen, len(vocab))))

model.add(Dense(len(vocab)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)



# train the model, output generated text after each iteration
for iteration in range(1, 10):
    print ""
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=256, epochs=1)

    start_index = random.randint(0, len(tokens) - maxlen - 1)

    generated = ''
    sentence = tokens[start_index: start_index + maxlen]
    generated += untokenize(sentence)
    generated += " /// "
    print ""
    addedWords = []
    for i in range(20):
        x = np.zeros((1, maxlen, len(vocab)))
        for t, word in enumerate(sentence):
            x[0, t, word_to_index[word]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, 1.)
        next_word = index_to_word[next_index]
        addedWords.append(next_word)
        sentence = sentence[1:]
        sentence.append(next_word)
    generated += untokenize(addedWords)
    print generated

model.save('testModel.h5')
  