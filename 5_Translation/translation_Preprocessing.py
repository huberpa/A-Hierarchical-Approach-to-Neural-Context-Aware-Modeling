
# Commandline Arguments
##############################################
import optparse
parser = optparse.OptionParser()
parser.add_option('--translation_path', action="store", dest="translation_path", help="The path to the translation data folder (default: .)", default=".")
parser.add_option('--mono_path', action="store", dest="mono_path", help="The path to the monolingual data folder (default: .)", default=".")
parser.add_option('--save_path', action="store", dest="save_path", help="The path to the monolingual data folder (default: .)", default=".")
parser.add_option('--name', action="store", dest="name", help="The name of the model (default: model)", default="model")

options, args = parser.parse_args()
data_path = options.translation_path
mono_path = options.mono_path
model_name = options.name
save_path = options.save_path

timesteps = 100
vocabulary_size = 30000
unknown_token = "<UNKWN>"
start_token = "<START>"
end_token = "<END>"
##############################################

# Imports
##############################################
#import tensorflow as tf 
import numpy as np
from xml.dom.minidom import parse, parseString
import json
import os
from collections import Counter
import datetime
import sys
from keras.preprocessing import sequence as kerasSequence
import nltk
reload(sys)  
sys.setdefaultencoding('utf-8')
##############################################

# Main
##############################################

# Load files
print "Loading English file..."
english_talks = []
english_words = []
english_path  = open(data_path+"/translation_training_english.txt", "r")
english_text = english_path.read().decode('unicode_escape')
splitted_talk = english_text.split('\n')
for line in splitted_talk:
	if len(line) > 1:
		english_talks.append(nltk.word_tokenize(line))
		english_words = english_words + nltk.word_tokenize(line)

print "Loading German file..."
german_talks = []
german_words = []
german_path  = open(data_path+"/translation_training_german.txt", "r")
german_text = german_path.read().decode('unicode_escape')
splitted_talk = german_text.split('\n')
for line in splitted_talk:
	if len(line) > 1:
		german_talks.append(nltk.word_tokenize(line))
		german_words = german_words + nltk.word_tokenize(line)

print "Loading monoligual data..."
mono_path  = open(mono_path, "r")
mono_text = mono_path.read().decode('unicode_escape')
mono_words = nltk.word_tokenize(mono_text)
mono_words = [word.lower() for word in mono_words]

print "Change files to lowercase..."
english_words = [word.lower() for word in english_words]
german_words = [word.lower() for word in german_words]
english_talks = [[word.lower() for word in line] for line in english_talks]
german_talks = [[word.lower() for word in line] for line in german_talks]

all_english_words = english_words + mono_words

print "Creating English vocabulary..."
english_allVocab = [word[0] for word in Counter(all_english_words).most_common()]
english_vocab = ["<PAD>"]+english_allVocab[:vocabulary_size]
english_vocab.append(unknown_token)
english_vocab.append(start_token)
english_vocab.append(end_token)

print "Creating German vocabulary..."
german_allVocab = [word[0] for word in Counter(german_words).most_common()]
german_vocab = ["<PAD>"]+german_allVocab[:vocabulary_size]
german_vocab.append(unknown_token)
german_vocab.append(start_token)
german_vocab.append(end_token)

# Mapping from word to Index and vice versa
print "Creating mappings..."
index_to_word_english = dict((index, word) for index, word in enumerate(english_vocab))
word_to_index_english = dict((word, index) for index, word in enumerate(english_vocab))
index_to_word_german = dict((index, word) for index, word in enumerate(german_vocab))
word_to_index_german = dict((word, index) for index, word in enumerate(german_vocab))

# Adding the unknown_token and transforming the text into indexes
print "Creating integer array English..."
for index1, sentence in enumerate(english_talks):
    for index2, word in enumerate(sentence):
        english_talks[index1][index2] = word_to_index_english[word] if word in word_to_index_english else word_to_index_english[unknown_token]

print "Creating integer array German..."
for index1, sentence in enumerate(german_talks):
    for index2, word in enumerate(sentence):
        german_talks[index1][index2] = word_to_index_german[word] if word in word_to_index_german else word_to_index_german[unknown_token]

# Adding the start- and end-tokens to the sentences
print "Adding start and end tokens..."
for index, _ in enumerate(english_talks):
    english_talks[index] = [word_to_index_english[start_token]] + english_talks[index] + [word_to_index_english[end_token]]
    german_talks[index] = [word_to_index_german[start_token]] + german_talks[index] + [word_to_index_german[end_token]]

# Cut all sentences that are longer than 50 words
for index, sentence in enumerate(english_talks):
    if len(sentence) > timesteps:
        english_talks[index] = english_talks[index][:timesteps]

for index, sentence in enumerate(german_talks):
    if len(sentence) > timesteps:
        german_talks[index] = german_talks[index][:timesteps]

# Pad the sequences with zeros
print "Padding sequences with zeros..."
pad_input_Words = []
pad_output_Words = []

for english_talk in english_talks:
	pad_input_Words.append(english_talk + [0]*(timesteps-len(english_talk)))

for german_talk in german_talks:
	pad_output_Words.append(german_talk + [0]*(timesteps-len(german_talk)))

print "Save preprocessed data into files..."
if not os.path.exists(save_path):
    os.makedirs(save_path)

if not os.path.exists(save_path+"/"+model_name):
    os.makedirs(save_path+"/"+model_name)

if not os.path.exists(save_path+"/"+model_name+"/input_data.txt"):
	with open(save_path+"/"+model_name+"/input_data.txt",'w') as f:
		json.dump(pad_input_Words, f)

if not os.path.exists(save_path+"/"+model_name+"/output_data.txt"):
	with open(save_path+"/"+model_name+"/output_data.txt",'w') as f:
		json.dump(pad_output_Words, f)

if not os.path.exists(save_path+"/"+model_name+"/word_to_index_eng.txt"):
	with open(save_path+"/"+model_name+"/word_to_index_eng.txt",'w') as f:
		json.dump(word_to_index_english, f)

if not os.path.exists(save_path+"/"+model_name+"/index_to_word_eng.txt"):
	with open(save_path+"/"+model_name+"/index_to_word_eng.txt",'w') as f:
		json.dump(index_to_word_english, f)

if not os.path.exists(save_path+"/"+model_name+"/word_to_index_ger.txt"):
	with open(save_path+"/"+model_name+"/word_to_index_ger.txt",'w') as f:
		json.dump(word_to_index_german, f)

if not os.path.exists(save_path+"/"+model_name+"/index_to_word_ger.txt"):
	with open(save_path+"/"+model_name+"/index_to_word_ger.txt",'w') as f:
		json.dump(index_to_word_german, f)

print "Preprocessing finished..."

