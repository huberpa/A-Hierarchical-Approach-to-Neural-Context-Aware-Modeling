
# Commandline Arguments
##############################################
import optparse
parser = optparse.OptionParser()
parser.add_option('--translation_path', action="store", dest="translation_path", help="The path to the translation data folder (default: .)", default=".")
parser.add_option('--data_path', action="store", dest="data_path", help="The path to the index-to-word data folder (default: .)", default=".")
parser.add_option('--save_path', action="store", dest="save_path", help="The path to the monolingual data folder (default: .)", default=".")
parser.add_option('--name', action="store", dest="name", help="The name of the model (default: model)", default="model")

options, args = parser.parse_args()
data_path = options.translation_path
model_name = options.name
save_path = options.save_path
index_to_word_path = options.data_path
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
english_path  = open(data_path+"/test_en.xml", "r")
english_text = english_path.read()
englishFile = parseString(english_text)
english_documents=englishFile.getElementsByTagName('doc')
for document in english_documents:
	content=document.getElementsByTagName('seg')
	for talk in content:
		node_value = talk.childNodes[0].nodeValue
		english_talks.append(nltk.word_tokenize(node_value))


print "Loading German file..."
german_talks = []
german_path  = open(data_path+"/test_de.xml", "r")
german_text = german_path.read()
germanFile = parseString(german_text)
german_documents=germanFile.getElementsByTagName('doc')
for document in german_documents:
	content=document.getElementsByTagName('seg')
	for talk in content:
		node_value = talk.childNodes[0].nodeValue
		german_talks.append(nltk.word_tokenize(node_value))


print "Change files to lowercase..."
english_talks = [[word.lower() for word in line] for line in english_talks]
german_talks = [[word.lower() for word in line] for line in german_talks]


with open (index_to_word_path+"/index_to_word_eng.txt", 'r') as f:
	index_to_word_english = json.load(f)
with open (index_to_word_path+"/word_to_index_eng.txt", 'r') as f:
	word_to_index_english = json.load(f)
with open (index_to_word_path+"/index_to_word_ger.txt", 'r') as f:
	index_to_word_german = json.load(f)
with open (index_to_word_path+"/word_to_index_ger.txt", 'r') as f:
	word_to_index_german = json.load(f)

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
pad_input_Words = kerasSequence.pad_sequences(english_talks, maxlen=timesteps, padding='post', value=0)
pad_output_Words = kerasSequence.pad_sequences(german_talks, maxlen=timesteps, padding='post', value=0)

print "Save preprocessed data into files..."
if not os.path.exists(save_path):
    os.makedirs(save_path)

if not os.path.exists(save_path+"/"+model_name):
    os.makedirs(save_path+"/"+model_name)

if not os.path.exists(save_path+"/"+model_name+"/input_data.txt"):
	with open(save_path+"/"+model_name+"/input_data.txt",'w') as f:
		json.dump(pad_input_Words.tolist(), f)

if not os.path.exists(save_path+"/"+model_name+"/output_data.txt"):
	with open(save_path+"/"+model_name+"/output_data.txt",'w') as f:
		json.dump(pad_output_Words.tolist(), f)

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
