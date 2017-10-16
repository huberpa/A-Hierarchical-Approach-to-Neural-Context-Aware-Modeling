
# Commandline Arguments
##############################################
import optparse
parser = optparse.OptionParser()
parser.add_option('--story_file', action="store", dest="story_file", help="File with the start of a story to be continued (default: ./story.txt)", default="./story.txt")
parser.add_option('--network_path', action="store", dest="network_path", help="The path to the network file (default: '.')", default=".")
parser.add_option('--data_path', action="store", dest="data_path", help="The path to the word index relation (default: '.')", default=".")
parser.add_option('--sentence_enc_path', action="store", dest="sentence_enc_path", help="The path to the baseline model for the sentence embedding - NEEDS TO BE A KERAS MODEL (Gen 8, 1e-3) (default: '.')", default=".")
parser.add_option('--sentence_log', action="store", dest="sentence_log", help="The path to the sentence embeddings logfiles (default: .)", default=".")

options, args = parser.parse_args()
model_path = options.network_path
data_path = options.data_path
sentence_log = options.sentence_log
story_file = options.story_file
sentence_enc_path = options.sentence_enc_path
##############################################


# Imports
##############################################
import tensorflow as tf 
import numpy as np
import json
import copy
import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, LSTM
import sys
import nltk
from collections import Counter
import math
reload(sys)  
sys.setdefaultencoding('utf-8')
##############################################

# Main
##############################################

# Load files
print "Reading network input data..."
with open (data_path+"/index_to_word.txt", 'r') as f:
	index_to_word = json.load(f)
with open (data_path+"/word_to_index.txt", 'r') as f:
	word_to_index = json.load(f)
with open (story_file, 'r') as f:
	story = f.read()

sentence_logging_text = open(sentence_log, "r").read()
parameters = json.loads(sentence_logging_text[:sentence_logging_text.find("}")+1].replace("'","\""))
embedding_size = int(parameters['embedding_dim'])
nb_hidden_layers = int(parameters['layers'])
hidden_dimensions = int(parameters['layer_dim'])
start_token = parameters['start_token']
end_token = parameters['end_token']
unknown_token = parameters['unknown_token']

word_story = []
sentence_embeddings = []
for sentence in nltk.sent_tokenize(story):
	word_story.append([start_token] + nltk.word_tokenize(sentence.lower()) + [end_token])

for index1, sentence in enumerate(word_story):
	for index2, word in enumerate(sentence):
		word_story[index1][index2] = word_to_index[word] if word in word_to_index else word_to_index["<UNKWN>"]



# GET THE SENTENCE REP NOW FOR THE START --> WORD_STORY NOT NEEDED AFTER
sentenceModel = load_model(sentence_enc_path)
newSentenceModel = Sequential()
newSentenceModel.add(Embedding(input_dim=len(word_to_index), output_dim=embedding_size, mask_zero=True, weights=sentenceModel.layers[0].get_weights()))
for layerNumber in range(0, nb_hidden_layers):
	print "add LSTM layer...."
	newSentenceModel.add(LSTM(units=hidden_dimensions, return_sequences=True, weights=sentenceModel.layers[layerNumber+1].get_weights()))
newSentenceModel.compile(loss='sparse_categorical_crossentropy', optimizer="adam")

for sentence in word_story:
	trainInput = np.zeros((1, len(sentence)), dtype=np.int16)
	for index, word in enumerate(sentence):
		trainInput[0, index] = word
	sentence_embeddings.append(newSentenceModel.predict(trainInput, verbose=0)[0][-1].tolist())

max_sentence_embeddings = 10
number_sentences = 10


for _ in range (0, number_sentences):

	encoder_length = len(sentence_embeddings)
	decoder_length = 100
	empty = [0]*512
	sentence_input = copy.copy(sentence_embeddings)
	for _ in range(0, 10-len(sentence_embeddings)):
		sentence_input.append(empty)

	print "Launch the graph..."
	session_config = tf.ConfigProto(allow_soft_placement=True)    
	session_config.gpu_options.per_process_gpu_memory_fraction = 0.90
	with tf.Session(config=session_config) as session:
		tf.train.import_meta_graph(model_path + ".meta").restore(session, model_path)
		variables = tf.get_collection('variables_to_store')

		print "Start Creating..."
		feed = {}
		feed["encoder_inputs"] = [sentence_input]
		feed["encoder_length"] = [encoder_length]
		feed["decoder_length"] = [decoder_length]
		feed["start_token_infer"] = [word_to_index[start_token]]

		#print feed["encoder_inputs"]
		#print feed["encoder_length"]
		#print feed["decoder_length"]
		#print feed["start_token_infer"]

		sentence_output = session.run(variables[9], feed_dict={variables[3]:feed["encoder_inputs"], variables[7]: feed["encoder_length"], variables[8]: feed["decoder_length"], variables[10]: feed["start_token_infer"]})[0]
		sentence_output = []
		for word in sentence_output:
			if word != 0:
				sentence_output = sentence_output + [index_to_word[str(word)]]

		print sentence_output
		#save in file
		with open(story_file, "a") as f:
			string_sentence = ""
			for element in sentence_output:
				string_sentence += string_sentence + " " + element	
			print "write into file: "+string_sentence
			f.write("{}\n".format(string_sentence))

	trainInput = np.zeros((1, len(sentence_output)), dtype=np.int16)
	for index, word in enumerate(output):
		trainInput[0, index] = word
	sentence_embeddings.append(newSentenceModel.predict(trainInput, verbose=0)[0][-1].tolist())
	if len(sentence_embeddings) > 10:
		sentence_embeddings = sentence_embeddings[1:]

##############################################

