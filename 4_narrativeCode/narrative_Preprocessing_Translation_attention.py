
# Commandline Arguments
##############################################
import optparse
parser = optparse.OptionParser()
parser.add_option('--sentence_forecast', action="store", dest="sentence_forecast", help="The number of past sentence that are taken into account (default: 10)", default=10)
parser.add_option('--embedding_sentence_length', action="store", dest="embedding_sentence_length", help="The maximal length for encoder inputs (default: 100)", default=100)
parser.add_option('--sentence_max_length', action="store", dest="sentence_max_length", help="The maximal sentence length for decoder inputs (default: 50)", default=50)
parser.add_option('--sentence_model', action="store", dest="sentence_model", help="The path to the sentence embeddings model (default: .)", default=".")
parser.add_option('--sentence_vocab', action="store", dest="sentence_vocab", help="The path to the sentence embeddings vocabulary (default: .)", default=".")
parser.add_option('--sentence_log', action="store", dest="sentence_log", help="The path to the sentence embeddings logfiles (default: .)", default=".")
parser.add_option('--save_path', action="store", dest="save_path", help="The path to save the files (default: .)", default=".")
parser.add_option('--corpus', action="store", dest="corpus", help="The corpus that should be used (PRD or DEV or TEST)  (default: PRD)", default="PRD")

options, args = parser.parse_args()
sentence_model = options.sentence_model
sentence_vocab = options.sentence_vocab
sentence_log = options.sentence_log
save_path = options.save_path
embedding_sentence_length = int(options.embedding_sentence_length)
max_sentence_length = int(options.sentence_max_length)
max_sentence_forecast = int(options.sentence_forecast)
corpus = options.corpus
if corpus == "DEV":
    training_data = "./../tedData/sets/development/talkseperated_indicated_development_texts.txt"
if corpus == "TEST":
    training_data = "./../tedData/sets/test/talkseperated_indicated_test_texts.txt"
if corpus == "PRD":
    training_data = "./../tedData/sets/training/talkseperated_original_training_texts.txt"
##############################################

# Imports
##############################################
import tensorflow as tf
import numpy as np
import nltk
import os
import json
import copy
##############################################

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Main
##############################################

# Open the sentence log to find out about the used parameters
print "Reading sentence log data..."
path  = open(sentence_log, "r")
sentence_logging_text = path.read().decode('utf8')
print sentence_logging_text[sentence_logging_text.find("{"):sentence_logging_text.find("}")+1].replace("'","\"")
parameters = json.loads(sentence_logging_text[sentence_logging_text.find("{"):sentence_logging_text.find("}")+1].replace("'","\""))

embedding_size = int(parameters['embedding_size'])
nb_hidden_layers = int(parameters['layer_number'])
hidden_dimensions = int(parameters['layer_dimension'])
start_token = "<START>"
end_token = "<END>"
unknown_token = "<UNKWN>"

# Open the training dataset
print "Reading training data..."
path  = open(training_data, "r")
train = path.read().decode('utf8')

# Open the sentence index_to_word and word_to_index
print "Reading sentence vocab data..."
word_to_index = []
with open(sentence_vocab + "/word_to_index_eng.txt") as f:    
	word_to_index = json.load(f)
index_to_word = []
with open(sentence_vocab + "/index_to_word_eng.txt") as f:    
	index_to_word = json.load(f)

# Split the text in talks, sentences and words --> tokens[#talks][#sentence][#word]
print "Tokenizing file..."
plain_talks = []
walkthrough_text = train
start_current_talk = 0
end_current_talk = 0
while walkthrough_text.find("</TALK>") > -1:
	start_current_talk = walkthrough_text.find("<TALK>")
	end_current_talk = walkthrough_text.find("</TALK>")
	plain_talks.append(walkthrough_text[start_current_talk+6:end_current_talk])
	walkthrough_text = walkthrough_text[end_current_talk+7:]

# Create 3D array with [#talks][#sentence][#word]
print "Create 3D Tensor..."
talks = [0]*len(plain_talks)
for index, talk in enumerate(plain_talks):
	sentences = nltk.sent_tokenize(talk)
	talks[index] =[0]*len(sentences)
	for idx, sentence in enumerate(sentences): 
	    talks[index][idx] = nltk.word_tokenize(sentence.lower())

# Launch the graph
print "Launch the graph..."
session_config = tf.ConfigProto(allow_soft_placement=True)    
session_config.gpu_options.per_process_gpu_memory_fraction = 0.90
with tf.Session(config=session_config) as session:
	tf.train.import_meta_graph(sentence_model + ".meta").restore(session, sentence_model)
	variables = tf.get_collection('variables_to_store')

	# Transform text into 3d tensor with indizes and sentence embeddings
	print "Calculate sentence embeddings..."
	talks_numerified = []
	talks_original = []
	talk_sentence_embedding = []
	talk_maxLength = 0 # 518 for PRD
	for index0, talk in enumerate(talks):
		if len(talk) > talk_maxLength:
			talk_maxLength = len(talk)
		talks_numerified.append([])
		talk_sentence_embedding.append([])
		talks_original.append([])

		for index1, sentence in enumerate(talk):
			talks_numerified[index0].append([])
			talks_original[index0].append([])
			
			talks_numerified[index0][index1].append(word_to_index[start_token])
			talks_original[index0][index1].append(start_token)

			for index2, word in enumerate(sentence):
				checkedWord = ""
				if word.find("___") == -1:
					checkedWord = word
				else:
					checkedWord = word[word.find("___")+3:word.rfind("___")]

				talks_numerified[index0][index1].append(word_to_index[checkedWord] if checkedWord in word_to_index else word_to_index[unknown_token])
				talks_original[index0][index1].append(word if checkedWord in word_to_index else unknown_token)

			talks_numerified[index0][index1].append(word_to_index[end_token])
			talks_original[index0][index1].append(end_token)

			trainInput = np.zeros((1, embedding_sentence_length), dtype=np.int16)
			for index, idx in enumerate(talks_numerified[index0][index1][:embedding_sentence_length]):
				trainInput[0, index] = idx

			test_result = session.run(variables[11], feed_dict={variables[3]:trainInput, variables[7]: [len(trainInput)
				]})
			test_result.tolist()
			sentence_encoding = test_result[0]
			talk_sentence_embedding[index0].append(sentence_encoding)

	for index, talk in enumerate(talk_sentence_embedding):
		for index2, sentence in enumerate(talk):
			talk_sentence_embedding[index][index2] = talk_sentence_embedding[index][index2].tolist()


# Shape input and output in proper shapes
training_encoder_input_data = []
training_decoder_input_data = []
training_decoder_output_data = []
training_decoder_original_output_data = []

print "Create network input shape..."
for index1, talk in enumerate(talk_sentence_embedding):
	training_encoder_input_talk = []

	# From sentence 2 ... n
	for index2, sentence_Embedding in enumerate(talk[:len(talk)-1]):
		# ENCODER
		if len(training_encoder_input_talk) < max_sentence_forecast:
			training_encoder_input_talk.append(sentence_Embedding)
		else:
			training_encoder_input_talk.pop(0)
			training_encoder_input_talk.append(sentence_Embedding)
		training_encoder_input_data.append(copy.copy(training_encoder_input_talk))

		#DECODER
		training_decoder_input_talk = talks_numerified[index1][index2+1]
		training_decoder_output_talk = talks_numerified[index1][index2+1]
		training_decoder_original_output_talk = talks_original[index1][index2+1]
		if len(training_decoder_input_talk) > max_sentence_length:
			training_decoder_input_talk = training_decoder_input_talk[:max_sentence_length+1]
			training_decoder_output_talk = training_decoder_output_talk[:max_sentence_length+1]
			training_decoder_original_output_talk = training_decoder_original_output_talk[:max_sentence_length+1]
		training_decoder_input_talk = training_decoder_input_talk[:len(training_decoder_input_talk)-1]
		training_decoder_output_talk = training_decoder_output_talk[1:]
		training_decoder_original_output_talk = training_decoder_original_output_talk[1:]
		training_decoder_input_data.append(training_decoder_input_talk)
		training_decoder_output_data.append(training_decoder_output_talk)
		training_decoder_original_output_data.append(training_decoder_original_output_talk)

# Find lengths
print "Calculate dynamic lengths..."
training_encoder_input_length = []
training_decoder_input_length = []
training_decoder_mask = []

for history in training_encoder_input_data:
	training_encoder_input_length.append(len(history))

for idx, sentence in enumerate(training_decoder_input_data):
	training_decoder_input_length.append(len(sentence))
	training_decoder_mask.append([])
	for _ in range(len(sentence)):
		training_decoder_mask[idx].append(1.0)


# Padding
print "Padding sequences..."
for idx, history in enumerate(training_encoder_input_data):
	training_encoder_input_data[idx] = history + [[0]*hidden_dimensions] * (max_sentence_forecast - len(history))

for idx, sentence in enumerate(training_decoder_input_data):
	training_decoder_input_data[idx] = sentence + [0] * (max_sentence_length - len(sentence))

for idx, sentence in enumerate(training_decoder_output_data):
	training_decoder_output_data[idx] = sentence + [0] * (max_sentence_length - len(sentence))

for idx, sentence in enumerate(training_decoder_original_output_data):
	training_decoder_original_output_data[idx] = sentence + [index_to_word["0"]] * (max_sentence_length - len(sentence))

for idx, sentence in enumerate(training_decoder_mask):
	training_decoder_mask[idx] = sentence + [0] * (max_sentence_length - len(sentence))



# Save the result of the preprocessing to be entered into the neural network
print "Save preprocessed data into files..."
if not os.path.exists(save_path):
    os.makedirs(save_path)

if not os.path.exists(save_path+"/encoder_input_data.txt"):
	with open(save_path+"/encoder_input_data.txt",'w') as f:
		json.dump(training_encoder_input_data, f)

if not os.path.exists(save_path+"/decoder_input_data.txt"):
	with open(save_path+"/decoder_input_data.txt",'w') as f:
		json.dump(training_decoder_input_data, f)

if not os.path.exists(save_path+"/decoder_output_data.txt"):
	with open(save_path+"/decoder_output_data.txt",'w') as f:
		json.dump(training_decoder_output_data, f)

if not os.path.exists(save_path+"/decoder_original_output_data.txt"):
	with open(save_path+"/decoder_original_output_data.txt",'w') as f:
		json.dump(training_decoder_original_output_data, f)

if not os.path.exists(save_path+"/encoder_input_length.txt"):
	with open(save_path+"/encoder_input_length.txt",'w') as f:
		json.dump(training_encoder_input_length, f)

if not os.path.exists(save_path+"/decoder_input_length.txt"):
	with open(save_path+"/decoder_input_length.txt",'w') as f:
		json.dump(training_decoder_input_length, f)

if not os.path.exists(save_path+"/decoder_mask.txt"):
	with open(save_path+"/decoder_mask.txt",'w') as f:
		json.dump(training_decoder_mask, f)

if not os.path.exists(save_path+"/word_to_index.txt"):
	with open(save_path+"/word_to_index.txt",'w') as f:
		json.dump(word_to_index, f)

if not os.path.exists(save_path+"/index_to_word.txt"):
	with open(save_path+"/index_to_word.txt",'w') as f:
		json.dump(index_to_word, f)

print "Preprocessing finished..."
