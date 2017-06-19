
# Commandline Arguments
##############################################
import optparse
parser = optparse.OptionParser()
parser.add_option('-i', '--input', action="store", dest="input", help="relative path to the dataset to test", default="")
parser.add_option('-t', '--test', action="store", dest="test", help="relative path to the modified dataset", default="")
parser.add_option('-m', '--model', action="store", dest="model", help="relative path to the Model to test", default="")
parser.add_option('-v', '--vocab', action="store", dest="vocab", help="relative path to the vocabulary folder", default="")
options, args = parser.parse_args()
input_path = options.input
model_path = options.model
vocab_path = options.vocab
test_path = options.test
##############################################



# Imports
##############################################
from keras.models import load_model
import numpy as np
import json
import math
import nltk
##############################################



# Main
##############################################

# Load the model to be tested against and output the structure
model = load_model(model_path)
print model.summary() 

# Load the files with the talks (original and modified) and tokenize them
path_original  = open(input_path, "r")
text_original = path_original.read().decode('utf8')

path_modified  = open(test_path, "r")
text_modified = path_modified.read().decode('utf8')

original_tokens = []
for index, sentence in enumerate(nltk.sent_tokenize(text_original)): 
    original_tokens.append(nltk.word_tokenize(sentence))

modified_tokens = []
for index, sentence in enumerate(nltk.sent_tokenize(text_modified)): 
    modified_tokens.append(nltk.word_tokenize(sentence))

original_tokens = [[word.lower() for word in sentence] for sentence in original_tokens]
modified_tokens = [[word.lower() for word in sentence] for sentence in modified_tokens]

proofTokens = []
for index, sentence in enumerate(original_tokens): 
	print original_tokens[index]
	print modified_tokens[index]
	print ('-'*50)
	print ""
# Load the index_to_word and word_to_index files to convert indizes into words


'''

word_to_index = []
with open(vocab_path + "/word_to_index.json") as f:    
	word_to_index = json.load(f)

index_to_word = []
with open(vocab_path + "/index_to_word.json") as f:    
	index_to_word = json.load(f)

# Compute the perplexity for each sentence
corpus_level_perplexity = 1.
sentence_level_perplexity = []
max_val = 50000

# Iterate through all the sentences in the data and compute the perplexity
for index, sentence in enumerate(original_tokens[:max_val]):
	interchanged_words = []
	value_sentence = []
	cut_sentence = ["<START>"] + sentence + ["<END>"]
	if len(sentence) > 50:
		cut_sentence = sentence[:50]
	for word in cut_sentence:
		try:
			value_sentence.append(word_to_index[word])
		except Exception:
			value_sentence.append(word_to_index["<UNKWN>"])

	input_perplexity = value_sentence[:len(cut_sentence)-1]
	output_perplexity = value_sentence[1:]
	length = len(input_perplexity)

	testInput = np.zeros((1, length), dtype=np.int16)
	for index, idx in enumerate(input_perplexity):
		testInput[0, index] = idx

	prediction = model.predict(testInput, verbose=0)[0]

	all_per = 0.
	for index, word in enumerate(prediction):
		all_per += (math.log(word[output_perplexity[index]], 2))/length
	all_per = -all_per
	if (2**(all_per)) > 10000:
		print 2**(all_per)
		print cut_sentence
	sentence_level_perplexity.append(int(2**(all_per)))

corpus_level_perplexity = np.mean(sentence_perplexity)
print ""
print ""
print "Sequence-wise perplexity: "
print sentence_perplexity
print ""
print ""
print "Max value: " + str(np.amax(sentence_perplexity))
print "Min value: " + str(np.amin(sentence_perplexity))
print ""
print ""
print "Mean Perplexity of all Sequences: " + str(corpus_level_perplexity)
print ""
print ""

with open("./perplexity.txt", "w") as f:
	f.write("{}\n".format(sentence_perplexity))
	f.write("{}\n".format(" "))
	f.write("{}\n".format(mean_perplexity_all))
	f.write("{}\n".format("*"*50))
'''

