
# Commandline Arguments
##############################################
import optparse
parser = optparse.OptionParser()
parser.add_option('-i', '--input', action="store", dest="input", help="relative path to the dataset to test", default="")
parser.add_option('-u', '--unigram', action="store", dest="unigram", help="relative path to the dataset that was trained on", default="")
parser.add_option('-m', '--model', action="store", dest="model", help="relative path to the Model to test", default="")
parser.add_option('-v', '--vocab', action="store", dest="vocab", help="relative path to the vocabulary folder", default="")
options, args = parser.parse_args()
print options
input_path = options.input
training_path = options.unigram
model_path = options.model
vocab_path = options.vocab
##############################################



# Imports
##############################################
from keras.models import load_model
from collections import Counter
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

# Load the file with the talks and tokenize
path  = open(input_path, "r")
text = path.read().decode('utf8')
tokens = []
for index, sentence in enumerate(nltk.sent_tokenize(text)): 
    tokens.append(nltk.word_tokenize(sentence))
tokens = [[word.lower() for word in sentence] for sentence in tokens]


# Unigram probabilities from training
unigram_path  = open(training_path, "r")
unigram_text = unigram_path.read().decode('utf8')
words = nltk.word_tokenize(unigram_text)
words = [word.lower() for word in words]
wordcount = Counter(words)
wordUnigram = sorted(wordcount.items(), key=lambda item: item[1])


# Load the index_to_word and word_to_index files to convert indizes into words
word_to_index = []
with open(vocab_path + "/word_to_index.json") as f:    
	word_to_index = json.load(f)

index_to_word = []
with open(vocab_path + "/index_to_word.json") as f:    
	index_to_word = json.load(f)

# Compute the perplexity for each sentence
probability_orig_mods = []
probability_with_unigram = []
sentence_level_perplexity = []
max_val = 50000
words_not_in_vocab = 0
# Iterate through all the sentences in the data and compute the perplexity
for index, sentence in enumerate(tokens[:max_val]):
	value_sentence = []

	print str(index) + " / " + str(max_val)

	cut_sentence = ["<START>"] + sentence + ["<END>"]
	if len(sentence) > 50:
		cut_sentence = sentence[:50]

	changes = []
	for index, word in enumerate(cut_sentence):

		#check if changed word --> if so cut and only use original, save index of word and changed value to compute later
		checkedWord = ""
		if word.find("___") == -1:
			checkedWord = word
		else:
			checkedWord = word[:word.find("___")]
			changes.append([(index-1), word[word.find("___")+3:]]) # Index-1 because the output that predicts the input predicts that one timestep in advance.
		try:
			value_sentence.append(word_to_index[checkedWord])
		except Exception:
			value_sentence.append(word_to_index["<UNKWN>"])

	input_perplexity = value_sentence[:len(cut_sentence)-1]
	output_perplexity = value_sentence[1:]
	length = len(input_perplexity)

	testInput = np.zeros((1, length), dtype=np.int16)
	for index, idx in enumerate(input_perplexity):
		testInput[0, index] = idx

	prediction = model.predict(testInput, verbose=0)[0]

	perplexity = 0.
	for index, word in enumerate(prediction):

		#check if word got replaced and if so, save in special array
		for change in changes:
			if change[0] == index:
				try:				
					better_alternatives_original = 1
					better_alternatives_modified = 1

					for selection in word:
						if selection > word[word_to_index[change[1]]]:
							better_alternatives_modified += 1
						if selection > word[output_perplexity[index]]:
							better_alternatives_original += 1

					count_o = 1
					count_m = 1
					for unigram in wordUnigram:
						if index_to_word[str(output_perplexity[index])] == unigram[0]:
							count_o = unigram[1]
						if change[1] == unigram[0]:
							count_m = unigram[1]

					probability_orig_mods.append([[word[output_perplexity[index]], (word[output_perplexity[index]]*len(words))/(count_o), better_alternatives_original], [word[word_to_index[change[1]]], (word[word_to_index[change[1]]]*len(words))/(count_m), better_alternatives_modified]])

				except Exception:
					print "Word not in vocabulary..."
					words_not_in_vocab += 1

		# Perplexity for overall perplexity computaion
		perplexity += (math.log(word[output_perplexity[index]], 2))/length

	# Calculate mean perplexity for the sentence
	perplexity = -perplexity
	sentence_level_perplexity.append(int(2**(perplexity)))

	# DEBUGGING
	if (2**(perplexity)) > 10000:
		print 2**(perplexity)
		print cut_sentence


# Store the mean values of the replaced words
mean_p_o = 0
mean_p_m = 0
mean_p_o_u = 0
mean_p_m_u = 0
mean_position_o = 0
mean_position_m = 0

classes = 4
percentage = 100/classes

classCount_o = [0]*classes
classCount_m = [0]*classes

for value in probability_orig_mods:
	mean_p_o += value[0][0]
	mean_p_m += value[1][0]
	mean_p_o_u += value[0][1]
	mean_p_m_u += value[1][1]
	mean_position_o += value[0][2]
	mean_position_m += value[1][2]

	counter_Percent = 100
	counter = 0
	while counter_Percent > -1:
		if ((len(word_to_index) - value[0][2]) / len(word_to_index))*100 >= counter_Percent:
			classCount_o[counter] += 1

		if ((len(word_to_index) - value[1][2]) / len(word_to_index))*100 >= counter_Percent:
			classCount_m[counter] += 1

		counter_Percent -= percentage
		counter += 1

mean_p_o /= len(probability_orig_mods)-words_not_in_vocab
mean_p_m /= len(probability_orig_mods)-words_not_in_vocab
mean_p_o_u /= len(probability_orig_mods)-words_not_in_vocab
mean_p_m_u /= len(probability_orig_mods)-words_not_in_vocab
mean_position_o /= len(probability_orig_mods)-words_not_in_vocab
mean_position_m /= len(probability_orig_mods)-words_not_in_vocab

# Output the results
print ""
print "Probabilities of changed words compared to the originals ( originals first --> [p(W/h), p(W/h)/p(W)] ): " 
print probability_orig_mods
print ""
print "Words not in vocabulary: " + str(words_not_in_vocab)
print "Mean Probability of correct Words: " + str(mean_p_o)
print "Mean Probability of modified Words: " + str(mean_p_m)
print "Mean Probability of correct Words divided by unigram probability: " + str(mean_p_o_u)
print "Mean Probability of modified Words divided by unigram probability: " + str(mean_p_m_u)
print "Mean Position of the correct word compared to the other alternatives is: " + str(mean_position_o) + " / " + str(len(word_to_index))
print "Mean Position of the modified word compared to the other alternatives is: " + str(mean_position_m) + " / " + str(len(word_to_index))
print "Distribution of correct word relative to other words in " + str(classes) + " classes: "
print classCount_o
print "Distribution of modified word relative to other words in " + str(classes) + " classes: "
print classCount_m
print "Mean Perplexity of all Sequences: " + str(np.mean(sentence_level_perplexity))
print "Mean Perplexity of all Sequences: " + str(np.mean(sentence_level_perplexity))
print ""


# Save the results in a file
with open("./perplexity.txt", "w") as f:
	f.write("{}\n".format(" "))
	f.write("{}\n".format(probability_orig_mods))
	f.write("{}\n".format(" "))
	f.write("{}\n".format(str(mean_p_o)))
	f.write("{}\n".format(str(mean_p_m)))
	f.write("{}\n".format(str(mean_p_o_u)))
	f.write("{}\n".format(str(mean_p_m_u)))
	f.write("{}\n".format(str(mean_position_o)))
	f.write("{}\n".format(str(mean_position_m)))
	f.write("{}\n".format(str(np.mean(sentence_level_perplexity))))
	f.write("{}\n".format(" "))
	f.write("{}\n".format("*"*50))


