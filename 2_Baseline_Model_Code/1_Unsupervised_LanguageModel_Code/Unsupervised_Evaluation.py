
# Commandline Arguments
##############################################
import optparse
parser = optparse.OptionParser()
parser.add_option('-i', '--input', action="store", dest="input", help="relative path to the dataset to test", default="")
parser.add_option('-u', '--unigram', action="store", dest="unigram", help="relative path to the dataset that was trained on", default="")
parser.add_option('-m', '--model', action="store", dest="model", help="relative path to the Model to test", default="")
parser.add_option('-v', '--vocab', action="store", dest="vocab", help="relative path to the vocabulary folder", default="")
parser.add_option('-f', '--file', action="store", dest="file", help="relative path to the vocabulary folder", default="")

options, args = parser.parse_args()
print options
input_path = options.input
training_path = options.unigram
model_path = options.model
vocab_path = options.vocab
fileName = options.file
##############################################



# Imports
##############################################
from keras.models import load_model
from collections import Counter
import numpy as np
import json
import math
import os
import numpy as np
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

unigrams = dict((word, count) for word, count in wordUnigram)

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
sentence_level_perplexity_filtered = []
words_not_in_vocab = 0
all_words_probability_with_modified = []

perplexity = np.float128(0.0)
perplexity_count = 0

for sentence in tokens:
	# For the start token
	perplexity_count += 1
	for word in sentence:
		perplexity_count += 1


# Iterate through all the sentences in the data and compute the perplexity
for index, sentence in enumerate(tokens):
	value_sentence = []

	# DEBUGGING output
#	if (index % 100) == 0:
#		print str(index) + " / " + str(len(tokens))

	# Adding start and end token to the sentence
	cut_sentence = ["<START>"] + sentence + ["<END>"]

	# Cutting sentences longer than 50 words
	if len(cut_sentence) > 50:
		cut_sentence = cut_sentence[:50]

	# Convert words into indexes and mark modified words
	changes = []
	for index, word in enumerate(cut_sentence):
		checkedWord = ""

		# Check if there is the "___" in the word --> the indicator for a modified word
		if word.find("___") == -1:
			checkedWord = word
		else:
			checkedWord = word[:word.find("___")]
			changes.append([(index-1), word[word.find("___")+3:]]) # Index-1 because the output that predicts the input predicts that one timestep in advance.
		
		# Convert word into the according index
		try:
			value_sentence.append(word_to_index[checkedWord])
		except Exception:
			value_sentence.append(word_to_index["<UNKWN>"])

	# Create input and output vectors
	input_perplexity = value_sentence[:len(cut_sentence)-1]
	output_perplexity = value_sentence[1:]
	length = len(input_perplexity)

	# SAve input vector in numpy array
	testInput = np.zeros((1, length), dtype=np.int16)
	for index, idx in enumerate(input_perplexity):
		testInput[0, index] = idx

	# Predict the next words
	prediction = model.predict(testInput, verbose=0)[0]

	# Evaluate the prediction
	for index, word in enumerate(prediction):

		# Add the probability of the correct output word to thw array (! This will be overwritten if the word is a modified one !)
		count_o = 1
		try:
			count_o = unigrams[index_to_word[str(output_perplexity[index])]]
		except Exception:
			count_o = count_o

		all_words_probability_with_modified.append([word[output_perplexity[index]], 0, (word[output_perplexity[index]]*len(words))/(count_o), index_to_word[str(output_perplexity[index])]])

		#check if word got replaced and if so, save in special array
		for change in changes:
			if change[0] == index:
				try:	

					# Get rank of original and modified word			
					better_alternatives_original = 1
					better_alternatives_modified = 1

					for selection in word:
						if selection > word[word_to_index[change[1]]]:
							better_alternatives_modified += 1
						if selection > word[output_perplexity[index]]:
							better_alternatives_original += 1

					#print "word "+ str(change[1]) + " has rank " + str(better_alternatives_modified) + " with probability " + str(word[word_to_index[change[1]]])

					# Get unigram probability
					count_o = 1
					count_m = 1
					try:
						count_o = unigrams[index_to_word[str(output_perplexity[index])]]
						count_m = unigrams[change[1]]
					except Exception:
						count_o = count_o

					# Substitude the original word with the modified one -> try to find these later
					all_words_probability_with_modified[len(all_words_probability_with_modified)-1][0] = word[word_to_index[change[1]]]
					all_words_probability_with_modified[len(all_words_probability_with_modified)-1][1] = 1
					all_words_probability_with_modified[len(all_words_probability_with_modified)-1][2] = (word[word_to_index[change[1]]]*len(words))/(count_m)
					all_words_probability_with_modified[len(all_words_probability_with_modified)-1][3] = change[1]
					#print all_words_probability_with_modified[len(all_words_probability_with_modified)-1]

					# Save analytics for changed words in list
					probability_orig_mods.append([[word[output_perplexity[index]], (word[output_perplexity[index]]*len(words))/(count_o), better_alternatives_original], [word[word_to_index[change[1]]], (word[word_to_index[change[1]]]*len(words))/(count_m), better_alternatives_modified]])

				except Exception:
					words_not_in_vocab += 1

		print "word probability:"
		print word[output_perplexity[index]]
		print "latest perplexity:"
		print perplexity
		print type(perplexity)
		if word[output_perplexity[index]] > 0.0:
			perplexity += (math.log(word[output_perplexity[index]], 2))
		#print "perplexity calculation adds to perplexity variable :"
		#print "math.log(" + str(word[output_perplexity[index]]) + ", 2) = " + str((math.log(word[output_perplexity[index]], 2)))

# Calculate mean perplexity for the sentence
#print "perplexity of the sentence is calculated by :"
#print str(-perplexity) + " / " + str(perplexity_count) + " = " + str(-perplexity/perplexity_count)
perplexity = -perplexity/perplexity_count
sentence_level_perplexity.append(int(2**(perplexity)))
#print "final perplexity by 2^pp(w): " + str(int(2**(perplexity)))


# Store the mean values of the replaced words
mean_p_o = 0
mean_p_m = 0
mean_p_o_u = 0
mean_p_m_u = 0
mean_position_o = 0
mean_position_m = 0

classes = 4
percentage = 100/classes
classCount_o = [0]*(classes)
classCount_m = [0]*(classes)

for value in probability_orig_mods:
	mean_p_o += value[0][0]
	mean_p_m += value[1][0]
	mean_p_o_u += value[0][1]
	mean_p_m_u += value[1][1]
	mean_position_o += value[0][2]
	mean_position_m += value[1][2]

	# Divide and sum rank in classes
	counter_Percent = 100 - percentage
	counter = 0
	while counter_Percent > -1:
		if (float(len(word_to_index) - value[0][2]) / float(len(word_to_index)))*100 >= counter_Percent and (float(len(word_to_index) - value[0][2]) / float(len(word_to_index)))*100 < counter_Percent + percentage:
			classCount_o[counter] += 1

		if (float(len(word_to_index) - value[1][2]) / float(len(word_to_index)))*100 >= counter_Percent and (float(len(word_to_index) - value[1][2]) / float(len(word_to_index)))*100 < counter_Percent + percentage:
			classCount_m[counter] += 1
		counter_Percent -= percentage
		counter += 1

# Divide summed up values to get the mean
mean_p_o /= len(probability_orig_mods)-words_not_in_vocab
mean_p_m /= len(probability_orig_mods)-words_not_in_vocab
mean_p_o_u /= len(probability_orig_mods)-words_not_in_vocab
mean_p_m_u /= len(probability_orig_mods)-words_not_in_vocab
mean_position_o /= len(probability_orig_mods)-words_not_in_vocab
mean_position_m /= len(probability_orig_mods)-words_not_in_vocab

#Sort all the probabilities to find modified words
all_words_probability_with_modified.sort(key=lambda row: row[0])
all_positions_modified = []
for idx, element in enumerate(all_words_probability_with_modified):
	if element[1] == 1:
		all_positions_modified.append([idx, element[3], element[0]])
print "Number of words modified and in all_words_probability_with_modified: "+str(len(all_positions_modified))
modifications_in_lowest_100 = 0
modifications_in_lowest_500 = 0
modifications_in_lowest_1000 = 0
modifications_in_lowest_4000 = 0
modifications_in_lowest_10_percent = 0
modifications_in_lowest_20_percent = 0
modifications_in_lowest_30_percent = 0

for index, prob in enumerate(all_words_probability_with_modified):
	if index <= 100:
		if prob[1] == 1:
			modifications_in_lowest_100 += 1
	if index <= 500:
		if prob[1] == 1:
			modifications_in_lowest_500 += 1
	if index <= 1000:
		if prob[1] == 1:
			modifications_in_lowest_1000 += 1
	if index <= 4000:
		if prob[1] == 1:
			modifications_in_lowest_4000 += 1

for index, prob in enumerate(all_words_probability_with_modified[:(len(all_words_probability_with_modified)/100*30)]):
	if index <= (len(all_words_probability_with_modified)/100*10):
		if prob[1] == 1:
			modifications_in_lowest_10_percent += 1
	if index <= (len(all_words_probability_with_modified)/100*20):
		if prob[1] == 1:
			modifications_in_lowest_20_percent += 1
	if index <= (len(all_words_probability_with_modified)/100*30):
		if prob[1] == 1:
			modifications_in_lowest_30_percent += 1

all_words_probability_with_modified.sort(key=lambda row: row[2])
unigram_modifications_in_lowest_100 = 0
unigram_modifications_in_lowest_500 = 0
unigram_modifications_in_lowest_1000 = 0
unigram_modifications_in_lowest_4000 = 0
unigram_modifications_in_lowest_10_percent = 0
unigram_modifications_in_lowest_20_percent = 0
unigram_modifications_in_lowest_30_percent = 0

for index, prob in enumerate(all_words_probability_with_modified):
	if index <= 100:
		if prob[1] == 1:
			unigram_modifications_in_lowest_100 += 1
	if index <= 500:
		if prob[1] == 1:
			unigram_modifications_in_lowest_500 += 1
	if index <= 1000:
		if prob[1] == 1:
			unigram_modifications_in_lowest_1000 += 1
	if index <= 4000:
		if prob[1] == 1:
			unigram_modifications_in_lowest_4000 += 1

for index, prob in enumerate(all_words_probability_with_modified[:(len(all_words_probability_with_modified)/100*30)]):
	if index <= (len(all_words_probability_with_modified)/100*10):
		if prob[1] == 1:
			unigram_modifications_in_lowest_10_percent += 1
	if index <= (len(all_words_probability_with_modified)/100*20):
		if prob[1] == 1:
			unigram_modifications_in_lowest_20_percent += 1
	if index <= (len(all_words_probability_with_modified)/100*30):
		if prob[1] == 1:
			unigram_modifications_in_lowest_30_percent += 1

# Output the results
print ""
print "Words not in vocabulary: " + str(words_not_in_vocab)
print "Mean Probability of correct Words: " + str(mean_p_o)
print "Mean Probability of modified Words: " + str(mean_p_m)
print "Mean Probability of correct Words divided by unigram probability: " + str(mean_p_o_u)
print "Mean Probability of modified Words divided by unigram probability: " + str(mean_p_m_u)
print "Mean Position of the correct word compared to the other alternatives is: " + str(mean_position_o) + " / " + str(len(word_to_index))
print "Mean Position of the modified word compared to the other alternatives is: " + str(mean_position_m) + " / " + str(len(word_to_index))
print "Modified words within most unlikely 100 words on dataset: " + str(modifications_in_lowest_100)
print "Modified words within most unlikely 500 words on dataset: " + str(modifications_in_lowest_500)
print "Modified words within most unlikely 1000 words on dataset: " + str(modifications_in_lowest_1000)
print "Modified words within most unlikely 4000 words on dataset: " + str(modifications_in_lowest_4000)
print "Modified words within most unlikely 10 percent on dataset (" + str(len(all_words_probability_with_modified)/100*10) + "): " + str(modifications_in_lowest_10_percent)
print "Modified words within most unlikely 20 percent on dataset (" + str(len(all_words_probability_with_modified)/100*20) + "): " + str(modifications_in_lowest_20_percent)
print "Modified words within most unlikely 30 percent on dataset (" + str(len(all_words_probability_with_modified)/100*30) + "): " + str(modifications_in_lowest_30_percent)
print "Modified words/unigram probability within most unlikely 100 words on dataset: " + str(unigram_modifications_in_lowest_100)
print "Modified words/unigram probability within most unlikely 500 words on dataset: " + str(unigram_modifications_in_lowest_500)
print "Modified words/unigram probability within most unlikely 1000 words on dataset: " + str(unigram_modifications_in_lowest_1000)
print "Modified words/unigram probability within most unlikely 4000 words on dataset: " + str(unigram_modifications_in_lowest_4000)
print "Modified words/unigram probability within most unlikely 10 percent on dataset (" + str(len(all_words_probability_with_modified)/100*10) + "): " + str(unigram_modifications_in_lowest_10_percent)
print "Modified words/unigram probability within most unlikely 20 percent on dataset (" + str(len(all_words_probability_with_modified)/100*20) + "): " + str(unigram_modifications_in_lowest_20_percent)
print "Modified words/unigram probability within most unlikely 30 percent on dataset (" + str(len(all_words_probability_with_modified)/100*30) + "): " + str(unigram_modifications_in_lowest_30_percent)
print "All words: "+str(len(all_words_probability_with_modified))
print "All modified words with positions: "
print all_positions_modified
print "Distribution of correct word relative to other words in " + str(classes) + " classes: "
print classCount_o
print "Distribution of modified word relative to other words in " + str(classes) + " classes: "
print classCount_m
print "Perplexity of all Sequences: " + str(sentence_level_perplexity)
print ""

with open("./perplexity_unsupervised_changed_calcun.txt", "a") as f:
	content = fileName
	#content += "; Mean Probability of modified Words, " + str(mean_p_m)
	#content += "; Mean Probability of correct Words divided by unigram probability, " + str(mean_p_o_u)
	#content += "; Mean Probability of modified Words divided by unigram probability, " + str(mean_p_m_u)
	#content += "; Mean Position of the correct word compared to the other alternatives is, " + str(mean_position_o)
	#content += "; Mean Position of the modified word compared to the other alternatives is, " + str(mean_position_m)
	content += "; Mean Perplexity of all Sequences, " + str(sentence_level_perplexity)
	#content += "; Modified words within most unlikely 100 words on dataset, " + str(modifications_in_lowest_100)
	#content += "; Modified words within most unlikely 500 words on dataset, " + str(modifications_in_lowest_500)
	#content += "; Modified words within most unlikely 1000 words on dataset, " + str(modifications_in_lowest_1000)
	content += "; Modified words within most unlikely 4000 words on dataset, " + str(modifications_in_lowest_4000)
	#content += "; Modified words within most unlikely 10 percent on dataset (" + str(len(all_words_probability_with_modified)/100*10) + "), " + str(modifications_in_lowest_10_percent)
	#content += "; Modified words within most unlikely 20 percent on dataset (" + str(len(all_words_probability_with_modified)/100*20) + "), " + str(modifications_in_lowest_20_percent)
	#content += "; Modified words within most unlikely 30 percent on dataset (" + str(len(all_words_probability_with_modified)/100*30) + "), " + str(modifications_in_lowest_30_percent)
	#content += "; Modified words/unigram probability within most unlikely 100 words on dataset, " + str(unigram_modifications_in_lowest_100)
	#content += "; Modified words/unigram probability within most unlikely 500 words on dataset, " + str(unigram_modifications_in_lowest_500)
	#content += "; Modified words/unigram probability within most unlikely 1000 words on dataset, " + str(unigram_modifications_in_lowest_1000)
	content += "; Modified words/unigram probability within most unlikely 4000 words on dataset, " + str(unigram_modifications_in_lowest_4000)
	#content += "; Modified words/unigram probability within most unlikely 10 percent on dataset (" + str(len(all_words_probability_with_modified)/100*10) + "), " + str(unigram_modifications_in_lowest_10_percent)
	#content += "; Modified words/unigram probability within most unlikely 20 percent on dataset (" + str(len(all_words_probability_with_modified)/100*20) + "), " + str(unigram_modifications_in_lowest_20_percent)
	#content += "; Modified words/unigram probability within most unlikely 30 percent on dataset (" + str(len(all_words_probability_with_modified)/100*30) + "), " + str(unigram_modifications_in_lowest_30_percent)
	#content += "; All modified words with positions: "
	#for element in all_positions_modified:
#		content += "; " + str(element)

#	content += "; Distribution of correct word relative to other words"
#	for element in classCount_o:
#		content += "; " + str(element)
#	content += "; Distribution of modified word relative to other words"
#	for element in classCount_m:
#		content += "; " + str(element)

	f.write("{}\n".format(content))






