#Imports
##############################################
from collections import Counter
from nltk import word_tokenize, pos_tag
import gensim
import sys
import random
import os.path
import argparse
##############################################

#HELP definition
##############################################
'''
parser=argparse.ArgumentParser(
    description='Hello description',
    epilog='Enjoy!')
parser.add_argument('dummy', type=str, default='/', help='The path to the file')
args=parser.parse_args()
'''
##############################################


allTalksPath = "./../tedData/allTEDTalkContent.txt"
singleTalkPath = "./../tedData/Automated_change/Nick_Bostrom"

#Load TEDTalk Corpus to count words
f = open(allTalksPath,"r")
wordcount = Counter(f.read().split())
wordsInOrder = sorted(wordcount.items(), key=lambda item: item[1])

#Load the article to change
f  = open(singleTalkPath + "/original_text.txt", "r")
text = f.read()

#load Google Word2Vec Model
model=gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin',binary=True, limit=500000)

#find reoccurent words
history = 5

tokens = word_tokenize(text)
print tokens
changedTokens = word_tokenize(text)
nouns = []
positions = []

for position, token in enumerate(tokens):
	if pos_tag([token])[0][1] == 'NN':
		nouns.append(token)
		positions.append(position)

nounsWithPosition = zip(positions, nouns)

doubleValues = []
for position, (abs_Position, noun) in enumerate(nounsWithPosition):
	if position >= history:
		for checkNoun in nouns[position-history:position]:
			if noun == checkNoun:
				doubleValues.append(nounsWithPosition[position])
				continue

replace_Words =[]
counter = 0
while counter < 10:
	counter += 1

	tokenToChange = doubleValues[random.randint(0, len(doubleValues) - 1)]
	word = tokenToChange[1]
	substitute_Word = word
	
	#find the word in the list of sorted words
	wordIndex = -1
	frame = 20
	for index, item in enumerate(wordsInOrder): 
		if item[0] == word: 
			wordIndex = index
			break

	#get the word with the same word type and similar appearances in the defined frame
	nearestWords = []
	for item in wordsInOrder[wordIndex-frame:wordIndex+frame]: 
		wordType = pos_tag([item[0]])
		THE_WORD_Type = pos_tag([word])
		if wordType[0][1] == THE_WORD_Type[0][1]:
			nearestWords.append(wordType)

	
	#compare the words by the Word2Vec distance and take the best fit
	maxValue = 0
	maxItem = ''
	for item in nearestWords:
		try:
			score = model.similarity(word, item[0][0])
			if score > maxValue and score < 0.999:
				maxValue = score
				maxItem = item
		except Exception:
			maxValue = maxValue

	#Output dependend on Word2Vec representation
	if maxValue < 0.001:
		try:
			substitute_Word = nearestWords[random.randint(0, len(nearestWords) - 1)]
		except Exception:
			print 'Problem with random.choice'
	else:
		substitute_Word =  maxItem

	replace_Words.append(tuple((tokenToChange[0], substitute_Word[0][0])))

#STDOUT output for word changes
print ''
print ''
print ''
print '################'

for new_token in replace_Words:
	print "Replacing word "+tokens[new_token[0]]+" with "+new_token[1]
	changedTokens[new_token[0]] = "###"+new_token[1]+"###"

print '################'
print ''
print ''
print ''

#untokenizing the text
text = ' '.join(changedTokens).replace(' , ',',').replace(' .','.').replace(' !','!').replace(' ?','?').replace(' : ',': ').replace(' \'', '\'')

#saving the text into a new file
fileNumber = 1
while 1:
	if not os.path.isfile(singleTalkPath + "/modified_text_"+str(fileNumber)+".txt"):
		writeTo = open(singleTalkPath + "/modified_text_"+str(fileNumber)+".txt","w")
		writeTo.write(text)
		writeTo.close()
		break
	else:
		fileNumber += 1