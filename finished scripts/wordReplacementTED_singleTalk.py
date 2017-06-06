#Imports
##############################################
from collections import Counter
from nltk import word_tokenize, pos_tag
import gensim
import sys
import random
import os.path
import argparse
import pickle
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



#Variable definition
##############################################
allTalksPath = "./../tedData/allTEDTalkContent.txt"
allTalksTokens = "./../tedData/allTEDTalkContent_Tokenized.txt"
singleTalkPath = "./../tedData/Automated_change/Nick_Bostrom"
allTokens = [] # To be load from all TED talks
history = 5 # The max distance between two referenced that will be replaced
FRAME = 10 # The distance of two words regarding their reoccurences on the dataset
##############################################



#Check if Corpus is already tokenized --> if not, save the tokens in a new file to save time on restart
if not os.path.exists(allTalksTokens):
	f = open(allTalksPath,"r")
	allTokens = word_tokenize(f.read().decode('unicode-escape'))
	with open(allTalksTokens,'w') as f:
		pickle.dump(allTokens, f)

#Load TEDTalk Corpus to count words
with open (allTalksTokens, 'r') as f:
    allTokens = pickle.load(f)
wordcount = Counter(allTokens)
wordsInOrder = sorted(wordcount.items(), key=lambda item: item[1])

#Load the article to change
g  = open(singleTalkPath + "/original_text.txt", "r")

#find reoccurent words

tokens = word_tokenize(g.read().decode('unicode-escape'))
changedTokens = list(tokens)
nouns = []
positions = []

for position, token in enumerate(tokens):
	if pos_tag([token])[0][1].startswith('N'):
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
	frame = FRAME
	for index, item in enumerate(wordsInOrder): 
		if item[0] == word: 
			wordIndex = index
			break

	#get the word with the same word type and similar appearances in the defined frame
	nearestWords = []
	found = 0
	while found < 10:
		for item in wordsInOrder[wordIndex-frame:wordIndex+frame]: 
			wordType = pos_tag([item[0]])
			THE_WORD_Type = pos_tag([word])
			if wordType[0][1] == THE_WORD_Type[0][1] and wordType[0][0] != THE_WORD_Type[0][0]:
				nearestWords.append(wordType)
				found = 10
		frame *= 2
		found += 1

	
	#compare the words by the Word2Vec distance and take the best fit
	maxValue = 0
	maxItem = ''
	for item in nearestWords:
		try:
			substitute_Word = nearestWords[random.randint(0, len(nearestWords) - 1)]
		except Exception:
			print 'Problem with random.choice'
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