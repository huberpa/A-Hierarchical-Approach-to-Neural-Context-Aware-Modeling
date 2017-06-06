#Imports
##############################################
from collections import Counter
from nltk import word_tokenize, pos_tag
import gensim
import sys
import random
##############################################

#get the commandline argument
THE_WORD = sys.argv[1]
frame = int(sys.argv[2])

#Variables
wordIndex = 0
THE_WORD_Type = pos_tag([THE_WORD])[0][1]

#load Google Word2Vec Model
model=gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin',binary=True, limit=500000)

#Load TEDTalk Corpus to count words
f = open("./../tedData/allTEDTalkContent.txt","r")
wordcount = Counter(f.read().split())
wordsInOrder = sorted(wordcount.items(), key=lambda item: item[1])

#find the word in the list of sorted words
for index, item in enumerate(wordsInOrder): 
	if item[0] == THE_WORD: 
		wordIndex = index

#get the word with the same word type and similar appearances in the defined frame
nearestWords = []
for item in wordsInOrder[wordIndex-frame:wordIndex+frame]: 
	wordType = pos_tag([item[0]])
	if wordType[0][1] == THE_WORD_Type:
		nearestWords.append(wordType)

#compare the words by the Word2Vec distance and take the best fit
maxValue = 0
maxItem = ''
for item in nearestWords:
	try:
		score = model.similarity(THE_WORD, item[0][0])
		if score > maxValue and score < 0.999:
			maxValue = score
			maxItem = item
	except Exception:
		print 'Word not found :('

#Output dependend on Word2Vec representation
if maxValue < 0.001:
	print 'No word found in Word2Vec'
	print 'Coosing random word'
	print random.choice(nearestWords)
else:
	print 'Closest word found in Word2Vec'
	print 'With similarity Index: '+ str(maxValue)
	print maxItem
