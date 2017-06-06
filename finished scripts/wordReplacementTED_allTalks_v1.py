#Imports
##############################################
from collections import Counter
from nltk import word_tokenize
import gensim
import sys
import random
import os.path
import argparse
import xml.etree.ElementTree as ET
import json
import re
import string
from nltk.tag.stanford import StanfordPOSTagger
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
path_to_model = './../lib/stanford-postagger/models/english-bidirectional-distsim.tagger'
path_to_jar = './../lib/stanford-postagger/stanford-postagger.jar'
mainPath = "./../tedData"
savePath = "./../tedData/Automated_change_All_Files"
allTokens = [] # To be load from all TED talks
history = 10 # The max distance between two referenced that will be replaced
FRAME = 10 # The distance of two words regarding their reoccurences on the dataset
COUNT_REPLACE = 15 # How many reoccurances should be replaced
##############################################



#Helper function
##############################################
def untokenize(words):
    text = ' '.join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
         "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()
##############################################



#Initialize tagger
tagger = StanfordPOSTagger(path_to_model, path_to_jar)

#txt = "The quick brown fox jumps over the lazy dog"
#print tagger.tag(txt.split())

#Check if Corpus is already tokenized --> if not, save the tokens in a new file to save time on restart
if not os.path.exists(mainPath+"/allTEDTalkContent_Tokenized.txt"):
	f = open(mainPath+"/allTEDTalkContent.txt","r")
	allTokens = word_tokenize(f.read().decode('unicode-escape'))
	with open(mainPath+"/allTEDTalkContent_Tokenized.txt",'w') as f:
		json.dump(allTokens, f)

#Load TEDTalk Corpus to count words
with open (mainPath+"/allTEDTalkContent_Tokenized.txt", 'r') as f:
    allTokens = json.load(f)
wordcount = Counter(allTokens)
wordsInOrder = sorted(wordcount.items(), key=lambda item: item[1])


taggedWordsInOrder = []
for (word, count) in wordsInOrder:
	print word
	taggedWordsInOrder.append(tagger.tag([word]))

with open(mainPath+"/allTEDTalkContent_Tokenized_2.txt",'w') as f:
		json.dump(taggedWordsInOrder, f)
		print "saved"



#load the XML
tree = ET.parse(mainPath+"/ted.xml")
root = tree.getroot()
for child in root:
	ident = child.find('head').find('talkid').text.encode('utf-8')
	speaker = child.find('head').find('speaker').text.encode('utf-8')
	folder = savePath+"/"+ident+"-"+speaker
	text = child.find('content').text.encode('utf-8')
	
	if os.path.exists(folder):
		print "Folder "+folder+" already exists"
	
	else:
		#Tokenize text
		tokens = word_tokenize(text.decode('unicode-escape'))
		changedTokens = list(tokens)
		changedTokensIndicated = list(tokens)
		nouns = []
		positions = []

		if len(tokens) < 200:
			print "The TEDTalk "+folder+" ist shorter than 200 words! Nothing will be created"

		else:
			os.makedirs(folder)
			print "create data in Folder "+folder

			g = open(folder+"/original_text.txt","w")
			g.write(text)
			g.close()

			tagged_Tokens = tagger.tag(tokens)
			for position, (word, token) in enumerate(tagged_Tokens):
				if token.startswith('N'):
					print "Found noun: "+word
					nouns.append(word)
					positions.append(position)

			nounsWithPosition = zip(positions, nouns)

			reoccurendValues = []
			for position, (abs_Position, noun) in enumerate(nounsWithPosition):
				if position >= history:
					for checkNoun in nouns[position-history:position]:
						if noun == checkNoun:
							reoccurendValues.append(nounsWithPosition[position])
							continue

			replace_Words =[]
			replaceTracker = []
			counter = 0
			while counter < COUNT_REPLACE:
				counter += 1
				print counter
				if len(reoccurendValues) > 0:
					tokenToChange = reoccurendValues[random.randint(0, len(reoccurendValues) - 1)]
					print "tokenToChange: "+tokenToChange
					word = tokenToChange[1]
					substitute_Word = word
					print "word: "+word
					
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
					while found < 3:
						for item in taggedWordsInOrder[wordIndex-frame:wordIndex+frame]: 
							wordType = tagger.tag([item[0]])
							if wordType[0][1] == tokenToChange[1] and wordType[0][0] != tokenToChange[1]:
								nearestWords.append(wordType)
								found = 3
						frame *= 2
						found += 1

					for item in nearestWords:
						try:
							substitute_Word = nearestWords[random.randint(0, len(nearestWords) - 1)]
							for char in substitute_Word:
								if char in string.punctuation:
									substitute_Word = word
						except Exception:
							print 'Problem with random.choice'
					replace_Words.append(tuple((tokenToChange[0], substitute_Word[0][0])))


			# Replace the selected words in the texts and save the replacement for the changeLog
			for new_token in replace_Words:
				changedTokens[new_token[0]] = new_token[1]
				changedTokensIndicated[new_token[0]] = "###"+new_token[1]+"###"
				replaceTracker.append([new_token[0], tokens[new_token[0]], new_token[1]])

			# Untokenizing the text
			text = untokenize(changedTokens)
			textIndicated = untokenize(changedTokensIndicated)

			# Saving the modified text into new file
			writeTo = open(folder+"/modified_text.txt","w")
			writeTo.write(text.encode('utf-8'))
			writeTo.close()

			# Saving the indicated text into new file
			writeToIndicated = open(folder+"/indicated_text.txt","w")
			writeToIndicated.write(textIndicated.encode('utf-8'))
			writeToIndicated.close()
			
			# Save the changed made into new file
			with open(folder+"/changes_text.txt","w") as f:
				json.dump(replaceTracker, f)


