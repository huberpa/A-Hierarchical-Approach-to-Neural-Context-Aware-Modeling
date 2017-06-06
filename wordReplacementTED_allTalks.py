
# Imports
##############################################
from collections import Counter
from nltk import word_tokenize
import sys
import random
import os.path
import argparse
import xml.etree.ElementTree as ET
import json
import re
import string
import time
from nltk.tag.stanford import StanfordPOSTagger
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
##############################################



# Variable definition
##############################################
path_to_model = './../lib/stanford-postagger/models/english-bidirectional-distsim.tagger'
path_to_jar = './../lib/stanford-postagger/stanford-postagger.jar'
mainPath = "./../tedData"
savePath = "./../tedData/Automated_change_All_Files_v3"
allTokens = [] # To be load from all TED talks
history = 10 # The max distance between two referenced that will be replaced
FRAME = 20 # The distance of two words regarding their reoccurences on the dataset
COUNT_REPLACE = 10 # How many reoccurances should be replaced
##############################################



# Helper function for text untokenizing
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



# Initialize tagger
tagger = StanfordPOSTagger(path_to_model, path_to_jar, java_options='-mx1000m -Xmx1028m')

# Check if Corpus is already tokenized --> if not, save the tokens in a new file to save time on restart
if not os.path.exists(mainPath+"/allTEDTalkContent_Tokenized.txt"):
	f = open(mainPath+"/allTEDTalkContent.txt","r")
	allTokens = word_tokenize(f.read().decode('utf-8'))
	with open(mainPath+"/allTEDTalkContent_Tokenized.txt",'w') as f:
		json.dump(allTokens, f)

# Load TEDTalk Corpus to count words
with open (mainPath+"/allTEDTalkContent_Tokenized.txt", 'r') as f:
    allTokens = json.load(f)
wordcount = Counter(allTokens)
wordsInOrder = sorted(wordcount.items(), key=lambda item: item[1])

'''
# Tag all words 
taggedWordsInOrder = []
tmp_words = []
rangeStep = 20
for number in range(0, len(wordsInOrder), rangeStep):
	for (word, count) in wordsInOrder[number:number+rangeStep]:
		tmp_words.append(word)
	print str(number) + " / " + str(len(wordsInOrder))
	taggedWordsInOrder.append(tagger.tag(tmp_words))

with open(mainPath+"/allTEDTalkContent_Tagged.txt",'w') as f:
		json.dump(taggedWordsInOrder, f)
		print "saved"
'''


# Load the XML with all talks and iterate through the childs of the root element
tree = ET.parse(mainPath+"/ted.xml")
root = tree.getroot()
for child in root:

	# Keep track of time per TEDTalk
	startTimeTalk = time.time()

	# Get the TalkID and the Speaker to generate the folders
	ident = child.find('head').find('talkid').text.encode('utf-8')
	speaker = child.find('head').find('speaker').text.encode('utf-8')
	folder = savePath+"/"+ident+"-"+speaker

	# Get the original text for the original file
	text = child.find('content').text.encode('utf-8')
	
	# Check if the TEDTalk modifications already has been generated
	if not os.path.exists(folder):

		os.makedirs(folder)

		# Tokenize text and copy the tokenized version to be modified later
		tokens = word_tokenize(text.decode('utf-8'))
		changedTokens = list(tokens)
		changedTokensIndicated = list(tokens)

		# Create local varibales to keep track of the nouns and their position in the text to keep track
		nouns = []
		positions = []

		# Only use TEDTalks that have more than 200 tokens, to exclude concerts
		if len(tokens) < 200:
			os.rmdir(folder)
			print "The TEDTalk "+folder+" ist shorter than 200 words! Nothing will be created"
		else:

			# If all the above conditions are met, create a new folder for the TEDTalk
			#print "create data in Folder "+ident+"-"+speaker

			# Save the original file in the folder
			g = open(folder+"/original_text.txt","w")
			g.write(text)
			g.close()

			# Create the tags for the current TEDTalk to find nouns later
			#tagged_Tokens = []
			#for token in tokens:
			#	tagged_Tokens.append(tagger.tag([token])[0])
			#	print tagged_Tokens

			#tagged_Tokens = tagger.tag(tokens)

			tagged_Tokens = []
			step = 100
			for start in range(0,len(tokens),step):
				end = start+step
				if start+step > len(tokens):
					end = len(tokens)
				tagged_Tokens.extend(tagger.tag(tokens[start:end]))
				print "From "+str(start)+" to "+str(end) + " out of "+str(len(tokens))

			# For every token in the text, check if it is some kind of noun and add it to the nouns list 
			# and store the respective position in the text
			for position, (word, token) in enumerate(tagged_Tokens):
				if token.startswith('N'):
					nouns.append(word)
					positions.append(position)

			# Join the nouns and their positions together in a list of tuples (position, word)
			nounsWithPosition = zip(positions, nouns)

			# Create an empty list for the nouns that are reoccuring in a specified interval
			reoccurendValues = []

			# Iterate through all the nouns in the text and compare the noun with its [history] predecessors
			# if the nouns has been occuring within the history, save it into the reoccurendValues list
			for position, (abs_Position, noun) in enumerate(nounsWithPosition):
				if position >= history:
					for checkNoun in nouns[position-history:position]:
						if noun == checkNoun:
							reoccurendValues.append(nounsWithPosition[position])
							continue

			# Generate lists to keep track of the words that should be interchanged from the original text later
			replace_Words =[]
			replaceTracker = []
			counter = 0

			# Choose [COUNT_REPLACE] random nouns that are reoccuring within the [history] to be changed
			while counter < COUNT_REPLACE:
				counter += 1
				print "Exchanging word " +str(counter) + " out of "+str(COUNT_REPLACE)
				# Check that there are reoccured values that can be changed
				if len(reoccurendValues) > 0:

					# Check if the value that we randomly choose has not been changed before already to avoid double changes
					tokenToChange = None
					count = 0
					while count < 100:
						count += 1
						tokenToChange = reoccurendValues[random.randint(0, len(reoccurendValues) - 1)]
						found = 0
						for item in replace_Words:
							if item[0] == tokenToChange[0]:
								found = 1
						if found == 0:
							break

					if tokenToChange == None:
						continue

					# Save the word that should be changed and initialize the substitute word with the word itself
					word = tokenToChange[1]
					substitute_Word = word
					
					# Search the word that we want to change in the list of words sorted by occurance
					# reason: take a word to interchange that has an equal number of occurances to keep a really rough context
					wordIndex = -1
					for index, item in enumerate(wordsInOrder): 
						if item[0] == word: 
							wordIndex = index
							break

					# Get the words with the same word type and similar appearances in the defined [FRAME]
					# to make sure to find usable words, the FRAME can double it's size twice if needed
					nearestWords = []
					frame = FRAME
					found = 0
					while found < 3:
						onlyWord = []
						# Iterate through the words in the FRAME around the word we want to change to get only the word out of the tuple
						for word in wordsInOrder[wordIndex-frame:wordIndex+frame]:
							onlyWord.append(word[0])
						
						# Get the tags of all the words in the FRAME around the word to change
						#taggedWordsInOrder = tagger.tag(onlyWord)
						taggedWordsInOrder = []
						step = 20
						for start in range(0,len(onlyWord),step):
							end = start+step
							if start+step > len(onlyWord):
								end = len(onlyWord)
							taggedWordsInOrder.extend(tagger.tag(onlyWord[start:end]))

						# Also get the tag of the word to change
						word_to_compare = tagger.tag([tokenToChange[1]])
						for item in taggedWordsInOrder: 

							# Check the words and the FRAME-words for the exactly same tag and check that they are not the exactly same word
							if item[1] == word_to_compare[0][1] and item[0] != word_to_compare[0][0]:
								
								#Safe all the words in the FRAME that have the right tag in the list nearsetWords
								nearestWords.append(item[0])
								found = 3
						frame *= 2
						found += 1
						#if found != 3:
							#print "Extended frame"

						# For all the possible interchange words with the right tag
						if len(nearestWords) > 0:
							substitute_Word = nearestWords[random.randint(0, len(nearestWords) - 1)]
							#print "substitude_Word: "+substitute_Word
							for char in substitute_Word:
								if char in string.punctuation:
									substitute_Word = word

					# Keep track of all the words that should be replaced
					replace_Words.append(tuple((tokenToChange[0], substitute_Word)))

			print "replace_Words"
			print replace_Words


			# After iterating through all the words that should be changed
			# Override tokens with substitutes and save changeLog
			for new_token in replace_Words:
				print "new_token"
				print new_token
				if tokens[new_token[0]] != new_token[1]:
					if not isinstance(new_token[1], tuple):
						print "new_token[1] is not a tupel"
						changedTokens[new_token[0]] = new_token[1]
						changedTokensIndicated[new_token[0]] = "###"+ new_token[1] +"###"
						replaceTracker.append([new_token[0], tokens[new_token[0]], new_token[1]])
					
					#check if it's a tupel with a string at position [0] --> bug in the code, should normally not happen
					else:
						changedTokens[new_token[0]] = new_token[1][0]
						changedTokensIndicated[new_token[0]] = "###"+ new_token[1][0] +"###"
						replaceTracker.append([new_token[0], tokens[new_token[0]], new_token[1][0]])

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
			
			# Saving the changeLog into new file
			with open(folder+"/changes_text.txt","w") as f:
				json.dump(replaceTracker, f)

			endTimeTalk = time.time()
			print "It took "+ str((endTimeTalk-startTimeTalk)/60.0) +" min to create the modifications for the TEDTalk " + ident + "-" + speaker 

