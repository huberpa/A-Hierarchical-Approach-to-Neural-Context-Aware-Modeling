# Imports
##############################################
from collections import Counter
from nltk import word_tokenize
import sys
import random
import os.path
import argparse
import xml.etree.ElementTree as ElementTree
import json
import re
import string
import time
from nltk.tag.stanford import StanfordPOSTagger
import optparse
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
##############################################



# Command Line Arguments
##############################################
cmdLineArgs = optparse.OptionParser()
cmdLineArgs.add_option('--pos_tagger', action="store", dest="pos_tagger", help="Relative path to the Stanford POS-Tagger folder (default: ./lib/stanford-postagger)", default="./lib/stanford-postagger")
cmdLineArgs.add_option('--data_dir', action="store", dest="data_dir", help="Relative path to the main directory of the data (default: .)", default=".")
cmdLineArgs.add_option('--save_dir', action="store", dest="save_dir", help="Relative path from the main directory (defined by --data_dir) to the output directory (default: /output)", default="/output")
cmdLineArgs.add_option('--load_file', action="store", dest="load_file", help="Relative path from the main directory (defined by --data_dir) to the data file (default: /ted.xml)", default="/ted.xml")
cmdLineArgs.add_option('--plain_file', action="store", dest="plain_file", help="Relative path from the main directory (defined by --data_dir) to the raw text file. This file is used to create a list of words (default: /tokenizedContent.txt)", default="/tokenizedContent.txt")
cmdLineArgs.add_option('--context_length', action="store", dest="context_length", help="Defines the number of consecuetive nouns generating the context (default: 10)", default=10)
cmdLineArgs.add_option('--replacement_frame', action="store", dest="replacement_frame", help="Defines the window around the original noun from which a suitable replacement is selected (default: 20)", default=20)
cmdLineArgs.add_option('--replacement_extensions', action="store", dest="replacement_extensions", help="Defines the number of window extensions around the original noun if no suitable replacement can be found within the original window (defined by --replacement_frame). Every extension increases the window by adding another replacement frame (default: 3)", default=3)
cmdLineArgs.add_option('--replacement_count', action="store", dest="replacement_count", help="Defines the total number of replacements per dataset (TEDTalk) (default: 10)", default=10)

options, args = cmdLineArgs.parse_args()
posTagger = options.pos_tagger
dataPath = options.data_dir
save_Ext = options.save_dir
load_Ext = options.load_file
plain_Ext = options.plain_file
context_length = int(options.context_length)
replacement_frame = int(options.replacement_frame)
replacement_extensions = int(options.replacement_extensions)
replacement_count = int(options.replacement_count)
##############################################



# Class Definition
##############################################
class SubstitutionProcess:

	def __init__(self, posTagger, filePath, saveExtension, loadExtension, plainExtension, contextLength, replacementFrame, replacementFrameExtensions, replacementCount):
		self.posTagger_Model = posTagger + "/models/english-bidirectional-distsim.tagger"
		self.posTagger_Jar = posTagger + "/stanford-postagger.jar"
		self.filePath = filePath
		self.savePath = filePath + saveExtension
		self.loadPath = filePath + loadExtension
		self.plainPath = filePath + plainExtension
		self.contextLength = contextLength
		self.replacementFrame = replacementFrame
		self.replacementFrameExtensions = replacementFrameExtensions
		self.replacementCount = replacementCount
		self.allTokens = []
		self.tagger = StanfordPOSTagger(self.posTagger_Model, self.posTagger_Jar, java_options='-mx1000m -Xmx1028m')

	def untokenize(self, words):
		text = ' '.join(words)
		step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
		step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
		step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
		step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
		step5 = step4.replace(" '", "'").replace(" n't", "n't").replace("can not", "cannot")
		step6 = step5.replace(" ` ", " '")
		return step6.strip()

	def tokenizeData(self):
		if not os.path.exists(self.filePath+"/Tokenized_Data.txt"):
			raw_input = word_tokenize(open(self.plainPath,"r").read().decode('utf-8'))
			with open(self.filePath+"/Tokenized_Data.txt",'w') as f:
				json.dump(raw_input, f)

		data  = open(self.filePath+"/Tokenized_Data.txt",'r')
		return data

	def countWords(self, data):
		wordsInOrder = sorted(Counter(json.load(data)).items(), key=lambda item: item[1])
		return wordsInOrder

	def getXMLElements(self):
		root = ElementTree.parse(self.loadPath).getroot()
		return root	

	def getMetaData(self, element):
		ident = element.find('head').find('talkid').text.encode('utf-8')
		speaker = element.find('head').find('speaker').text.encode('utf-8')
		folder = self.savePath+"/"+ident+"-"+speaker
		text = element.find('content').text.encode('utf-8')
		return ident, speaker, folder, text

	def removeInvalidData(self, data, folder, split):
		# Only use TEDTalks that have more than SPLIT tokens, to exclude concerts
		if len(data) < split:
			return False
		else:
			# Save the original file in the folder
			origText = self.untokenize(data)
			if not os.path.exists(folder):
				os.makedirs(folder)
			g = open(folder+"/original_text.txt","w")
			g.write(origText)
			g.close()
			return True

	def tagData(self, tokens, batchSize):
		tagged_Tokens = []
		for start in range(0,len(tokens),batchSize):
			end = start+batchSize
			if start+batchSize > len(tokens):
				end = len(tokens)
			tagged_Tokens.extend(self.tagger.tag(tokens[start:end]))
		return tagged_Tokens

	def findNouns(self, tagged_Tokens):
		nouns = []
		positions = []
		for position, (word, token) in enumerate(tagged_Tokens):
			if token.startswith('N'):
				nouns.append(word)
				positions.append(position)
		nounsWithPosition = zip(positions, nouns)
		return nounsWithPosition, nouns

	def findReoccurrentNouns(self, nounsWithPosition, onlyNouns):
		reoccurrendValues = []
		# Iterate through all the nouns in the text and compare the noun with its [history] predecessors
		# if the nouns has been occuring within the history, save it into the reoccurendValues list
		for position, (abs_Position, noun) in enumerate(nounsWithPosition):
			if position >= self.replacementFrame:
				for checkNoun in onlyNouns[position-self.contextLength:position]:
					if noun == checkNoun:
						reoccurrendValues.append(nounsWithPosition[position])
		return reoccurrendValues

	def findAbsoluteWordPosition(self, wordsInOrder, word):
		wordIndex = -1
		for index, item in enumerate(wordsInOrder): 
			if item[0] == word: 
				wordIndex = index
				break
		return wordIndex

	def findUniqueReplacement(self, replaced_Words, reoccurrentNouns):
		for selection in range(0, 100):
			tokenToChange = reoccurrentNouns[random.randint(0, len(reoccurrentNouns) - 1)]
			found = 0
			for item in replaced_Words:
				if item[0] == tokenToChange[0]:
					found = 1
			if found == 0:
				break
		if tokenToChange == None:
			return False
		else:
			return tokenToChange

	def findSubstitute(self, wordsInOrder, wordIndex, word_to_compare):
		# Get the words with the same word type and similar appearances
		# to make sure to find usable words, the replacementFrame can double it's size twice if needed
		nearestWords = []
		frame = self.replacementFrame
		found = 0
		while found < self.replacementFrameExtensions:
			
			onlyWord = []
			for word in wordsInOrder[wordIndex-frame:wordIndex+frame]:
				onlyWord.append(word[0])
			
			# Get the tags of all the words in the FRAME around the word to change
			taggedWordsInOrder = self.tagData(onlyWord, 20)

			for item in taggedWordsInOrder: 

				# Check the words for the exactly same tag and check that they are not the exactly same word
				if item[1] == word_to_compare[0][1] and item[0] != word_to_compare[0][0]:
					
					#Safe all the words in the frame that have the right tag
					nearestWords.append(item[0])
					found = 3

			frame *= 2
			found += 1

			# For all the possible interchange words with the right tag
			if len(nearestWords) > 0:
				substitute_Word = nearestWords[random.randint(0, len(nearestWords) - 1)]
				for char in substitute_Word:
					if char in string.punctuation:
						substitute_Word = word
				if len(substitute_Word) < 3:
						substitute_Word = word

		return substitute_Word

	def createOutputs(self, replace_Words, changedTokens, changedTokensSync, changedTokensIndicated, replaceTracker):
		for new_token in replace_Words:
			if tokens[new_token[0]] != new_token[1]:
				if not isinstance(new_token[1], tuple):
					changedTokens[new_token[0]] = new_token[1]
					changedTokensSync[new_token[0]] = tokens[new_token[0]] + "___" + new_token[1]
					changedTokensIndicated[new_token[0]] = "___"+ new_token[1] +"___"
					replaceTracker.append([new_token[0], tokens[new_token[0]], new_token[1]])
				else:
					changedTokens[new_token[0]] = new_token[1][0]
					changedTokensSync[new_token[0]] = tokens[new_token[0]] + "___" + new_token[1][0]
					changedTokensIndicated[new_token[0]] = "___"+ new_token[1][0] +"___"
					replaceTracker.append([new_token[0], tokens[new_token[0]], new_token[1][0]])

		return changedTokens, changedTokensSync, changedTokensIndicated, replaceTracker
##############################################



# Main
##############################################
print "Creating Object"
process = SubstitutionProcess(posTagger = posTagger, 
                              filePath = dataPath, 
                              saveExtension = save_Ext, 
                              loadExtension = load_Ext, 
                              plainExtension = plain_Ext, 
                              contextLength = context_length, 
                              replacementFrame = replacement_frame, 
                              replacementFrameExtensions = replacement_extensions, 
                              replacementCount = replacement_count)

print "Tokenizing Data"
tokenized_data = process.tokenizeData()

print "Counting Words"
apperanceOrderedWords = process.countWords(tokenized_data)

print "Decomposing XML File"
dataRoot = process.getXMLElements()

print "Starting Loop"
for child in dataRoot:

	# Keep track of time per Instance
	startTime = time.time()

	# Get the elements Metadata
	print "Getting MetaData"
	ident, speaker, folder, text = process.getMetaData(child)

	print "Iteration with: "+ str(ident) + " - " + speaker

	# Initialize the Token Sets
	tokens = word_tokenize(text.decode('utf-8'))
	changedTokens = list(tokens)
	changedTokensIndicated = list(tokens)
	changedTokensSync = list(tokens)
	replaceTracker = []
	replace_Words =[]

	# Remove the element if there are less than 200 words
	print "Removing Invalid Data"
	valid = process.removeInvalidData(tokens, folder, 200)
	if not valid:
		continue

	# Tag the Data
	print "Tagging Data"
	tagged_Tokens = process.tagData(tokens, 100)

	# Find the Nouns and their respective position
	print "Searching Nouns"
	nounsWithPosition, onlyNouns = process.findNouns(tagged_Tokens)

	# Find the Nouns that reoccur within the defined window
	print "Searching Reoccurrent Nouns"
	reoccurrentNouns = process.findReoccurrentNouns(nounsWithPosition, onlyNouns)

	print "Getting Replacements"
	for element in range(0, process.replacementCount):
		if len(reoccurrentNouns) > 0:

			print "Replacement "+str(element+1)
			tokenToChange = process.findUniqueReplacement(replace_Words, reoccurrentNouns)
			if tokenToChange == False:
				continue

			word_to_compare = process.tagger.tag([tokenToChange[1]])
			word = tokenToChange[1]
			substitute_Word = word
			
			# Search the word that we want to change in the list of words sorted by occurance
			wordIndex = process.findAbsoluteWordPosition(apperanceOrderedWords, word)

			substitute_Word = process.findSubstitute(apperanceOrderedWords, wordIndex, word_to_compare)

			# Keep track of all the words that should be replaced
			replace_Words.append(tuple((tokenToChange[0], substitute_Word)))


	changedTokens, changedTokensSync, changedTokensIndicated, replaceTracker = process.createOutputs(replace_Words, changedTokens, changedTokensSync, changedTokensIndicated, replaceTracker)

	# Untokenizing the text
	text = process.untokenize(changedTokens)
	textSync = process.untokenize(changedTokensSync)
	textIndicated = process.untokenize(changedTokensIndicated)

	# Saving the modified text into new file
	writeTo = open(folder+"/modified_text.txt","w")
	writeTo.write(text.encode('utf-8'))
	writeTo.close()

	# Saving the modified text into new file
	writeTo = open(folder+"/sync_text.txt","w")
	writeTo.write(textSync.encode('utf-8'))
	writeTo.close()

	# Saving the indicated text into new file
	writeToIndicated = open(folder+"/indicated_text.txt","w")
	writeToIndicated.write(textIndicated.encode('utf-8'))
	writeToIndicated.close()
	
	# Saving the changeLog into new file
	with open(folder+"/changes_text.txt","w") as f:
		json.dump(replaceTracker, f)

	endTime = time.time()
	print "It took "+ str((endTime-startTime)/60.0) +" min to create the modifications for the TEDTalk " + ident + "-" + speaker 
##############################################
