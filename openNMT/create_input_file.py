import optparse
parser = optparse.OptionParser()
parser.add_option('--dataset', action="store", dest="dataset", help="Choose the dataset to train on [PRD,DEV,SMALL] (default: PRD)", default="PRD")
parser.add_option('--save', action="store", dest="save", help="Choose the save path (default: .)", default=".")
options, args = parser.parse_args()
save = options.save
dataset = options.dataset
if dataset == "DEV":
	training_data_src = "./../translation_files/dev_en.xml"
	training_data_tgt = "./../translation_files/dev_de.xml"
if dataset == "PRD":
	training_data_src = "./../translation_files/translation_training_english.txt"
	training_data_tgt = "./../translation_files/translation_training_german.txt"
##############################################

# Imports
##############################################
import numpy as np
import random
import sys
import nltk
import os
import datetime
import json
from xml.dom.minidom import parse, parseString
reload(sys)  
sys.setdefaultencoding('utf-8')
##############################################

# Main
##############################################

tokens = []

if dataset == "DEV":
	path  = open(training_data_src, "r")
	txt = path.read()
	file = parseString(txt)
	documents=file.getElementsByTagName('doc')
	for document in documents:
		content=document.getElementsByTagName('seg')
		for talk in content:
			node_value = talk.childNodes[0].nodeValue
			tokens.append(nltk.word_tokenize(node_value))
	tokens = [[word.lower() for word in line] for line in tokens]

if dataset == "PRD":
	path  = open(training_data_src, "r")
	text = path.read().decode('utf8').lower()
	tokens = []
	for index, sentence in enumerate(nltk.sent_tokenize(text)): 
		tokens.append(nltk.word_tokenize(sentence))

with open(save+"_src.txt", "w") as f:
	for sentence in tokens:
		sent = ""
		for word in sentence:
			sent += word + " "
		f.write("{}\n".format(sent))



if dataset == "DEV":
	path  = open(training_data_tgt, "r")
	txt = path.read()
	file = parseString(txt)
	documents=file.getElementsByTagName('doc')
	for document in documents:
		content=document.getElementsByTagName('seg')
		for talk in content:
			node_value = talk.childNodes[0].nodeValue
			tokens.append(nltk.word_tokenize(node_value))
	tokens = [[word.lower() for word in line] for line in tokens]

if dataset == "PRD":
	path  = open(training_data_tgt, "r")
	text = path.read().decode('utf8').lower()
	tokens = []
	for index, sentence in enumerate(nltk.sent_tokenize(text)): 
		tokens.append(nltk.word_tokenize(sentence))

with open(save+"_tgt.txt", "w") as f:
	for sentence in tokens:
		sent = ""
		for word in sentence:
			sent += word + " "
		f.write("{}\n".format(sent))
print "done"