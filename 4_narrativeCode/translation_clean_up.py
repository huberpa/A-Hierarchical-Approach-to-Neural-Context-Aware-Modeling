
# Commandline Arguments
##############################################
import optparse
parser = optparse.OptionParser()
parser.add_option('--ted_path', action="store", dest="ted_path", help="The path to the ted data folder (default: .)", default=".")
parser.add_option('--translation_path', action="store", dest="translation_path", help="The path to the translation data folder (default: .)", default=".")
parser.add_option('--save_path', action="store", dest="save_path", help="The path to the monolingual data folder (default: .)", default=".")
parser.add_option('--name', action="store", dest="name", help="The name of the model (default: model)", default="model")

options, args = parser.parse_args()
data_path = options.translation_path
ted_path = options.ted_path
model_name = options.name
save_path = options.save_path
##############################################

# Imports
##############################################
from xml.dom.minidom import parse, parseString
import json
import os
from collections import Counter
import datetime
import sys
reload(sys)  
sys.setdefaultencoding('utf-8')
##############################################

# Main
##############################################

# Load files
print "Loading English file..."
english_talks = []
count = 0
english_path  = open(data_path+"/train.tags.en-de.en", "r")
english_text = english_path.read()
englishFile = parseString(english_text)
english_documents=englishFile.getElementsByTagName('doc')
for document in english_documents:
	content=document.getElementsByTagName('talkid')
	for talk in content:
		node_value = talk.childNodes[0].nodeValue
		found_value = 0
		for f in os.listdir(ted_path):
			child = os.path.join(ted_path, f)
			if os.path.isdir(child):
				if f.split("-",1)[0] == node_value:
					found_value = 1
					count += 1
					content=document.getElementsByTagName('content')
					for talk in content:
						node_value_2 = talk.childNodes[0].nodeValue
						splitted_talk = node_value_2.split('\n')
						for line in splitted_talk:
							if len(line) > 1:
								english_talks.append(line)
		if found_value == 0:
			print "Talk ID not found: " + node_value

print "Count of talks found in english: "+str(count)

count = 0
print "Loading German file..."
german_talks = []
german_path  = open(data_path+"/train.tags.en-de.de", "r")
german_text = german_path.read()
germanFile = parseString(german_text)
german_documents=germanFile.getElementsByTagName('doc')
for document in german_documents:
	content=document.getElementsByTagName('talkid')
	for talk in content:
		node_value = talk.childNodes[0].nodeValue
		for f in os.listdir(ted_path):
			child = os.path.join(ted_path, f)
			if os.path.isdir(child):
				if f.split("-",1)[0] == node_value:
					count += 1
					content=document.getElementsByTagName('content')
					for talk in content:
						node_value_2 = talk.childNodes[0].nodeValue
						splitted_talk = node_value_2.split('\n')
						for line in splitted_talk:
							if len(line) > 1:
								german_talks.append(line)

print "Count of talks found in german: "+str(count)

with open(save_path+"/"+model_name+"_english.txt", "w") as f:	
	for element in english_talks:
		f.write("{}\n".format(element))

#with open(save_path+"/"+model_name+"_english.txt", "w") as f:
#		json.dump(english_talks, f)

with open(save_path+"/"+model_name+"_german.txt", "w") as f:	
	for element in german_talks:
		f.write("{}\n".format(element))

#with open(save_path+"/"+model_name+"_german.txt", "w") as f:
#		json.dump(german_talks, f)

