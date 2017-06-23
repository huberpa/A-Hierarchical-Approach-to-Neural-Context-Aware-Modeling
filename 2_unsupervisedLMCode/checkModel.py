from keras.models import load_model
import numpy as np
import json
import optparse
import nltk

parser = optparse.OptionParser()
parser.add_option('-i', '--input', action="store", dest="input", help="Input sentence", default="")
parser.add_option('-m', '--model', action="store", dest="model", help="relative path to the Model to test", default="")
parser.add_option('-v', '--vocab', action="store", dest="vocab", help="relative path to the vocabulary folder", default="")

options, args = parser.parse_args()
print options
input_sentence = options.input.lower()
model_path = options.model
vocab_path = options.vocab

model = load_model(model_path)
print model.summary() 

word_to_index = []
with open(vocab_path + "/word_to_index.json") as f:    
	word_to_index = json.load(f)

index_to_word = []
with open(vocab_path + "/index_to_word.json") as f:    
	index_to_word = json.load(f)

words = nltk.word_tokenize(input_sentence)
input_idx = []
for word in words:
	try:
		input_idx.append(word_to_index[word])
	except Exception:
		input_idx.append(word_to_index['<UNKWN>'])

print ""
print ('#'*50)
print ""

testInput = np.zeros((1, len(input_idx)), dtype=np.int16)
for index, idx in enumerate(input_idx):
	testInput[0, index] = idx

prediction = model.predict(testInput, verbose=0)[0]

outputs = []
for index1, word in enumerate(prediction):
	maxValue = 0
	maxIndex = 0
	for index2, wordProbability in enumerate(word):
		if wordProbability > maxValue:
			if not index_to_word[str(index2)] == "<UNKWN>":
				maxValue = wordProbability
				maxIndex = index2
	outputs.append(maxIndex)

for idx, indx in enumerate(input_idx):
	print index_to_word[str(input_idx[idx])] + "\t -->\t " + index_to_word[str(outputs[idx])]

print ""
print ('#'*50)
print ""

