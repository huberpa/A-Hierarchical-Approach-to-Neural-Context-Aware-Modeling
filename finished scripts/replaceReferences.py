from nltk import word_tokenize, pos_tag

history = 5

f  = open("./../tedData/Manual_change/original_text.txt", "r")
text = f.read().decode('unicode-escape')

nouns = [token for token, wordType in pos_tag(word_tokenize(text)) if wordType.startswith('N')]

doubleValues = []

for position, noun in enumerate(nouns):
	if position >= history:
		for checkNoun in nouns[position-history:position]:
			#print "checking "+noun+" with "+checkNoun
			if noun == checkNoun:
				doubleValues.append(noun)
				continue

for position, word in enumerate(doubleValues):
	print position, word

#find other word with Word2Vec or NLTK
#substitute word by k nearest word in space

#newsshuffle.de