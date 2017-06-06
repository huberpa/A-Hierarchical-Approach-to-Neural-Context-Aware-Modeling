from nltk import word_tokenize

f  = open("./../tedData/allTEDTalkContent.txt", "r")

text = f.read().decode('unicode-escape')
tokens = word_tokenize(text)
print tokens

