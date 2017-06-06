import nltk
from nltk import word_tokenize

text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())

print len(text)
print ""
print "Words similar to information:"
print text.similar('information')
print ""
print ""
print "Words similar to intelligence:"
print text.similar('intelligence')
print ""
print ""
print "Words similar to people:"
print text.similar('people')
print ""
print ""
print "Words similar to technology:"
print text.similar('technology')
print ""
print ""