import gensim
import sys

model=gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin',binary=True, limit=500000)

words = ["technology", "Artificial", "intelligence", "handcraft", "knowledge", "tricks", "human", "level", "limit", "super", "intelligence", "changes", "everything"]
print ""
print ""

for word in words:
	try:
		print "Words close to "+word+": "
		print model.most_similar_cosmul(positive=[word], topn=20)
		print ""
		print ""
	except Exception:
		print "Words close to "+word+" could not be found!"
		print ""
		print ""
