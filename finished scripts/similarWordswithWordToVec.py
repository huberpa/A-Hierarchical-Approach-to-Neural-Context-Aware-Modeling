import gensim
import sys


model=gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin',binary=True, limit=500000)
print ""
print ""
print "Words close to woman+king-man:"
print model.most_similar_cosmul(positive=['woman', 'king'], negative=['man'], topn=20)
print ""
print ""
print "Words close to train-tracks+car:"
print model.most_similar_cosmul(positive=['train', 'car'], negative=['tracks'], topn=20)
print ""
print ""
print "Words close to technology:"
print model.most_similar_cosmul(positive=['technology'], topn=20)
print ""
print ""
print "Words close to challenge:"
print model.most_similar_cosmul(positive=['challenge'], topn=20)

