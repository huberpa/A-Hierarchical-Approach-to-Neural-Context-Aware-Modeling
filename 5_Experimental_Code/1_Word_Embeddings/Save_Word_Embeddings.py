import numpy as Math
import pylab as Plot
import tsne
import optparse
import json
from sklearn import manifold

parser = optparse.OptionParser()
parser.add_option('--data_path', action="store", dest="data_path", help="The path to the embedding.txt file (default: .)", default=".")

options, args = parser.parse_args()
data_path = options.data_path

with open (data_path, 'r') as f:
    matrix = json.load(f)

matrix = matrix[:2]
rows = [word[1] for word in matrix[:4]]
target_matrix = [word[2] for word in matrix[:4]]


tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
reduced_matrix = tsne.fit_transform(target_matrix)

Plot.figure(figsize=(200, 200), dpi=100)
max_x = Math.amax(reduced_matrix, axis=0)[0]
max_y = Math.amax(reduced_matrix, axis=0)[1]
Plot.xlim((-max_x,max_x))
Plot.ylim((-max_y,max_y))

Plot.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], 20);

for row_id in range(0, len(rows)):
    target_word = rows[row_id]
    x = reduced_matrix[row_id, 0]
    y = reduced_matrix[row_id, 1]
    Plot.annotate(target_word, (x,y))

Plot.savefig("image.png");