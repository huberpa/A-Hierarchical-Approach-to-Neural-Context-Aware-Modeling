import numpy as np
from skdata.mnist.views import OfficialImageClassification
from matplotlib import pyplot as plt
from tsne import bh_sne
import optparse
import json

parser = optparse.OptionParser()
parser.add_option('--data_path', action="store", dest="data_path", help="The path to the embedding.txt file (default: .)", default=".")

options, args = parser.parse_args()
data_path = options.data_path

with open (data_path, 'r') as f:
    data = json.load(f)

x_data = []
y_data = []
#data = OfficialImageClassification(x_dtype="float32")
for point in data:
	x_data.append(point[1])
	y_data.append(point[0])

# convert image data to float64 matrix. float64 is need for bh_sne
x_data = np.asarray(x_data).astype('float64')
x_data = x_data.reshape((x_data.shape[0], -1))

# For speed of computation, only run on a subset
n = 200
x_data = x_data[:n]

# perform t-SNE embedding
vis_data = bh_sne(x_data)

# plot the result
vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]

plt.scatter(vis_x, vis_y)

for i, txt in enumerate(y_data[:n]):
    plt.annotate(txt, (vis_x[i],vis_y[i]))
plt.show()