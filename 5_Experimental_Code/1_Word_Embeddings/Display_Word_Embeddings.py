import numpy as np
from skdata.mnist.views import OfficialImageClassification
from matplotlib import pyplot as plt
from tsne import bh_sne
import json

# load up data
with open ("./embeddings.txt", 'r') as f:
    data = json.load(f)

x_data = []
y_data = []
#data = OfficialImageClassification(x_dtype="float32")
for point in data:
	x_data.append(point[2])
	y_data.append(point[1])

# convert image data to float64 matrix. float64 is need for bh_sne
x_data = np.asarray(x_data).astype('float64')
x_data = x_data.reshape((x_data.shape[0], -1))

# For speed of computation, only run on a subset
n = 2000
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