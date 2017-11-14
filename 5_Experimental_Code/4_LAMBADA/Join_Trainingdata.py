
# Imports
########################################################################
import os
import numpy
import shutil, errno
########################################################################

path = "../../../lambada/train-novels"

with open(path + "/LAMBADA_joined.txt", "w") as file:
	file.write(" ")

count = 0

for f in os.listdir(path):
	child = os.path.join(path, f)
	if os.path.isdir(child):
		for g in os.listdir(child):
			grand_child = os.path.join(path, f, g)
			path_gc  = open(grand_child, "r")
			text = path_gc.read().decode('utf8')
			with open(path + "/LAMBADA_joined.txt", "a") as file:
				file.write(text.replace('\n', '').encode('utf8'))
			count += 1
			print "Text nr:" + str(count)


