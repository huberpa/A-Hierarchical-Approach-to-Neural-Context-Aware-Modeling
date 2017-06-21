
# Imports
########################################################################
import os
import numpy
import shutil, errno
########################################################################



# Function to save the texts together in one file
########################################################################
def collapse_data(type_of_set):
	for f in os.listdir(destination+"/"+type_of_set):
		child = os.path.join(destination+"/"+type_of_set, f)
		if os.path.isdir(child):

			with open(child+"/original_text.txt", "r") as txt:
				text = txt.read()
				with open(destination+"/"+type_of_set+"/original_"+type_of_set+"_texts.txt", "a") as all_txt:
					 all_txt.write("{}\n".format(text))

			with open(child+"/modified_text.txt", "r") as txt:
				text = txt.read()
				with open(destination+"/"+type_of_set+"/modified_"+type_of_set+"_texts.txt", "a") as all_txt:
					 all_txt.write("{}\n".format(text))

			with open(child+"/sync_text.txt", "r") as txt:
				text = txt.read()
				with open(destination+"/"+type_of_set+"/sync_"+type_of_set+"_texts.txt", "a") as all_txt:
					 all_txt.write("{}\n".format(text))


			with open(child+"/indicated_text.txt", "r") as txt:
				text = txt.read()
				with open(destination+"/"+type_of_set+"/indicated_"+type_of_set+"_texts.txt", "a") as all_txt:
					 all_txt.write("{}\n".format(text))

			with open(child+"/changes_text.txt", "r") as txt:
				text = txt.read()
				with open(destination+"/"+type_of_set+"/changes_"+type_of_set+"_texts.txt", "a") as all_txt:
					 all_txt.write("{}\n".format(f + ": " + text))
########################################################################



# Execution
########################################################################
count = 0
path = "./../tedData/Automated_change_All_Files_v3"
destination = "./sets"

training = 0
test = 0
dev = 0

if not os.path.exists("./sets"):
    os.makedirs("./sets")

if not os.path.exists("./sets/training"):
    os.makedirs("./sets/training")

if not os.path.exists("./sets/development"):
    os.makedirs("./sets/development")

if not os.path.exists("./sets/test"):
    os.makedirs("./sets/test")

for f in os.listdir(path):
	child = os.path.join(path, f)
	if os.path.isdir(child):
  		count += 1
  		x = numpy.random.rand(1)[0]
		if x < .6:
			print "TRAINING"
			training += 1
			shutil.copytree(child, "./sets/training/"+f)

		if x < .8 and x > .6:
			print "DEV"
			dev += 1
			shutil.copytree(child, "./sets/development/"+f)

		if x > .8:
			print "TEST"
			test += 1
			shutil.copytree(child, "./sets/test/"+f)			

collapse_data("training")
collapse_data("development")
collapse_data("test")

print "There are "+str(count)+" folders"
print "Training: " + str(training) + " --- Dev: " + str(dev) + " --- Test: " + str(test)
########################################################################


