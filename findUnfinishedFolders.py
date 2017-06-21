import os
import sys

count = 0
countNoChange = 0
countWrongEncoding = 0
path = "./../tedData/Automated_change_All_Files_v3"

for f in os.listdir(path):
	child = os.path.join(path, f)
	if os.path.isdir(child):
		if len([name for name in os.listdir(child) if os.path.isfile(os.path.join(child, name))]) < 4:
			print "folder with only one file: "+f
  			count += 1
  			for root, dirs, files in os.walk(child, topdown=False):
				for name in files:
					os.remove(os.path.join(root, name))
				for name in dirs:
					os.rmdir(os.path.join(root, name))
				os.rmdir(root)
		else:
			with open(child+"/changes_text.txt", "r") as g:
				text = g.read()
				if text == "[]":
					countNoChange += 1
					for root, dirs, files in os.walk(child, topdown=False):
						try:
							for name in files:
								os.remove(os.path.join(root, name))
							for name in dirs:
								os.rmdir(os.path.join(root, name))
							os.rmdir(root)
						except Exception:
							print "Exception"

			with open(child+"/changes_text.txt", "r") as g:
				text = g.read()
				if text.find("\\") != -1:
					countWrongEncoding += 1
					
					for root, dirs, files in os.walk(child, topdown=False):
						try:
							for name in files:
								os.remove(os.path.join(root, name))
							for name in dirs:
								os.rmdir(os.path.join(root, name))
							os.rmdir(root)
						except Exception:
							print "Exception"

print "There are "+str(count)+" folders with less than 4 files. all of them got deleted!"
print "There are "+str(countNoChange)+" folders with no changes. all of them got deleted!"
print "There are "+str(countWrongEncoding)+" folders with wrong encoding changes. all of them got deleted!"

