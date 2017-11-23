import os
path ="."
endValue = []

for f in os.listdir(path):
	child = os.path.join(path, f)
	if f != "collect.py" and f != ".DS_Store":
		with open(child, "r") as h:
			tosave = h.read().replace(".",",").split("\n")
			if len(f[f.rfind("_")+1:]) == 1:
					f = f[:-1]+"0"+f[-1:]
			if len(endValue) == 0:
				endValue.append(";" + f + ";")
				for value in tosave:
					line = value.split(";")
					if len(line)>3:
						endValue.append(line[0] + ";" + line[2] + ";")
			else:
				endValue[0] = endValue[0] + f + ";"
				for idx, value in enumerate(tosave):
					line = value.split(";")
					if len(line)>3 and idx < (len(endValue)-1):
						endValue[idx+1] = endValue[idx+1] + line[2] + ";"

with open("./all.txt", "w") as g:
	for element in endValue:
		g.write("{}\n".format(element))

print "done"
