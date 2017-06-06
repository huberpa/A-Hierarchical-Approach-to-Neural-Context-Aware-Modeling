import os
import requests
import json
import sys

basedir = '/Users/Patrick/Documents/Masterthesis/OpenSubtitles2016/raw/en'

for year in os.listdir(basedir):
	if "." not in year: 
		if int(year) > 2009:
			print "Year: "+year
			for movie in os.listdir(basedir+"/"+year):
				print "Movie: "+movie
				if "-" not in movie:
					req = requests.get("http://www.omdbapi.com/?i=tt"+movie.zfill(7))
					data = json.loads(req.text)
					#print "Data: "+str(data)
					if data['Response'] == 'True':
						print data['Title']
						print os.path.join(basedir, year, movie)
						os.rename(os.path.join(basedir, year, movie), os.path.join(basedir, year, movie+" - "+data['Title'].replace("/", "-")))