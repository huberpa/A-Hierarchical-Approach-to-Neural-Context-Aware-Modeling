import xml.etree.ElementTree as ET

concatenatedText = ""
tree = ET.parse('./../tedData/ted.xml')
root = tree.getroot()
for child in root:
	concatenatedText += child.find('content').text.encode('utf-8')

f = open("./../tedData/allTEDTalkContent.txt","w")
f.write(concatenatedText)
f.close()