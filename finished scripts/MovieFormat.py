from xml.etree import ElementTree

with open('./../../SubtitleData/raw/en/2007/382932 - Ratatouille/4907298.xml', 'rt') as f:
    tree = ElementTree.parse(f)

text = "";
for node in tree.iter('s'):
    text += node.text

print "".join([s for s in text.strip().splitlines(True) if s.strip()]).replace("- ", "").replace("\n", " ")
