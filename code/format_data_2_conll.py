#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
import re
import codecs
#from string import maketrans

## 
# author: Ahmed Abdelali
# Date : Sun Jan 15 10:52:40 AST 2017
_unicode = u"\u0622\u0624\u0626\u0628\u062a\u062c\u06af\u062e\u0630\u0632\u0634\u0636\u0638\u063a\u0640\u0642\u0644\u0646\u0648\u064a\u064c\u064e\u0650\u0652\u0670\u067e\u0686\u0621\u0623\u0625\u06a4\u0627\u0629\u062b\u062d\u062f\u0631\u0633\u0635\u0637\u0639\u0641\u0643\u0645\u0647\u0649\u064b\u064d\u064f\u0651\u0671"
_buckwalter = u"|&}btjGx*z$DZg_qlnwyNaio`PJ'><VApvHdrsSTEfkmhYFKu~{"
_safebuckwalter = u"MWQbtjGxVzcDZg_qlnwyNaio`PJCOIVApvHdrsSTEfkmhYFKu~{"

_forwardMap = {ord(a):b for a,b in zip(_unicode, _buckwalter)}
_backwardMap = {ord(b):a for a,b in zip(_unicode, _buckwalter)}

_safeforwardMap = {ord(a):b for a,b in zip(_unicode, _safebuckwalter)}
_safebackwardMap = {ord(b):a for a,b in zip(_unicode, _safebuckwalter)}

def trans(to_translate):
	tabin = u'привет'
	tabout = u'тевирп'
	tabin = [ord(char) for char in tabin]
	translate_table = dict(zip(tabin, tabout))
	return to_translate.translate(translate_table)

def toBuckWalter(s):
	return s.translate(_forwardMap)

def fromSafeBuckWalter(s):
	#print("in:'"+s+"'")
	s_out = ''
	for i in s:
		if(i.decode('unicode-escape') in _safebuckwalter):
			#
			#print(type(_backwardMap))
			#print((u''+i).translate(_backwardMap))
			s_out = s_out + (u''+i).translate(_safebackwardMap)
		else:
			s_out = s_out + i
	#return trans('привет')
	#print("out"+str(s.translate(transtab)))

	return s_out

def fromBuckWalter(s):
	#print("in:'"+s+"'")
	s_out = ''
	for i in s:
		if(i.decode('unicode-escape') in _buckwalter):
			#
			#print(type(_backwardMap))
			#print((u''+i).translate(_backwardMap))
			s_out = s_out + (u''+i).translate(_backwardMap)
		else:
			s_out = s_out + i
	#return trans('привет')
	#print("out"+str(s.translate(transtab)))

	return s_out


def main(infile):
	fp = codecs.open(infile, 'r','utf-8')
	word =''
	for line in fp:
		#
		#print("-->"+line.strip())
		if('\t' in line):
			(c,t)=line.strip().split('\t')
			#print("==>"+c+":"+t)
			#c=fromBuckWalter(c)
			if(t=='B'):
				#print("NW:"+word+" "),
				word = word+('+' if(len(word)>0) else '')+c
			elif(t=='M'):
				word = word+c
			elif(t=='E'):
				word = word+c
			elif(t=='WB'):
				#print(word)
				print(fromSafeBuckWalter(str(word)))
				word = ''
			elif (t=='S'):
				word = word+('+' if(len(word)>0) else '')+c
		else:
			print('')

	fp.close()

def convertTrainingData(infile):
	fp = codecs.open(infile, 'r','utf-8')
	for line in fp:
		line = line.strip()
		#print('In:',line)
		if('<EOTWEET>' in line): #EOS
			print('')
			continue
		for elt in line.split('+'):
			if(len(elt)==1):
				print(elt+'\t'+'S')
			elif(len(elt)>=2):
				print(elt[0]+'\t'+'B')
				for i in range(1,len(elt)-1):
					print(elt[i]+'\tM')
				print(elt[-1]+'\t'+'E')
		print('WB\tWB')
	fp.close()

if __name__ == '__main__':
    #main(sys.argv[1])
    convertTrainingData(sys.argv[1])
    #a = fromBuckWalter(u'albustan aljameel')
    #print(a)

