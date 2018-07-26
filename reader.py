import sys

'''
Corpus class to read sentences from dataset and build them into sentence structure
'''
class Corpus:
	def __init__(self, path, model_type, language):
		self.path = path #path, takes the dataset file
		self.type = model_type #model_type, either train/dev/test
		self.language = language # en or de
		self.sentences = []
		self.sentence_add(self.path)

	def sentence_add(self, path):
		i=0
		f = open(path)
		sentence_ob = Sentence()
		for content in f:
			if content!="\n":
				element = content.split("\t")
				if element[0]=="1":
					sentence_ob.form.append('ROOT')
					sentence_ob.lemma.append('ROOT')
					sentence_ob.pos.append('ROOT_POS')
					if self.type!='test':
						''' Add gold set head only on train/dev'''
						sentence_ob.head.append(-1)
						#sentence_ob.rel.append('ROOT_REL')
					if self.language=='de':
						''' Add german morphological feature '''
						sentence_ob.morph.append('ROOT_MORPH')
				sentence_ob.form.append(element[1])
				sentence_ob.lemma.append(element[2])
				sentence_ob.pos.append(element[3])
				if self.language=='de':
					sentence_ob.morph.append(element[5])
				if self.type!='test':
					sentence_ob.head.append(int(element[6]))
					sentence_ob.gold_arcs.append(( int(element[6]), int(element[0]) ))
				#sentence_ob.rel.append(element[7])
			else:
				'''
				i+=1
				if i==10:
					break
				'''
				self.sentences.append(sentence_ob)
				sentence_ob = Sentence()

		#print(self.sentences[0].gold_arcs)


class Sentence:
	def __init__(self):
		#self.id = id
		self.form = []
		self.lemma = []
		self.pos = []
		self.morph = []
		self.head = []
		#self.rel = []
		self.gold_arcs = []
