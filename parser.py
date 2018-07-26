'''
Author: Touhidul Alam
'''

import sys
from collections import deque
from reader import Corpus
import numpy as np
from model import FeatureMapper
from model import Model
import pickle

'''
State class initialize initial stack, buffer, arcs, left-dependency and right-dependency
'''
class State:
	def __init__(self, stack, buffer):
		self.stack = deque(stack)
		self.buffer = deque(buffer)
		self.arcs = np.empty(len(self.buffer)+len(self.stack),dtype=np.int32)
		self.arcs.fill(-1)
		self.ld = np.empty(len(self.buffer)+len(self.stack),dtype=np.int32)
		self.rd = np.empty(len(self.buffer)+len(self.stack),dtype=np.int32)
		self.ld.fill(-1)
		self.rd.fill(-1)

	def get_dependency(self, head):
		dep_list = []
		for i in range(0,len(self.arcs)):
			if self.arcs[i]==head:
				dep_list.append(i)
		return dep_list

'''
Instance class saves the transition label with its corresponding features
'''
class Instance:
	def __init__(self, label, featureVector):
		self.label = label
		self.featureVector = featureVector

'''
Parser class does oracle parsing during training and predicting and making a parser
'''
class Parser:
	def __init__(self, state, sentence, feature_map):
		self.sentence = sentence
		self.state = state
		self.map = feature_map
		self.data = []

	'''Oracle parsing done after extracting feature in each transition '''
	def oracle(self):
		train_data = []
		while self.state.buffer:
			feature_list = np.asarray(self.map.feature_template(self.state, self.sentence))
			if self.state.stack and self.should_left_arc():
				self.left_arc()
				instance = Instance(0, feature_list)
			elif self.state.stack and self.should_right_arc():
				self.right_arc()
				instance = Instance(1, feature_list)
			elif self.state.stack and self.should_reduce():
				self.reduce()
				instance = Instance(2, feature_list)
			else:
				self.shift()
				instance = Instance(3, feature_list)
			train_data.append(instance)
		self.data = train_data
		return self.data

	'''Parse functions predict each transitions'''
	def parse(self, loaded_model):
		weights = loaded_model.weights
		while self.state.buffer:
			feature_list = self.map.feature_template(self.state, self.sentence)

			scores = np.zeros((4,))

			for index in feature_list:
				for i in range(0,4):
					scores[i] += weights[i][index]

			'''np.argsort, sort array, - means with high-to-low'''
			predicted = np.argsort(-scores)
			for item in predicted:
				if item==0 and self.state.stack and self.can_left_arc():
					self.left_arc()
					break
				elif item==1 and self.state.stack:
					self.right_arc()
					break
				elif item==2 and self.state.stack and self.can_reduce():
					self.reduce()
					break
				elif item==3 and self.state.stack:
					self.shift()
					break
		''' Add to left-neighbour if headless word found, except first element'''
		for i in range(0, len(self.state.arcs)):
			if i==0:
				continue
			else:
				if self.state.arcs[i]==-1:
					self.state.arcs[i] = i-1

		return self.state.arcs

	def print_transition(self, pred):
		print(pred, self.state.stack, self.state.buffer, self.state.arcs)

	'''
	should_XX() , checks the oracle condition before executing
	'''

	def should_left_arc(self):
		result = False
		stack_top = self.state.stack[-1]
		buff_front = self.state.buffer[0]
		if (buff_front, stack_top) in self.sentence.gold_arcs:
			return True
		return result

	def should_right_arc(self):
		result = False
		stack_top = self.state.stack[-1]
		buff_front = self.state.buffer[0]
		if (stack_top, buff_front) in self.sentence.gold_arcs:
			return True
		return result

	def should_reduce(self):
		result = False
		stack_top = self.state.stack[-1]
		buff_front = self.state.buffer[0]
		if self.has_head(stack_top) and self.has_all_children(stack_top):
			return True
		return result


	'''
	can_XX() checks the precondition of a transition before parsing
	'''

	def can_left_arc(self):
		result = False
		stack_top = self.state.stack[-1]
		if stack_top!=0 and self.state.arcs[stack_top]==-1:
			return True
		return result

	def can_right_arc(self):
		pass

	def can_reduce(self):
		result = False
		stack_top = self.state.stack[-1]
		if self.state.arcs[stack_top]!= -1:
			return True
		return result

	def can_shift(self):
		result = False
		if len(self.state.buffer)>=1 or self.state.stack:
			return True
		return result


	'''hasHead and hasAllChildren condition before checking reduce'''

	def has_head(self, stack_top):
		count=0
		if self.state.arcs[stack_top]!=-1:
			count+=1

		if count<1:
			return False
		else:
			return True
	def has_all_children(self, stack_top):
		for gold_arcs in self.sentence.gold_arcs:
			head = gold_arcs[0]
			dep = gold_arcs[1]
			if head==stack_top:
				if self.state.arcs[dep] != stack_top:
					return False
		return True

	'''
	do_XX() execute the transition
	'''

	def left_arc(self):
		last_stack = self.state.stack.pop()
		self.state.arcs[last_stack] = self.state.buffer[0]
		self.state.ld[self.state.buffer[0]] = min(self.state.get_dependency(self.state.buffer[0]))
		self.state.rd[self.state.buffer[0]] = max(self.state.get_dependency(self.state.buffer[0]))

	def right_arc(self):
		last_stack = self.state.stack[-1]
		self.state.arcs[self.state.buffer[0]] = last_stack
		self.state.ld[last_stack] = min(self.state.get_dependency(last_stack))
		self.state.rd[last_stack] = max(self.state.get_dependency(last_stack))

		self.state.stack.append(self.state.buffer[0])
		self.state.buffer.popleft()

	def reduce(self):
		self.state.stack.pop()

	def shift(self):
		last_buffer = self.state.buffer.popleft()
		self.state.stack.append(last_buffer)


def load_model(language):
	f = open('model_'+language, 'rb')
	model = pickle.load(f)
	f.close()
	return model

if __name__ == "__main__":
	'''
	Arc-eager Transition-based parser with Averaged-perceptron

	USAGE: python3 parser.py [train/test/dev] [en/de] dataset_file_location
	'''
	model_type = sys.argv[1]
	language = sys.argv[2]
	file_path = sys.argv[3]
	init_stack = np.arange(1)
	feature_map = FeatureMapper()

	'''
	TRAIN----
	'''
	if model_type=='train':
		sentences = Corpus(file_path, model_type, language).sentences
		train_data = []
		for sentence in sentences:
			init_buffer = np.arange(1, len(sentence.form))
			state = State(init_stack, init_buffer)
			parser_data = Parser(state, sentence, feature_map).oracle()
			train_data.append(parser_data)
		train_data = np.concatenate(train_data).flatten().tolist()
		feature_map.frozen = True
		weights = np.zeros((4, feature_map.id), dtype=np.float32)
		print(weights.shape)
		model = Model(feature_map, weights)
		model.train(train_data)
		model.save_model(model, language)
	
	'''
	DEV----
	'''
	if model_type=='dev':
		sentences = Corpus(file_path, model_type, language).sentences
		loaded_model = load_model(language)
		i=0
		j=0
		for sentence in sentences:
			init_buffer = np.arange(1, len(sentence.form))
			state = State(init_stack, init_buffer)
			arcs = Parser(state, sentence, loaded_model.map).parse(loaded_model)
			
			'''checking each word wise arc comparisong, first word is root, which is ommitted'''
			for gold_arc, cur_arc in zip(sentence.head[1:], arcs[1:]):
				if cur_arc==gold_arc:
					j+=1
				i+=1
		print("Dev Accuracy: ",str(j/i))

	'''
	TEST----
	'''
	if model_type=='test':
		sentences = Corpus(file_path, model_type, language).sentences
		loaded_model = load_model(language)
		outfile = open('pred_'+language+'.conll06','w')
		i=0
		j=0
		for sentence in sentences:
			init_buffer = np.arange(1, len(sentence.form))
			state = State(init_stack, init_buffer)
			arcs = Parser(state, sentence, loaded_model.map).parse(loaded_model)
			
			for arc in range(1,len(arcs)):
				''' In german, we have extra morph line to be added'''
				if language=='en':
					outfile.write(str(arc)+"\t"+sentence.form[arc]+"\t"+sentence.lemma[arc]+"\t"+sentence.pos[arc]+"\t"+"_"+"\t"+"_"+"\t"+str(arcs[arc])+"\t"+"_"+"\t"+"_"+"\t"+"_")
				else:
					outfile.write(str(arc)+"\t"+sentence.form[arc]+"\t"+sentence.lemma[arc]+"\t"+sentence.pos[arc]+"\t"+"_"+"\t"+sentence.morph[arc]+"\t"+str(arcs[arc])+"\t"+"_"+"\t"+"_"+"\t"+"_")
				outfile.write('\n')
			outfile.write('\n')
		outfile.close()
	