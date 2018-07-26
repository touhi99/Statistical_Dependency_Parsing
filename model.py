import numpy as np 
import pickle
import random

'''
FeatureMapper takes the unique feature and add it to the feature dictionary
'''
class FeatureMapper():
	def __init__(self):
		self.feature_map = {}
		self.id = 1
		self.frozen = False

	def feature_template(self, state, sentence):
		feature_list = np.empty((0), int)
		stack = state.stack
		buffer = state.buffer
		ld = state.ld
		rd = state.rd
		form = sentence.form
		pos = sentence.pos
		lemma = sentence.lemma
		morph = sentence.morph
		head = state.arcs

		if stack:
			s0 = stack[-1]
			''' For stack[0] element'''
			feature_list = np.append(feature_list, self.get_feature("s0_form="+form[s0]))
			feature_list = np.append(feature_list, self.get_feature("s0_pos="+pos[s0]))
			feature_list = np.append(feature_list, self.get_feature("s0_lemma="+lemma[s0]))
			feature_list = np.append(feature_list, self.get_feature("s0_form_pos="+form[s0]+pos[s0]))
			feature_list = np.append(feature_list, self.get_feature("s0_lemma_pos="+lemma[s0]+pos[s0]))
			
			'''For arc-eager, taking head, ld, rd of stack[0] elements'''
			if head[s0]>=0:
				feature_list = np.append(feature_list, self.get_feature("head_s0_form="+form[head[s0]]))
				feature_list = np.append(feature_list, self.get_feature("head_s0_pos="+pos[head[s0]]))
			if ld[s0]>=0:
				feature_list = np.append(feature_list, self.get_feature("ld_s0_form="+form[ld[s0]]))
				feature_list = np.append(feature_list, self.get_feature("ld_s0_pos="+pos[ld[s0]]))
			if rd[s0]>=0:
				feature_list = np.append(feature_list, self.get_feature("rd_s0_form="+form[rd[s0]]))
				feature_list = np.append(feature_list, self.get_feature("rd_s0_pos="+pos[rd[s0]]))
			if len(stack)>1:
				''' For stack[1] element '''
				s1 = stack[-2]
				feature_list = np.append(feature_list, self.get_feature("s1_form="+form[s1]))
				feature_list = np.append(feature_list, self.get_feature("s1_pos="+pos[s1]))
				feature_list = np.append(feature_list, self.get_feature("s1_form_pos="+form[s1]+pos[s1]))
			if morph:
				''' For German language only Morphological feature of s[0]'''
				feature_list = np.append(feature_list, self.get_feature("s0_morph="+morph[s0]))

		if buffer:
			b0 = buffer[0]
			''' For buffer[0] element '''
			feature_list = np.append(feature_list, self.get_feature("b0_form="+form[b0]))
			feature_list = np.append(feature_list, self.get_feature("b0_pos="+pos[b0]))
			feature_list = np.append(feature_list, self.get_feature("b0_lemma="+lemma[b0]))
			feature_list = np.append(feature_list, self.get_feature("b0_form_pos="+form[b0]+pos[b0]))
			feature_list = np.append(feature_list, self.get_feature("b0_lemma_pos="+lemma[b0]+pos[b0]))

			''' head, ld and rd for buffer[0] element '''
			if head[b0]>=0:
				feature_list = np.append(feature_list, self.get_feature("head_b0_form="+form[head[b0]]))
				feature_list = np.append(feature_list, self.get_feature("head_b0_pos="+pos[head[b0]]))
			if ld[b0]>=0:
				feature_list = np.append(feature_list, self.get_feature("ld_b0_form="+form[ld[b0]]))
				feature_list = np.append(feature_list, self.get_feature("ld_b0_pos="+pos[ld[b0]]))
			if rd[b0]>=0:
				feature_list = np.append(feature_list, self.get_feature("rd_b0_form="+form[rd[b0]]))
				feature_list = np.append(feature_list, self.get_feature("rd_b0_pos="+pos[rd[b0]]))

			if len(buffer)>1:
				b1 = buffer[1]
				''' buffer[1] features '''
				feature_list = np.append(feature_list, self.get_feature("b1_form="+form[b1]))
				feature_list = np.append(feature_list, self.get_feature("b1_pos="+pos[b1]))
				feature_list = np.append(feature_list, self.get_feature("b1_form_pos="+form[b1]+pos[b1]))
				feature_list = np.append(feature_list, self.get_feature("b0_form+b1_form="+form[b0]+form[b1]))
				feature_list = np.append(feature_list, self.get_feature("b0_pos+b1_pos="+pos[b0]+pos[b1]))
				if stack:
					feature_list = np.append(feature_list, self.get_feature("b0_pos+b1_pos+s0_pos="+pos[b0]+pos[b1]+pos[stack[-1]]))				
			if len(buffer)>2:
				''' buffer[2] features '''
				b2 = buffer[2]
				feature_list = np.append(feature_list, self.get_feature("b2_pos="+pos[b2]))
				feature_list = np.append(feature_list, self.get_feature("b2_form="+form[b2]))
				feature_list = np.append(feature_list, self.get_feature("b2_form_pos="+form[b2]+pos[b2]))
				feature_list = np.append(feature_list, self.get_feature("b0_pos+b1_pos+b2_pos="+pos[b0]+pos[b1]+pos[b2]))			
			if len(buffer)>3:
				''' buffer[3] POS only '''
				b3 = buffer[3]
				feature_list = np.append(feature_list, self.get_feature("b3_pos="+pos[b3]))

			if morph:
				''' Morphological feature for buffer[0] for German language '''
				feature_list = np.append(feature_list, self.get_feature("b0_morph="+morph[b0]))
				feature_list = np.append(feature_list, self.get_feature("b0_pos_morph="+pos[b0]+morph[b0]))
		if stack and buffer:
			s0 = stack[-1]
			b0 = buffer[0]
			''' Both stack and buffer '''
			feature_list = np.append(feature_list, self.get_feature("s0_form_pos+b0_form_pos="+form[s0]+pos[s0]+form[b0]+pos[b0]))
			feature_list = np.append(feature_list, self.get_feature("s0_form_pos+b0_form="+form[s0]+pos[s0]+form[b0]))
			feature_list = np.append(feature_list, self.get_feature("s0_form+b0_form_pos="+form[s0]+form[b0]+pos[b0]))
			feature_list = np.append(feature_list, self.get_feature("s0_form_pos+b0_pos="+form[s0]+pos[s0]+pos[b0]))
			feature_list = np.append(feature_list, self.get_feature("s0_pos+b0_form_pos="+pos[s0]+form[b0]+pos[b0]))
			feature_list = np.append(feature_list, self.get_feature("s0_form+b0_form="+form[s0]+form[b0]))
			feature_list = np.append(feature_list, self.get_feature("s0_pos+b0_pos="+pos[s0]+pos[b0]))
			feature_list = np.append(feature_list, self.get_feature("s0_lemma+b0_lemma="+lemma[s0]+lemma[b0]))
			
			if morph:
				feature_list = np.append(feature_list, self.get_feature("s0_b0_morph="+morph[s0]+morph[b0]))
				feature_list = np.append(feature_list, self.get_feature("s0_b0_pos_morph="+pos[s0]+pos[b0]+morph[s0]+morph[b0]))

			''' Distance between stack[0] and buffer[0] '''
			distance = str(b0 - s0)

			if head[s0]>=0:
				feature_list = np.append(feature_list, self.get_feature("s0_pos+b0_pos+hd_s0_pos="+pos[s0]+pos[b0]+pos[head[s0]]))
			if ld[b0]>=0:
				feature_list = np.append(feature_list, self.get_feature("s0_pos+b0_pos+ld_s0_pos="+pos[s0]+pos[b0]+pos[ld[s0]]))
				feature_list = np.append(feature_list, self.get_feature("s0_pos+b0_pos+ld_b0_pos="+pos[s0]+pos[b0]+pos[ld[b0]]))
			if rd[b0]>=0:
				feature_list = np.append(feature_list, self.get_feature("s0_pos+b0_pos+rd_s0_pos="+pos[s0]+pos[b0]+pos[rd[s0]]))
				feature_list = np.append(feature_list, self.get_feature("s0_pos+b0_pos+rd_b0_pos="+pos[s0]+pos[b0]+pos[rd[b0]]))

			''' combined distance of s0 and b0 features'''
			feature_list = np.append(feature_list, self.get_feature("s0_lemma+b0_lemma+dist="+lemma[s0]+lemma[b0]+distance))
			feature_list = np.append(feature_list, self.get_feature("s0_form+dist="+form[s0]+distance))
			feature_list = np.append(feature_list, self.get_feature("s0_pos+dist="+pos[s0]+distance))
			feature_list = np.append(feature_list, self.get_feature("b0_form+dist="+form[b0]+distance))
			feature_list = np.append(feature_list, self.get_feature("b0_pos+dist="+pos[b0]+distance))
			feature_list = np.append(feature_list, self.get_feature("s0_form+b0_form+dist="+form[s0]+form[b0]+distance))
			feature_list = np.append(feature_list, self.get_feature("s0_pos+b0_pos+dist="+pos[s0]+pos[b0]+distance))
		
		return feature_list


	def get_feature(self, feature):
		if self.frozen:
			if feature not in self.feature_map:
				return 0
			else:
				return self.feature_map[feature]
		else:
			if feature not in self.feature_map:
				self.feature_map[feature] = self.id
				self.id += 1
			return self.feature_map[feature]

'''
Model saves the FeatureMapper and weights
'''
class Model:
	def __init__(self, feature_map, weights):
		pass
		self.map = feature_map
		self.weights = weights

	def save_model(self, model, language):
		f = open('model_'+language, 'wb')
		pickle.dump(model, f, -1)
		f.close()

	''' this function runs an averaged perceptron '''
	def train(self, train_data):
		print("In trainer...")
		u= np.zeros(self.weights.shape, dtype=np.float32)
		q=0
		for epoch in range(0,15):
			correct=0
			print("epoch: ",epoch+1)
			j=0
			#Shuffling between each epochs
			random.shuffle(train_data, random.random)
			for data in train_data:
				q += 1
				j += 1
				scores = np.zeros((4,))
				feature_vector = np.ones((len(data.featureVector),), dtype = np.float32)

				for index in data.featureVector:
					for i in range(0,4):
						scores[i] += self.weights[i][index]
				predicted = np.argmax(scores)

				if predicted!= data.label:
					for index in data.featureVector:
						self.weights[data.label][index]+=1
						self.weights[predicted][index]-=1
						u[data.label][index]+=q
						u[predicted][index]-=q

				if predicted==data.label:
					correct+=1

				if j%5000==0:
					print("States",j,": ", (correct/j))
			print("Accuracy: ", (correct/len(train_data)))
		self.weights -= u * (1/q)
#3epochs - Average - English
#Accuracy:  0.9722499306538014
#Testing Accuracy:  0.8738860688099267

#3epochs - Normal Perceptron - English
#Accuracy:  0.9709464539174227
#Testing Accuracy:  0.8332393307012597


#15epochs - Average - English
#Accuracy:  0.9923746224592055
#Testing Accuracy:  0.8739988719684151

#15epochs - Average - German
#Accuracy:  0.9964475086687264
#Testing Accuracy:  0.8832525031289111
