
import random
import errno
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,TensorDataset, ConcatDataset
import math
import os
import pdb
from utils import *
import numpy as np
from random import shuffle


class SequenceLabelingDataSet(Dataset):
	"""
	Sequence Dataset
	"""
	def __init__(self,dataset, labels):
		"""
		Args:
			dataset: list of sequences represented in numeric format
			labels: labels is a dict representing task and corresponding label per amino acid
		"""
		self.dataset = dataset
		self.labels = labels
	def __len__(self):
		return len(self.dataset)

	def __getitem__(self,ix):
		x = torch.FloatTensor(self.dataset[ix])
# so that dict starts from zero, for nn.Embedding
		x[:,0] = x[:,0]-1
		y = {}
		for taskno,task in enumerate((self.labels.keys())):
			y[task] = (torch.LongTensor(self.labels[task][ix])-1)

		return x,y

def read_data(datafile):
	"""
	Args:
		datafile : this is a file containing an input sequence on each line 
	"""
	seqs = {}
	datasetfile = open(datafile,'r')
	dataset = datasetfile.readlines()
	new_dataset = []
	for seq in dataset:
		new_dataset.append([float(i) for i in seq.strip(' \n').split(' ')])
	return new_dataset



def preprocess_data(loaddir, dataset, use_psi_features):
	"""
	Args:
		1. 
	"""
	iput = read_data(os.path.join(loaddir,dataset,'data/seq.dat')) 
	final_dataset = []
	labels_dict = {}
	tasks = []

	psi_features = [np.expand_dims(np.array(read_data(os.path.join(loaddir,dataset,'data/','psi'+str(i+1)+'.dat.NR'))),axis=1) for i in range(20)]
	for seq in range(len(iput)):
		to_cnc = []
		to_cnc.append(np.expand_dims(np.array(iput[seq]),axis=1))
		to_cnc.extend([np.expand_dims(psi[seq][0],axis=1) for psi in psi_features])
		to_cnc = (np.concatenate(to_cnc,axis=1))
		final_dataset.append(to_cnc)

	return final_dataset





def load_data(loaddir, dataset_tasks, use_psi_features = False):
	
	# tasks is a dict with each entry indexed by the dataset a dict containing info about task_name and number of classes 
	train_data, valid_data, test_data = [],[],[]
	for dataset in dataset_tasks.keys():
		input_sequences = preprocess_data(loaddir, dataset, use_psi_features)

		labels_dict = {}


	# get label dictionary.... for all tasks being considered for current setting
		for i,task in enumerate(dataset_tasks[dataset].keys()):
			labels_dict[task] = read_data(os.path.join(loaddir,dataset,'data/',task+'.lab.tag.dat'))

	# load psi features together 
		nsequences = len(input_sequences)
		ids = [i for i in range(nsequences)]
		#shuffle(ids)
		input_sequences = [input_sequences[i] for i in ids]
		for i,task in enumerate(dataset_tasks[dataset].keys()):
			labels_dict[task] = [labels_dict[task][i] for i in ids]	
		nvalid = math.floor(0.8*nsequences) - math.floor(0.6*nsequences)
		ntrain = math.floor(0.6*nsequences)
		ntest = nsequences - (math.floor(0.8*nsequences))
		train_labels_dict = {}
		for task in labels_dict.keys():
			train_labels_dict['_'.join([dataset,task])] = labels_dict[task][0:ntrain]

		valid_labels_dict = {}
		for task in labels_dict.keys():
			valid_labels_dict['_'.join([dataset,task])] = labels_dict[task][ntrain:(ntrain+nvalid)]
		test_labels_dict = {}
		for task in labels_dict.keys():
			test_labels_dict['_'.join([dataset,task])] = labels_dict[task][(ntrain+nvalid):]

		train_data.extend([SequenceLabelingDataSet(input_sequences[0:ntrain],train_labels_dict)])
		valid_data.extend([SequenceLabelingDataSet(input_sequences[ntrain:(ntrain+nvalid)],valid_labels_dict)])
		test_data.extend([SequenceLabelingDataSet(input_sequences[(ntrain+nvalid):],test_labels_dict)])


	train_data = ConcatDataset(train_data)
	valid_data = ConcatDataset(valid_data)
	test_data = ConcatDataset(test_data)

	return train_data, valid_data, test_data


