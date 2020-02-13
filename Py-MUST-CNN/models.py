import random
import errno
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,TensorDataset
import math
import os
import pdb
from utils import *
import numpy as np
import math
nltbl = {'tanh' : nn.Tanh(),'relu': nn.ReLU(),'prelu': nn.PReLU()}


class conv_base_model(nn.Module):
	# base conv model-- pads, conv,nonlinear,maxpool --- this is one conv
	# layer
	def __init__(self, embedsize , kernelsize, hiddenunit, poolingsize, nonlinearity, convdropout):
		super().__init__()
		self.hiddenunit = hiddenunit
		self.poolingsize = poolingsize
		self.kernelsize = kernelsize
		self.embedsize = embedsize
		self.padding = int((self.kernelsize-1)/2)
		self.nonlinearity = nltbl[nonlinearity]
		self.convdropout = convdropout
		self.conv1d = nn.Conv1d(self.embedsize, self.hiddenunit, self.kernelsize)
		self.maxpool = nn.MaxPool1d(self.poolingsize)



	def forward(self, iput):
		# pad sequence 
# then duplicate padded with pooling 

		iput =\
		torch.cat(([F.pad(iput,(0,0,self.padding-j,self.kernelsize-self.padding-1+j+self.poolingsize))\
			for j in range(self.poolingsize)]),dim=0)
		iput = iput.permute(0,2,1)
		# convolution 1D
		output = self.conv1d(iput)
		output = self.nonlinearity(output)
		output = self.maxpool(output)
		output = F.dropout(output,p=self.convdropout,training=self.training)
		output = output.permute(0,2,1)
		return output





class MUSTCNN(nn.Module):
	def __init__(self,params):
		super().__init__()
		self.embedsize = params.embedsize
		self.kernelsize = params.kernelsize
		self.poolingsize = params.poolingsize
		self.input_dropout = params.input_dropout
		self.hiddenunit = params.hiddenunit
		self.lookuptable = nn.Embedding(params.dictsize,self.embedsize)
		self.finetune = params.finetune
		self.finetune_task = params.finetune_task
		self.use_psi_features = params.use_psi_features
		self.task_class_dict = params.task_class_dict
		if self.use_psi_features:
			extend_psi = 20
		else:
			extend_psi = 0
		self.conv_base = conv_base_model( params.embedsize + extend_psi , params.kernelsize[0], self.hiddenunit, params.poolingsize, params.nonlinearity, params.convdropout)
		conv_layers = []
		self.nlayers = params.nlayers
		for i in range(1,self.nlayers):
				conv_layers.append(conv_base_model(self.hiddenunit , params.kernelsize[i], self.hiddenunit, params.poolingsize, params.nonlinearity, params.convdropout))
		self.conv_layers = nn.Sequential(*conv_layers)

		
		# linear layers for different tasks 1kernel size is basically linear layer 
		self.temporal_conv = nn.ModuleDict()
		for taskno,task in enumerate((self.task_class_dict.keys())):
			self.temporal_conv[task] = nn.Conv1d(self.hiddenunit,self.task_class_dict[task], 1)

		'''
		for task in (self.temporal_conv.keys()):
						self.add_module('predictor_{}'.format(task), self.temporal_conv[task])
		'''
	def reinitialize(self):


		self.temporal_conv = nn.ModuleDict()
		for taskno,task in enumerate((self.task_class_dict.keys())):
			self.temporal_conv[task] = nn.Conv1d(self.hiddenunit,self.task_class_dict[task], 1)

	def forward(self, sequences):
		seq_len = sequences.size()[0]
		output = self.lookuptable(sequences[:,0].long())
		output = F.dropout(output,p=self.input_dropout,training=self.training)
		output = torch.cat((output,sequences[:,1:]),dim=1).unsqueeze(0)
		# pad,conv,stitch
		output = self.conv_base(output)
		if self.nlayers > 1: output = self.conv_layers(output)
		# stitch together sequence
		L,P,H = output.size()
		output = \
		(output.permute(2,1,0).contiguous().view(H,P*L,1))
		output = output.permute(2,0,1)
		outputs = {}
		# predict and make output same size as input
		for taskname in self.temporal_conv.keys():
			outputs[taskname]=F.log_softmax(self.temporal_conv[taskname](output).squeeze(0).t(),dim=-1)[0:seq_len,:]
		return outputs



