import tqdm
import random
import errno
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,TensorDataset
import math
import os
import pdb
from utils import *
import numpy as np
from sklearn.metrics import confusion_matrix

def train_one_epoch(data_loader, model, optimizer, device):
	"""
	training of one epoch
	Args:
		1. data_loader: train/valid/test DataLoader object
		2. model: model being optimized
		3. optimizer: optimizer object updtaes the parameters based on the computed gradients.
		4. device:	object representing the device on which a torch.Tensor is or will be allocated.
	"""
	model.train()
	train_loss = []
	error_accum = 0.0
	labels_all = {}
	outputs_all = {}

	for taskname in model.temporal_conv.keys():
		labels_all[taskname] = []
		outputs_all[taskname] = []
	inner = tqdm.tqdm(total=len(data_loader), desc='Batch', position=1)
	for i,(sequence,labels) in enumerate(data_loader):
		optimizer.zero_grad()
		batch_loss = 0
		sequence = sequence.to(device)
		labels = labels[0]
		outputs = model(sequence)
		# accumulate multiple task loss
		for task in (labels.keys()):
			batch_loss += F.nll_loss(outputs[task].squeeze(0),labels[task].long().to(device), reduction= 'mean')
			labels_all[task].extend(labels[task].cpu().numpy()) 
			outputs_all[task].extend((torch.max(outputs[task],dim=-1)[1]).detach().cpu().numpy())
		train_loss.append(batch_loss.item())
		error_accum = 0.95*error_accum + 0.05*batch_loss.item()
		# bacward and update parameter
		batch_loss.backward()
		optimizer.step()
		inner.update(1)
	# store labels and outputs differently depending on task.... 
	# make confusion matix for every task 
	confusion = get_confusion_matrix(labels_all,outputs_all,model.temporal_conv.keys())
	return confusion, train_loss, error_accum


def test_one_epoch(data_loader, model, device):
	"""
	testing of one epoch
	Args:
		1. data_loader: train/valid/test DataLoader object
		2. model: model being optimized
		3. device:	object representing the device on which a torch.Tensor is or will be allocated.
	"""
	model.eval()
	test_loss = []
	error_accum = 0.0
	labels_all = {}
	outputs_all = {}

	for taskname in model.temporal_conv.keys():
		labels_all[taskname] = []
		outputs_all[taskname] = []
	inner = tqdm.tqdm(total=len(data_loader), desc='Batch', position=1)
	with torch.no_grad():
		for i,(sequence,labels) in enumerate(data_loader):
			# get input and psi and labels 
			batch_loss = 0
			sequence = sequence.to(device)
			labels = labels[0]
			outputs = model(sequence)
			# accumulate multiple task loss
			for task in (labels.keys()):
				batch_loss += F.nll_loss(outputs[task].squeeze(0),labels[task].long().to(device), reduction= 'mean')
				labels_all[task].extend(labels[task].cpu().numpy()) 
				outputs_all[task].extend((torch.max(outputs[task],dim=-1)[1]).detach().cpu().numpy())
			test_loss.append(batch_loss.item())
			error_accum = 0.95*error_accum + 0.05*batch_loss.item()
			inner.update(1)
	# store labels and outputs differently depending on task.... 
	# make confusion matix for every task 
	confusion = get_confusion_matrix(labels_all,outputs_all,model.temporal_conv.keys())
	return confusion, test_loss, error_accum







