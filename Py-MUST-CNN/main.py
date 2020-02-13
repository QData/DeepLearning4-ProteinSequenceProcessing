# author: Arshdeep Sekhon
import argparse
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
from data_loading import *
from models import *
from train_fxns import *
from trainer import *
import logging




def my_collate(batch):
	"""
	custom collate function
	"""
	data = [item[0] for item in batch]
	data = torch.FloatTensor(np.concatenate(data,axis=0))
	target = [item[1] for item in batch]
	return data, target



def main():
	parser = argparse.ArgumentParser('MUST CNN.')
	parser.add_argument('--nocuda',action='store_true',help = 'cuda support' )
	parser.add_argument('--use_psi_features',action='store_true',help = 'use psi features' )
	parser.add_argument('--finetune',action='store_true',help = 'if finetuning from a pretrained shared model' )
	parser.add_argument('--save_every_epoch',action='store_true',help = 'logging for every epoch' )
	parser.add_argument('--gpu-id',type = int, default = 0, help='gpu id to use')
	parser.add_argument('--seed',type = int, default = 0, help='random seed')
	parser.add_argument('--dictsize',type = int, default = 23, help='input features dict size(23 for protein sequences)')
	parser.add_argument('--task',nargs='+', default = 'absolute',help = 'Task name')
	parser.add_argument('--loaddir',type = str, default = './data',help = 'load data from')
	parser.add_argument('--finetune-task',type = str, default = 'absolute',help = 'Task name')
	parser.add_argument('--savedir',type = str, default = 'LOGS/',help = 'save results')
	parser.add_argument('--optimizer',type = str, default = 'sgd',help = 'optimizer to train model')
	parser.add_argument('--lr',type = float, default = 0.001,help = 'learning rate')
	parser.add_argument('--convdropout',type = float, default = 0.0,help = 'dropout after cnn layer')
	parser.add_argument('--input_dropout',type = float, default = 0.35,help = 'dropout after lookup table')
	parser.add_argument('--embedsize',type = int, default = 15,help = 'embedding size for lookup table')
	parser.add_argument('--kernelsize',nargs='+', default = 9,help = 'size of kernel')
	parser.add_argument('--hiddenunit',type = int, default = 189,help = 'hidden units in conv')
	parser.add_argument('--poolingsize',type = int, default = 2,help = 'pooling kernel size')
	parser.add_argument('--batchsize',type = int, default = 1,help = 'batch size')
	parser.add_argument('--epochs',type = int, default = 300,help = 'epochs')
	parser.add_argument('--nlayers',type = int, default = 3,help = 'cnn layers')
	parser.add_argument('--nonlinearity',type = str, default = 'relu',help = 'relu/prelu/tanh')

	params = parser.parse_args()
		# necessary initalization	and device set up....
	random.seed(params.seed)
	np.random.seed(params.seed)
	params.kernelsize = [int(i) for i in params.kernelsize]
	params.cuda = not params.nocuda and torch.cuda.is_available()
	device = torch.device("cuda" if params.cuda else "cpu")
	torch.cuda.set_device(params.gpu_id)
	torch.manual_seed(params.seed)
	torch.cuda.manual_seed_all(params.seed)
	embedsize = params.embedsize
	kernelsize = params.kernelsize
	hiddenunit = params.hiddenunit
	poolingsize = params.poolingsize
	alltask = params.task



# get multi task dataset -- multiple datasets with multiple tasks
	params.sequence_task_dict = {}
	params.task_class_dict = {}
	list_of_tasks = ' '.join(alltask).split(' ')
	for task in list_of_tasks:
		dataset_name = task.split('.')[0]
		if dataset_name not in params.sequence_task_dict.keys():
			params.sequence_task_dict[dataset_name] = {}
		if len(task.split('.')) != 1:
			hashfile = os.path.join(params.loaddir,dataset_name, 'hash/' + task.split('.')[1] + '.lab.tag.lst')	 
			params.sequence_task_dict[dataset_name][task.split('.')[1]] = len(open(hashfile).readlines()) 
# change dots to underscore as module dict cant take dot names 
			params.task_class_dict['_'.join(task.split('.'))] = len(open(hashfile).readlines())
	if not params.finetune:
		params.current_task_dict = params.sequence_task_dict.copy()
	else:
		params.current_task_dict = {}
		params.current_task_dict[params.finetune_task.split('.')[0]] = {}
		params.current_task_dict[params.finetune_task.split('.')[0]][params.finetune_task.split('.')[1]] = params.sequence_task_dict[params.finetune_task.split('.')[0]][params.finetune_task.split('.')[1]]





	# data loaders
	# use own collate function as variable sized length sequences
	train_dataset, valid_dataset, test_dataset	= load_data(params.loaddir,params.current_task_dict,use_psi_features = params.use_psi_features)
	train_loader=DataLoader(train_dataset,shuffle=True,batch_size=params.batchsize,collate_fn = my_collate)
	valid_loader=DataLoader(valid_dataset,shuffle=False,batch_size=params.batchsize,collate_fn= my_collate)
	test_loader=DataLoader(test_dataset,shuffle=False,batch_size=params.batchsize,collate_fn= my_collate)



		# initialize MUST CNN
	print(params)
	model = MUSTCNN(params).to(device)

		# initalize opimizer 
	if params.optimizer == 'adam':
		optimizer = torch.optim.Adam(list(model.parameters()),lr=params.lr)
	else:
		optimizer = torch.optim.SGD(list(model.parameters()),lr=params.lr,momentum=0.9) 


	savedir = params.savedir+('_').join(params.task)+'/' + str(params.hiddenunit)+'/'
	os.makedirs(savedir, exist_ok=True) 

		# savedir to save fintuned model on single task... nested under
		# multitask folder
	if params.finetune:
		model.load_state_dict(torch.load(savedir+'best_model.pt'))
		model.reinitialize()
		model = model.to(device)
		savedir = savedir + params.finetune_task +'/'
		os.makedirs(savedir, exist_ok=True)

	set_logger(os.path.join(savedir+'/logs.txt'))
	logging.info('starting training.....')
		# training wrapper
	log_dict, valid_log_dict, test_log_dict = train(savedir, train_loader, valid_loader, test_loader, model,\
			optimizer, params.epochs, device, params.save_every_epoch)

		# save the final epochs data...
	logging.shutdown()
	with open(savedir+'final_train_log', 'wb') as f:
		pickle.dump(log_dict, f, pickle.HIGHEST_PROTOCOL)

	with open(savedir+'final_valid_log', 'wb') as f:
		pickle.dump(valid_log_dict, f, pickle.HIGHEST_PROTOCOL)
	with open(savedir+'final_test_log', 'wb') as f:
		pickle.dump(test_log_dict, f, pickle.HIGHEST_PROTOCOL)
	previous_stats = ['train_log','valid_log','test_log']
	for stat_dict in previous_stats:
		try:
			os.remove(savedir+stat_dict)
		except:
			print("couldn't delete : ",stat_dict)

























if __name__ == '__main__':
	main()
