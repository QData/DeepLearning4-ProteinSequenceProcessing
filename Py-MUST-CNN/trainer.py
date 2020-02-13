import tqdm
import random
import errno
import pickle
import time
import torch
import torch.nn as nn
import math
import os
import pdb
#from utils import *
import numpy as np
from sklearn.metrics import confusion_matrix
from train_fxns import *

def train(savedir, train_loader, valid_loader, test_loader, model, optimizer, epochs, device, save_every_epoch):
	"""
	Train wrapper that trains, validation and tests on given dataset	over multiple epochs. 
	Args:
		1. savedir: directory where best models and stats are saved
		2. train_loader, 3. valid_loader, 4. test_loader are DataLoader objects from train.valid, test respectively
		5. model to train
		6. optimizer : optimizer object, that will update the model parameters based on the computed gradients.
		7. epochs: number of epochs to train
		8. device
		9. save_every_epoch: flag to indicate level of logs recorded
	"""

	log_dict = {'time':[],'confusion':[],'loss':[],'error_accum':[]}
	valid_log_dict = {'time':[],'confusion':[],'loss':[],'error_accum':[]}
	best_loss = 100000000.0
	test_log_dict = {'time':[],'confusion':[],'loss':[],'error_accum':[]}
	outer = tqdm.tqdm(total=epochs, desc='Epoch', position=0)
	for epoch in range(epochs):
		logging.info(f'currently on epoch.....{epoch}')
		start = time.time()

		logging.info('training.......')
		confusion, train_loss, train_error_accum	= train_one_epoch(train_loader, model, optimizer, device)
		end = time.time()
		log_dict['time'].append(end-start)
		log_dict['confusion'].append(confusion)
		log_dict['loss'].append(np.mean(train_loss))
		log_dict['error_accum'].append(train_error_accum)
		start = time.time()
		logging.info('validation.......')
		valid_confusion, valid_loss, valid_error_accum = test_one_epoch(valid_loader, model, device)
		end = time.time()
		valid_log_dict['time'].append(end-start)
		valid_log_dict['confusion'].append(valid_confusion)
		valid_log_dict['loss'].append(np.mean(valid_loss))
		valid_log_dict['error_accum'].append(np.mean(valid_error_accum))
		start = time.time()
		logging.info('testing.....')
		test_confusion, test_loss, test_error_accum = test_one_epoch(test_loader, model, device)
		end = time.time()
		test_log_dict['time'].append(end-start)
		test_log_dict['confusion'].append(test_confusion)
		test_log_dict['loss'].append(np.mean(test_loss))
		test_log_dict['error_accum'].append(test_error_accum)
		outer.update(1)
		curr_train_loss = log_dict['loss'][-1] 
		curr_valid_loss = valid_log_dict['loss'][-1]	
		logging.info(f'stats at epoch {epoch} training loss:{curr_train_loss}, validation_loss: {curr_valid_loss}')
		if valid_log_dict['loss'][-1]<best_loss:
				best_loss = np.mean(valid_loss)
				torch.save(model.state_dict(),savedir+'best_model.pt')
				curr_train_loss = log_dict['loss'][-1]
				logging.info(f'saved best model at {epoch} training loss:{curr_train_loss}, validation_loss: {best_loss}')
				with open(savedir+'best_train_log', 'wb') as f:
					pickle.dump(log_dict, f, pickle.HIGHEST_PROTOCOL)

				with open(savedir+'best_valid_log', 'wb') as f:
					pickle.dump(valid_log_dict, f, pickle.HIGHEST_PROTOCOL)
				with open(savedir+'best_test_log', 'wb') as f:
					pickle.dump(test_log_dict, f, pickle.HIGHEST_PROTOCOL)
		# save logs and models every epoch
		if save_every_epoch:
			torch.save(model.state_dict(),savedir+'epoch_'+str(epoch)+'_model.pt')
			with open(savedir+'train_log', 'wb') as f:
				pickle.dump(log_dict, f, pickle.HIGHEST_PROTOCOL)

			with open(savedir+'valid_log', 'wb') as f:
				pickle.dump(valid_log_dict, f, pickle.HIGHEST_PROTOCOL)
			with open(savedir+'test_log', 'wb') as f:
				pickle.dump(test_log_dict, f, pickle.HIGHEST_PROTOCOL)
	return log_dict, valid_log_dict, test_log_dict





