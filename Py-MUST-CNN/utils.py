
import random
import errno
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,TensorDataset
import math
import os
import pdb
import numpy as np
from sklearn.metrics import confusion_matrix
import logging

def get_confusion_matrix(dict_of_labels, dict_of_outputs, task_dict):
		"""
		Args:
			dict_of_labels: dict[task] stores truth for task(= seqlen) for each sequence
			dict_of_outputs: dict[task] stores predictions for task(= seqlen) for each sequence
			task_dict: dict storing specific tasknames 


	 	"""
		ntasks = len(task_dict)
		confusion = {}
		for task in task_dict:
				confusion[task] = confusion_matrix(dict_of_labels[task],dict_of_outputs[task])
		return confusion




def set_logger(log_path):
		"""
		log_path is a path to a file 

		"""

		logger	= logging.getLogger()
		logger.setLevel(logging.INFO)
		if not logger.handlers:
				# Logging to a file
				file_handler = logging.FileHandler(log_path,mode='w')
				file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
				logger.addHandler(file_handler)

				# Logging to console
				stream_handler = logging.StreamHandler()
				stream_handler.setFormatter(logging.Formatter('%(message)s'))
				logger.addHandler(stream_handler)

