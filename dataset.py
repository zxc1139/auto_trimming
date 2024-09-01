import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from keras.utils.data_utils import pad_sequences
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class FPS_Dataset(Dataset):
    def __init__(self, trimming_dir: Path,  whole_dir: Path, train_model = True, trained_max_len = None):
        self.trimming_dir = trimming_dir
        self.whole_dir = whole_dir

        self.start_time = []
        self.file_name = []
        fps = []
        session_whole_time = []
        for f in glob.glob(os.path.join(whole_dir, "*.csv")):
            session_df = pd.read_csv(f)
            session_start_time = session_df["time"].iat[0]
            session_end_time = session_df["time"].iat[-1]
            if (session_end_time - session_start_time) != (len(session_df) - 1):
                session_df["relative_time"] = session_df["time"] - session_start_time
                print(session_df["relative_time"].iat[-1])
                print(f)
    
            else: 
                session_df["relative_time"] = session_df["time"] - session_start_time
            self.start_time.append(session_df["time"].iat[0])
            self.file_name.append(f)
            session_whole_time.append(session_df["time"])
            fps.append(session_df["FPS"])

        seq_max_len = max(len(x) for x in session_whole_time)
        if seq_max_len % 2 == 1: 
             seq_max_len += 1

        self.fps_pad = pad_sequences(fps, maxlen = seq_max_len, padding='post')
        if not train_model:
            self.fps_pad = pad_sequences(fps, maxlen = trained_max_len, padding='post')

        trimmed_from_time = []
        trimmed_to_time = []
        trimmed_len  = []

        for f in glob.glob(os.path.join(trimming_dir, "*.csv")):
            trimming_df = pd.read_csv(f)
            trimmed_from_time.append(trimming_df["time"].iat[0])
            trimmed_to_time.append(trimming_df["time"].iat[-1])
            trimmed_len.append(len(trimming_df))
    
        trimming_idx = [[0] * len(s) for s in session_whole_time]
        for start_t, end_t, whole_s, idx_lst in zip(trimmed_from_time, trimmed_to_time, session_whole_time, trimming_idx):
            for i, (t, idx) in enumerate(zip(whole_s, idx_lst)):
                if t >= start_t and t <= end_t + 1:
                    idx_lst[i] = 1


        self.trimming_idx_pad = pad_sequences(trimming_idx, maxlen = seq_max_len, padding = 'post')
        if not train_model:
            self.trimming_idx_pad = pad_sequences(trimming_idx, maxlen = trained_max_len, padding = 'post')

        whole_time_lst = []
        for lst in self.trimming_idx_pad:
            num_time = []
            for i, j in enumerate(lst):
                if j:
                    num_time.append(i)
            whole_time_lst.append(num_time)

    def get_start_time(self):
        return self.start_time
    
    def get_file_name(self):
        return self.file_name

    def __getitem__(self, i):
        inputs = torch.Tensor(self.fps_pad[i])
        targets = torch.Tensor(self.trimming_idx_pad[i])
        inputs = inputs.unsqueeze(0)
        return inputs, targets

    def __len__(self):
        return len(self.fps_pad)
    

class DataStack(torch.utils.data.Dataset):
	def __init__(self, fps_dataset):
		self.fps_dataset = fps_dataset
		inputs = [] 
		targets = []
		self.norm_stack = None
		self.targets_stack = None
		
		for i, t in self.fps_dataset:
			inputs.append(i)
			targets.append(t)
		inputs_stack = torch.stack(inputs)
		self.targets_stack = torch.stack(targets)
		

		stack_mean = torch.mean(inputs_stack)
		stack_std = torch.std(inputs_stack) 
		stack_min = torch.min(inputs_stack)
		stack_max = torch.max(inputs_stack)
		# self.norm_stack = (inputs_stack - stack_min) / (stack_max - stack_min) #MinMax Scaler method 
		self.norm_stack = (inputs_stack - stack_mean) / stack_std #Standardization, mean = 0, std = 1
		print("stack mean and stack std: ", stack_mean, stack_std)
    
	def __getitem__(self, index: int):
		input = self.norm_stack[index]
		mask = self.targets_stack[index]
		return input, mask
		
	def __len__(self):
		return len(self.targets_stack)