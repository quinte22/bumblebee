import time
import sys
sys.path.insert(0, 'bumblebee/pno_ai/')
from pno_ai.preprocess import PreprocessingPipeline
from pno_ai.helpers import prepare_batches
from pno_ai.train import batch_to_tensors
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = (device.type == "cuda")


class LSTMClassifier(nn.Module):
    """
    A regular one layer LSTM classifier using LSTMCell -> FFNetwork structure
    """
    def __init__(self, input_dim, hidden_dim, label_size, device=torch.device("cuda"), dropout_rate=0.1):
        super().__init__()
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        self.hidden2ff = nn.Linear(hidden_dim,  int(np.sqrt(hidden_dim)))
        self.ff2label = nn.Linear(int(np.sqrt(hidden_dim)), label_size)
        self.hidden_dim = hidden_dim
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)
        self.device = device
        self.initialize_weights()

    def initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0001)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.hidden2ff.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0001)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.ff2label.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0001)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, x):

        hs = torch.zeros(x.size(0), self.hidden_dim).to(self.device)
        cs = torch.zeros(x.size(0), self.hidden_dim).to(self.device)

        for i in range(x.size()[1]):
            hs, cs = self.lstm(x[:, i], (hs, cs))

        hs = self.dropout(hs)
        hs = self.hidden2ff(hs)
        return self.sigmoid(self.ff2label(hs))


sampling_rate = 125
n_velocity_bins = 32
seq_length = 1024

pipeline = PreprocessingPipeline(input_dir="pno_ai/data", stretch_factors=[0.975, 1, 1.025],
                                 split_size=30, sampling_rate=sampling_rate, n_velocity_bins=n_velocity_bins,
                                 transpositions=range(-2, 3), training_val_split=0.9, max_encoded_length=seq_length +1,
                                 min_encoded_length=257)
pipeline_start = time.time()
pipeline.run()
runtime = time.time() - pipeline_start
print(f"MIDI pipeline runtime: {runtime / 60 : .1f}m")

training_sequences = pipeline.encoded_sequences['training']
validation_sequences = pipeline.encoded_sequences['validation']

batch_size = 16
num_workers = 4

max_length = max((len(L)
                  for L in (training_sequences + validation_sequences))) - 1

training_batches = prepare_batches(training_sequences, batch_size)
validation_batches = prepare_batches(validation_sequences, batch_size)

print(np.shape(training_batches))
print(np.shape(validation_batches))

train_data, train_targets, train_mask = batch_to_tensors(training_batches, 0,
                                max_length)

validation_data, validation_targets, validation_mask = batch_to_tensors(validation_batches, 0,
                                max_length)

train_dataset = TensorDataset(train_data, train_targets)
train_sampler = RandomSampler(train_data)
train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers,
                          pin_memory=pin_memory)
validation_dataset = TensorDataset(validation_data, validation_targets)
validation_sampler = RandomSampler(validation_dataset)
validation_loader = DataLoader(validation_dataset, sampler=validation_sampler, batch_size=batch_size,
                               num_workers=num_workers, pin_memory=pin_memory)
