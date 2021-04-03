import time
import sys
sys.path.insert(0, 'bumblebee/pno_ai/')
from pno_ai.preprocess import PreprocessingPipeline
from pno_ai.helpers import prepare_batches
from pno_ai.train import batch_to_tensors
import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = (device.type == "cuda")


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
