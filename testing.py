import numpy as np
import os

sample = os.path.join('data/train_tisv/speaker6.npy')

sample_npy = np.load(sample)
print(sample_npy)
