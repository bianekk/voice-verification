# './split_data/train/*/*.wav'

from torch.utils.data import Dataset
import glob
from Speaker_Verification.utils import mfccs_and_spec
import torch
from torchmetrics.functional import pairwise_cosine_similarity
import os

class SpeakerDataset(Dataset):
    
    def __init__(self, path, device):
        self.path = path
        self.utterance_number = 1
        self.speakers = glob.glob(os.path.dirname(self.path))
        self.device = device
        
    def __len__(self):
        return len(self.speakers)

    def __getitem__(self, idx):
        
        speaker = self.speakers[idx]
        wav_files = glob.glob(speaker+'/*.wav')
        wav_files = wav_files
        
        mel_dbs = []
        for f in wav_files:
            _, mel_db, _ = mfccs_and_spec(f, wav_process = True)
            mel_dbs.append(torch.Tensor(mel_db))
        return torch.stack(mel_dbs).to(self.device)
    

def make_experent(dist_threshold, profile_tensor, voice_tensor):

    cosine_dist_matrix = 1 - pairwise_cosine_similarity(profile_tensor, voice_tensor)

    cosine_dist_matrix = cosine_dist_matrix <= dist_threshold

    decisions = cosine_dist_matrix.float().mean(dim=0)  > 0.5

    predicted_labels = decisions.cpu().long().numpy().tolist()
    
    return predicted_labels

def measure_cos_sim(profile_tensor, voice_tensor):

    cosine_dist_matrix = 1 - pairwise_cosine_similarity(profile_tensor, voice_tensor)
    
    return cosine_dist_matrix

def find_thresh(model, train_dataset, val_dataset):
    import numpy as np
    from tqdm import tqdm
    import os
    import random
    from notebooks.metrics import calculate_by_treshold, plot_far_frr

    # prepare the training and validation sets for pairing

    ids = range(len(train_dataset))

    random_ids = random.sample(ids, 50)
    profiles = []

    for person_id in random_ids:
        
        profiles.append({'train': train_dataset[person_id],
                         'dev': val_dataset[person_id]})

    y_true = []
    y_pred = []

    for profile_id, profile in enumerate(tqdm(profiles)):
        img_orig = profile['train'][0]
        for i in range(20):
            if i < len(profile['dev']):
                try:
                    img_to_compare = profile['dev'][i]
                    dist = 1 - torch.nn.functional.cosine_similarity(model(img_orig.unsqueeze(0)), 
                                                                    model(img_to_compare.unsqueeze(0)), 
                                                                    dim=1, eps=1e-8)
                    y_true.append(1)
                    y_pred.append(dist)
                except IndexError:
                    break
        for other_profile_id, other_profile in enumerate(profiles):
            if other_profile_id == profile_id:
                continue
            else:
                if i < len(other_profile['dev']):
                    img_to_compare = other_profile['dev'][i]
                    dist = 1 - torch.nn.functional.cosine_similarity(model(img_orig.unsqueeze(0)), 
                                                                    model(img_to_compare.unsqueeze(0)), 
                                                                    dim=1, eps=1e-8)
                    y_true.append(0)
                    y_pred.append(dist)


    unique, counts = np.unique(y_true, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    # result = np.subtract(1.0, y_pred)
    far, ffr = calculate_by_treshold(y_true, y_pred, counts_dict[1], counts_dict[0])
    plot_far_frr(far, ffr)