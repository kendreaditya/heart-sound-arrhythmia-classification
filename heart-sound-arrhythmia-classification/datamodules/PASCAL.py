import os
import torch
import pytorch_lightning as pl
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader, TensorDataset
from torchsampler import ImbalancedDatasetSampler
from scipy.io import wavfile
from scipy import signal
import numpy as np


class PASCAL(pl.LightningDataModule):
    def __init__(self, data_path, sample_length=2500, dataset_split=[0.7, .1, .2], file_extention="wav", freq=500):
        self.file_extention = file_extention
        self.freq = freq
        self.sample_length = sample_length
        self.dataset_split = dataset_split
        self.dataset_dir = {"normal": [f"{data_path}/Atraining_normal/", f"{data_path}/Training B Normal/"],
                            "murmur": [f"{data_path}/Atraining_murmur/", f"{data_path}/Btraining_murmur/"],
                            "extra-heart-sounds": [f"{data_path}/Atraining_extrahls/", f"{data_path}/Btraining_extrastole/"],
                            "artifact": [f"{data_path}/Atraining_artifact/"]}

        self.dataset_lbls = {key: i for i, key in enumerate(self.dataset_dir.keys())}

        self.data = []
        self.x = []
        self.y = []

    def setup(self):
        print("Preparing data...")
        for label in self.dataset_dir:
            # Lable for current directory
            set_lbl = self.dataset_lbls[label]
            for path in self.dataset_dir[label]:
                for fn in os.listdir(path):

                    # File extention check
                    if fn.split(".")[-1] != self.file_extention:
                        continue

                    # Reading audio file
                    freq, audio = wavfile.read(f"{path}{fn}")
                    self.data.append([freq, audio, set_lbl])
        print("Done Preparing data!")

    def prepare_data(self, stage=None):
        print("Starting Preprocessing...")
        for freq, audio, lbl in tqdm(self.data):

            # Resample to freq
            freq, audio = self.resample(audio, freq, self.freq)

            # Standardize
            audio = (np.array(audio) - np.mean(audio)) / np.std(audio)

            # Splitting
            audio_splits = self.split_data(audio, self.sample_length)

            for audio_sample in audio_splits:
                self.x.append(audio_sample)
                self.y.append(lbl)

        # Sparce Split Dataset
        self.traning_set, self.validation_set, self.testing_set = self.split_dataset(self.x, self.y, self.dataset_split)
        self.traning_set = [torch.tensor(self.traning_set[0]), torch.tensor(self.traning_set[1])]
        self.validation_set = [torch.tensor(self.validation_set[0]), torch.tensor(self.validation_set[1])]
        self.testing_set = [torch.tensor(self.testing_set[0]), torch.tensor(self.testing_set[1])]
        print("Done Preprocessing!")

    def resample(self, data, freq, target_freq):
        if freq != target_freq:
            return freq, data

        secs = len(data) / freq
        num_samples = int(secs * target_freq)
        data = signal.resample(data, num_samples)
        return freq, data

    def split_data(self, data, sample_length):
        # implament spltting basedon s1 and s2
        data_splits = []

        for i in range(0, len(data), sample_length):
            if len(data[i:i + sample_length]) == sample_length:
                data_splits.append(data[i:i + sample_length])
        return data_splits

    def split_dataset(self, x, y, ratio, stratified=True):
        """
        testing set -> get full length via ratio*len(y)
        according get the type number of labels types
        1/2 = number of labels for type for testing set
        """
        dataset = list(map(list, zip(x, y)))

        if stratified == False:
            return None

        label_maped = {}

        for datum in dataset:
            lbl = str(datum[1])
            if lbl not in label_maped.keys():
                label_maped[lbl] = list()
            label_maped[lbl].append(datum)

        idx_ratio = ratio.copy()
        idx_ratio.insert(0, 0)
        idx_ratio = np.array(idx_ratio)

        split_idx = {lbl: np.floor(idx_ratio * len(label_maped[lbl])).astype(int) for lbl in label_maped.keys()}

        split_x, split_y = [[] for _ in ratio], [[] for _ in ratio]

        for set_n in range(len(ratio)):
            for lbl in label_maped.keys():
                start, end = sum(split_idx[lbl][:set_n + 1]), sum(split_idx[lbl][:set_n + 2])
                split_lbl = label_maped[lbl][start:end]

                [split_x[set_n].append(i[0]) for i in split_lbl]
                [split_y[set_n].append(i[1]) for i in split_lbl]

        return list(map(list, zip(split_x, split_y)))

    def train_dataloader(self):
        dataset = TensorDataset(self.traning_set[0], self.traning_set[1])
        data_loader = DataLoader(dataset, batch_size=128, sampler=ImbalancedDatasetSampler(
            dataset, callback_get_label=lambda dataset, idx: int(dataset[idx][1])))
        return data_loader

    def val_dataloader(self):
        dataset = TensorDataset(self.validation_set[0], self.validation_set[1])
        data_loader = DataLoader(dataset, batch_size=128, shuffle=True)
        return data_loader

    def test_dataloader(self):
        dataset = TensorDataset(self.testing_set[0], self.testing_set[1])
        data_loader = DataLoader(dataset, batch_size=128, shuffle=True)
        return data_loader


if __name__ == "__main__":
    datamodule = PASCAL(
        "K:\OneDrive - Cumberland Valley School District\Education\Activates\Science Fair\PCG-Science-Fair\PCG-arrhythmia-detection\data\PASCAL")
    datamodule.setup()
    datamodule.prepare_data()
