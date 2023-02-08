import os
import pickle as pk
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.stats import pearsonr
from torch.nn import CosineSimilarity
from tqdm.notebook import tqdm

from src.architectures import SimpleCoembedding
from src.featurizers import MorganFeaturizer, ProtBertFeaturizer

DEVICE = 6
MODEL_PATH = "best_models/reverse_margin_best_model.pt"


def cosine_similarity(x, y):
    cs = CosineSimilarity(dim=0)
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
    return cs(x, y).item()


def jaccard_score(x, y):
    x = np.array(x).astype(bool)
    y = np.array(y).astype(bool)
    inter = x & y
    union = x | y
    if sum(union) == 0:
        return 0
    else:
        return sum(inter) / sum(union)


if __name__ == "__main__":
    try:
        data_file = sys.argv[1]
        print(data_file)
    except IndexError:
        print("usage: python analyze_drug_space.py [assay data file]")

    # Load Data
    df = pd.read_csv(data_file, sep="\t")

    # Load Featurizers and Model
    device = torch.device(6)
    drug_f = MorganFeaturizer().cuda(device)
    target_f = ProtBertFeaturizer().cuda(device)
    model = SimpleCoembedding(drug_f.shape, target_f.shape, 1024)
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(device)
