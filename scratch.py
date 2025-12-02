from dataset.dataset_wmt25 import Wmt25Dataset
from dataset.dataset_tatoeba import Tatoeba
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

dataset_wmt = Wmt25Dataset()
dataset_tatoeba = Tatoeba()

dataset_union = ConcatDataset([dataset_tatoeba, dataset_wmt])

dataloader = DataLoader(dataset_union, batch_size=4, shuffle=True)

for batch in dataloader:
    pass

