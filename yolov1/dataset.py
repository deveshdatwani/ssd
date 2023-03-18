from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image


class imageDataset(Dataset):
    def __init__(self, annotation_file_address: str, data_directory: str) -> object:
        self._annotations = pd.read_csv(annotation_file_address)
        self._data_directory = data_directory