from torch.utils.data import Dataset
from PIL import Image
import os
import pandas

class MainDataset(Dataset):
    def __init__(self, csv_file, dir, transform=None):

        self.data_frame = pandas.read_csv(csv_file)
        self.dir = dir
        self.transform = transform

        self.classes = sorted(list(self.data_frame['folder_name'].unique()))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        folder = self.data_frame.iloc[idx, 0]
        filename = self.data_frame.iloc[idx, 1]
        img_path = os.path.join(self.dir, folder, filename)
        
        image = Image.open(img_path).convert('RGB')
        
        label = self.class_to_idx[folder]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label