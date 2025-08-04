'''Added aggressive commenting to explain my logic.'''

#Imports
from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import json
import torchvision.transforms as transforms

#Dataloader class
class PolygonDataset(Dataset):
    def __init__(self, root_dir, color_to_id, transform=None):
        self.root_dir = root_dir
        self.inputs_path = os.path.join(root_dir, "inputs")
        self.outputs_path = os.path.join(root_dir, "outputs")
        self.color_to_id = color_to_id
        self.transform = transform

        with open(os.path.join(root_dir, "data.json")) as f:
            self.metadata = json.load(f)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]

        # Keys based on data.json
        input_img_path = os.path.join(self.inputs_path, item["input_polygon"])
        output_img_path = os.path.join(self.outputs_path, item["output_image"])
        color_label = item["colour"] 

        #Open the input-output images based on keys
        input_img = Image.open(input_img_path).convert("RGB")
        output_img = Image.open(output_img_path).convert("RGB")

        # Custom transformatiion logic (Just setup - unused)
        if self.transform:
            input_img = self.transform(input_img)
            output_img = self.transform(output_img)
        
        # If no custom transformation fall back to deafault : Convert to tensor and nornalize [0,1]
        else:
            input_img = transforms.ToTensor()(input_img)
            output_img = transforms.ToTensor()(output_img)

        # Convert color label to color id based on color_to_id dictionary.
        color_id = self.color_to_id[color_label.lower()]

        # Return transformed images, and tensor of color_id
        return input_img, output_img, torch.tensor(color_id, dtype=torch.long)
