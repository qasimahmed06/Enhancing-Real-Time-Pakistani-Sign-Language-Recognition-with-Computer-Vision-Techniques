from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10), # random rotation +-10 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # randomly increase brightness contrast etc
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # random crop and resize
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
])


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
])






class PSLDataset(Dataset):
    def __init__(self , root_dir , train_transform=None , val_transform=None):
        self.root_dir = root_dir
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.image_paths = [] # storing image paths for preprocessing
        self.labels = [] # storing labels for the those images
        self.label_map = {} # mapping class names to integer labels

        counter = 0 # for assigning labels to each class
        
        # First pass: Check which folders have videos and which have images
        for subdir in sorted(os.listdir(root_dir)):  # Sort for consistent ordering
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                files = os.listdir(subdir_path)
                
                has_videos = any(f.lower().endswith('.mp4') for f in files)
                
                if has_videos:
                    print(f"⚠️  Skipping folder '{subdir}' - contains MP4 files")
                    continue 
                
                # Check if folder has image files
                has_images = any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files)
                
                if not has_images:
                    print(f" Skipping folder '{subdir}' - no image files found")
                    continue  
                
                self.label_map[counter] = subdir
                for img_name in files:
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(subdir_path, img_name))
                        self.labels.append(counter)
                
                counter += 1 
        
        
    def __len__(self):
        return len(self.image_paths)
    

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        
        
        image = Image.open(image_path).convert('RGB')
        
        if self.train_transform:
            image = self.train_transform(image)
        else:
            image = self.val_transform(image)

        return image, label
