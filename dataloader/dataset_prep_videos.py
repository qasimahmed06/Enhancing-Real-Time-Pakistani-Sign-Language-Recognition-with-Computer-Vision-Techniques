from torch.utils.data import Dataset
import os
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.Resize(224, 224),
    A.Rotate(limit=10, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


class PSLVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_frames=30, sample_rate=1):
        self.root_dir = root_dir
        self.transform = transform
        self.max_frames = max_frames
        self.sample_rate = sample_rate
        self.video_paths = []
        self.labels = []
        self.label_map = {}
        
        counter = 0
        for subdir in sorted(os.listdir(root_dir)):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                video_files = [f for f in os.listdir(subdir_path) 
                              if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                
                if video_files:
                    self.label_map[counter] = subdir
                    for video_name in video_files:
                        self.video_paths.append(os.path.join(subdir_path, video_name))
                        self.labels.append(counter)
                    counter += 1
    
    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while cap.isOpened() and len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % self.sample_rate == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            frame_count += 1
        
        cap.release()
        return frames if frames else [cv2.cvtColor(cv2.imread('placeholder.jpg'), cv2.COLOR_BGR2RGB)]
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, index):
        video_path = self.video_paths[index]
        label = self.labels[index]
        frames = self.extract_frames(video_path)
        
        transformed_frames = []
        for frame in frames:
            if self.transform:
                transformed = self.transform(image=frame)
                transformed_frames.append(transformed['image'])
            else:
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                transformed_frames.append(frame_tensor)
        
        return torch.stack(transformed_frames), label, video_path

if __name__ == "__main__":
    dataset = PSLVideoDataset("Dataset", transform=train_transform, max_frames=30, sample_rate=2)
    if len(dataset) > 0:
        frames, label, video_path = dataset[0]
        print(f"Sample: {video_path}, Label: {label}, Shape: {frames.shape}")
 