import os
import glob
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

class AudioSteganalysisDataset(Dataset):
    def __init__(self, cover_dir, stego_dir, limit=None):
        self.cover_paths = sorted(glob.glob(os.path.join(cover_dir, '*.wav')))
        self.stego_paths = sorted(glob.glob(os.path.join(stego_dir, '*.wav')))
        
        # Ensure balanced
        min_len = min(len(self.cover_paths), len(self.stego_paths))
        if limit:
            min_len = min(min_len, limit)
            
        self.cover_paths = self.cover_paths[:min_len]
        self.stego_paths = self.stego_paths[:min_len]
        
        self.files = [(p, 0) for p in self.cover_paths] + [(p, 1) for p in self.stego_paths]
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path, label = self.files[idx]
        try:
            waveform, sr = torchaudio.load(path)
            # Ensure mono
            if waveform.shape[0] > 1:
                waveform = waveform[0:1, :]
            
            # Normalize? Standard scaling -1 to 1 is usually done by load
            return waveform, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros(1, 16000), torch.tensor(label, dtype=torch.float32) # Dummy

def create_audio_dataloaders(cover_dir, stego_dir, batch_size=32, limit=None):
    dataset = AudioSteganalysisDataset(cover_dir, stego_dir, limit=limit)
    
    total_size = len(dataset)
    train_size = int(total_size * 0.8)
    val_size = int(total_size * 0.1)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
