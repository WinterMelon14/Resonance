import torch, h5py, numpy as np
from torch.utils.data import Dataset
class MaestroChunkDataset(Dataset):
    def __init__(self, h5_path: str):
        print("Loading entire dataset into system RAM... Please wait.")
        with h5py.File(str(h5_path), "r") as hf:
            # [:] reads the entire array from disk into memory immediately
            self.X = hf["X"][:]  
            self.Y = hf["Y"][:]
        print("Successfully loaded dataset into RAM!")

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        X = self.X[idx]
        Y = self.Y[idx]
        
        # Perform conversions instantly in RAM
        X = torch.from_numpy(X.astype(np.float32) / 255.0).unsqueeze(0)
        Y = torch.from_numpy(Y.astype(np.float32))
        return X, Y
