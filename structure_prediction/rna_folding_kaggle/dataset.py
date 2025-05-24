from torch.utils.data import Dataset


class RNADataset(Dataset):
    def __init__(self, data, max_len=384):
        super().__init__()
        self.data = data
        self.tokens = {token:idx for idx, token in enumerate('ACGU')}
        self.keys = sorted(self.data)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.keys[idx]
        coords = self.data[sequence]

        if len(coords) > self.max_len:
            coords = coords[:self.max_len]
        
        return {
            'sequence': sequence,
            'coords': torch.tensor(coords, dtype=torch.float32),
            'length': len(coords)
        }



if __name__ == '__main__':
    test = {'a' : [3, 2, 1,3,4,6], 'b' : [3,1,5,6,6,1,2,3,5,1,3,4,5]}
    keys = sorted(test)
    print(keys, type(keys))