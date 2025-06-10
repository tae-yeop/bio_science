import torch
import torch.nn as nn



import deepchem as dc
import selfies as sf

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader




class MoleculeDataset(Dataset):
    def __init__(self, X):
        noise = np.random.uniform(0, 1, X.shape).astype(np.float32)
        # normalizing flow to operate on continuous inputs (rather than discrete)
        # original inputs can easily be recovered by applying a floor function
        self.data = torch.from_numpy(X + noise)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataloader(np_rng, n_samples):
    # 1. 데이터셋 로드
    tasks, datasets, transformers = dc.molnet.load_qm9(featurizer='ECFP')

    # 판다스로도 불러올 수 있지만 넘파이가 더 효율적
    # df = pd.DataFrame(data={'smiles': datasets[0].ids})
    # data = df[['smiles']].sample(2500, random_state=42).reset_index(drop=True)

    idx = np_rng.choice(len(datasets[0].ids), n_samples, replace=False)
    smiles = list(np.asarray(datasets[0].ids)[idx])


    # 2. SMILES -> SELFIE

    # bond_constraint 라는 dict를 정의
    # 원자와 이온이 만들 수 있는 본수 갯수 제한 => valid molecules 증가
    sf.set_semantic_constraints()  # reset constraints
    constraints = sf.get_semantic_constraints()
    constraints['?'] = 3
    sf.set_semantic_constraints(constraints)

    selfies = [sf.encoder(smile) for smile in smiles]

    # 3. build Vocab
    alphabet = sorted(
        sf.get_alphabet_from_selfies(np.array(selfies)).union({"[nop]", "."})
    )

    s2i = {s: i for i, s in enumerate(alphabet)}
    i2s = {i:s for s, i in s2i.items()}
    max_len = max(sf.len_selfies(s) for s in selfies)

    onehot = sf.batch_selfies_to_flat_hot(selfies, s2i, max_len, legacy=False).astype(np.float32)
    
    X_train, X_temp = train_test_split(onehot, train_size=args.train_size, random_state=args.seed)
    X_val, X_test = train_test_split(X_temp, test_size=500, random_state=args.seed)
    
    trainset = MoleculeDataset(X_train)
    valset = MoleculeDataset(X_val)

    train_loader = DataLoader(trainset, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(valset, batch_size=args.batch, shuffle=False, drop_last=False)
    
    return train_loader, val_loader


def fix_random_seed(seed=31):
    """
    Fix random seeds.
    """
    random.seed(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return rng


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def set_mask(self, mask: torch.Tensor):
        self.mask.data.copy_(mask)

    def forward(self, x):
        return nn.functional.linear(x, self.mask * self.weight, self.bias)

def create_masks(in_featuers, hidden_features: List[int]):
    """
    Generate MADE connectivity masks.
    """
    degrees = []
    D
class MADE(nn.Moduel):
    """
    고정된 바이너리 마스크를 사용하여 autogressive property를 강제화시킴
    MADE autoregressive network
    Input:  (B, D)
    Output: shift (B, D), log_scale (B, D)
    """
    def __init__(self, in_features, hidden_features: List[int]):
        super.__init__()
        self.net = nn.ModuleList()
        layer_sizes = [in_features] + hidden_features + [2 * in_features]
        masks = create_masks(in_features, hidden_features)

        self.layers = nn.ModuleList()
        for idx, (h_in, h_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            linear = MaskedLinear(h_in, h_out)
            linear.set_mask(masks[idx])
            self.layers.append(linear)

        self.activation = nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        shift, log_scale = x.chunk(2, dim=-1)
        return shift, log_scale.clamp(-5.0, 5.0)

class MAFLayer(nn.Module):
    """
    One affine autoregressive transform
    Affine trasformation, y = x · exp(s) + t
    s, t = MADE(x)
    exact log‑determinant is `sum(s)`

    Input:  (B, D)
    Output: (B, D), log_det (B,)
    """
    def __init__(self, features, hidden):
        super().__init__()
        self.made = MADE(features, [hidden, hidden])
        self.permutation = torch.randperm(features)

    def forward(self, x):
        x = x[:, self.permutation]                 # (B, D)
        shift, log_scale = self.made(x)            # (B, D), (B, D)
        y = x * torch.exp(log_scale) + shift       # (B, D)
        log_det = log_scale.sum(dim=1)             # (B,)
        return y, log_det

    def inverse(self, y):
        x = torch.zeros_like(y)                    # (B, D)
        for i in range(y.size(1)):
            shift, log_scale = self.made(x)        # (B, D), (B, D)
            x[:, i] = (y[:, i] - shift[:, i]) * torch.exp(-log_scale[:, i])
        x = x[:, torch.argsort(self.permutation)]  # undo perm, (B, D)
        return x

        
class Flow(nn.Module):
    """
    alternating (MAF ▸ random permutation) blocks
    """
    def __init__(self, features, num_layers=8, hidden=512):
        super().__init__()
        self.layer = nn.ModuleList(
            MAFLayer(features, hidden) for _ in range(num_layers)
        )

        self.base = torch.distribution.Normal(0, 1)
        self.features = features

    def sample(self, )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=2500)
    parser.add_argument("--train_size", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    np_rng = fix_random_seed(args.seed)

    train_loader, val_loader = get_dataloader(np_rng, args.n_samples)

    

    model = Flow()

