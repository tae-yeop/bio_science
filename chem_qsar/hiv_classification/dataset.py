import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset
import deepchem as dc

import matplotlib.pyplot as plt


class FeaturePreprocessor:
    def __init__(self):
        
    def imbalance_process(self, csv_path, out_csv_path):
        # 불균형하기 때문에 중복을 시켜서 샘플링했을 때 비슷한 비율로 샘플되게끔
        data = pd.read_csv(csv_path)
        neg_class = data["HIV_active"].value_counts()[0]
        pos_class = data["HIV_active"].value_counts()[1]

        multiplier = int(neg_class/pos_class) - 1
        replicated_data = [data[data["HIV_active"] == 1]]*multiplier

        data = pd.concat([data] + replicated_data, ignore_index=True)
        data = data.sample(frac=1).reset_index(drop=True) # drop=True :  기존 인덱스 제거
        data.to_csv(out_csv_path)

    def featurization(
        self,
        csv_path,
        processed_dir,
        output_name='processed_hiv.pt',
        pre_transform=None
    ):
        df = pd.read_csv(csv_path)
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        data_list = []

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            smiles = row['smiles']
            label = row['HIV_active']  # 예: 0 또는 1

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            # DeepChem -> PyG Data
            f = featurizer._featurize(mol)
            data = f.to_pyg_graph()

            # 라벨
            data.y = torch.tensor([label], dtype=torch.long)
            data.smiles = smiles

            if pre_transform is not None:
                data = pre_transform(data)
            
            data_list.append(data)

        torch.save(data_list, os.path.join(processed_dir, output_name))


# torch_geometric.data.Dataset은 processed_file_names의 존재 여부에 따라 process()가 실행하는 원리
# 이를 통해 전처리를 담당할 수 있으나 간단한 케이스만 어차피 적용할거니 쓸모 없다
# super.__init__()에서 이를 실행하게 됨
class HIVDataset(Dataset):
    def __init__(
        self,
        processed_dir,
        file_name='processed_hiv.pt',
        test=False, 
        transform=None, 
        pre_transform=None
    ):

        super().__init__(
            root=None,
            transform=transform, 
            pre_transform=pre_transform
        )

        processed_file = os.path.join(processed_dir, file_name)
        self.data_list = None
        if os.path.exists(processed_file):
            self.data_list = torch.load(processed_file)


    def len(self):
        return len(self.data_list)

    def get(self, idx):
        """
        한 그래프(PyG Data 객체)를 반환.
        이미 self.data_list에 전체가 들어있으므로, 바로 인덱싱.
        """
        return self.data_list[idx]




if __name__ == '__main__':
    csv_path = '/home/tyk/bio_science/chem_qsar/hiv_classification/HIV.csv'
    data = pd.read_csv(csv_path)

    """
        smiles	activity	HIV_active
    0	CCC1=[O+][Cu-3]2([O+]=C(CC)C1)[O+]=C(CC)CC(CC)...	CI	0
    1	C(=Cc1ccccc1)C1=[O+][Cu-3]2([O+]=C(C=Cc3ccccc3...	CI	0
    2	CC(=O)N1c2ccccc2Sc2c1ccc1ccccc21	CI	0
    3	Nc1ccc(C=Cc2ccc(N)cc2S(=O)(=O)O)c(S(=O)(=O)O)c1	CI	0
    4	O=S(=O)(O)CCS(=O)(=O)O	CI	0
    5	CC(=O)N1c2ccccc2Sc2c1ccc1ccccc21	CI	0
    """

    class_counts = data['HIV_active'].value_counts()
    print(class_counts)
    # HIV_active
    # 0    39684
    # 1     1443
    # Name: count, dtype: int64

    # 불균형 데이터셋 확인
    plt.figure(figsize=(6, 4))
    class_counts.plot(kind="bar", color=["blue", "orange"])
    plt.xlabel("Class (HIV_active)")
    plt.ylabel("Count")
    plt.title("Distribution of HIV_active classes")
    plt.xticks(rotation=0)
    plt.show()


    neg_class = data["HIV_active"].value_counts()[0]
    pos_class = data["HIV_active"].value_counts()[1]

    print(neg_class, pos_class) # (39684, 1443)