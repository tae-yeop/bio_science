# https://www.kaggle.com/code/shujun717/ribonanzanet-3d-finetune
# https://www.kaggle.com/code/ogurtsov/rhofold-ribonanzanet-msas-lb-0-215

import numpy as np
import pandas as pd
from tqdm import tqdm


from pydantic import BaseModel

class Config(BaseModel):
    seed: int = 0
    cutoff_date: str = "2020-01-01"
    test_cutoff_date: str = "2022-05-01"
    max_len: int = 384
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    mixed_precision: str = "bf16"
    model_config_path: str = "../working/configs/pairwise.yaml"  # Adjust path as needed
    epochs: int = 10
    cos_epoch: int = 5
    loss_power_scale: float = 1.0
    max_cycles: int = 1
    grad_clip: float = 0.1
    gradient_accumulation_steps: int = 1
    d_clamp: int = 30
    max_len_filter: int = 9999999
    min_len_filter: int = 10
    structural_violation_epoch: int = 50
    balance_weight: bool = False

if __name__ == '__main__':

    cfg = Config()
    # Load the CSV file
    train_sequences=pd.read_csv("/home/bio_science/structure_prediction/rna_folding_kaggle/data/train_sequences.csv")
    train_labels=pd.read_csv("/home/bio_science/structure_prediction/rna_folding_kaggle/data/train_labels.csv")


    train_labels["pdb_id"] = train_labels["ID"].apply(lambda x: x.split("_")[0]+'_'+x.split("_")[1])
    print(train_labels["pdb_id"])

    all_xyz=[]

    for pdb_id in tqdm(train_sequences['target_id']):
        df = train_labels[train_labels["pdb_id"]==pdb_id]
        # print(df)
        #break
        xyz=df[['x_1','y_1','z_1']].to_numpy().astype('float32')
        # print(type(xyz))
        # print(xyz)
        xyz[xyz<-1e17]=float('Nan') # 여기서 마이너스 값이 엄청 큰건 Nan 처리한다. 왜?
        all_xyz.append(xyz)



    # filter the data
    # Filter and process data
    filter_nan = [] # 여기에는 True, False가 들어간다. True면 거르지 않고 False이면 거른다
    max_len = 0
    cnt = 0
    for xyz in all_xyz:
        if len(xyz) > max_len:
            max_len = len(xyz)

        filter_nan.append((np.isnan(xyz).mean() <= 0.5) & \
                      (len(xyz)<cfg.max_len_filter) & \
                      (len(xyz)>cfg.min_len_filter))

        cnt += 1
        if cnt == 10:
            break

    print(filter_nan)
    
    print(f"Longest sequence in train: {max_len}")

    filter_nan = [True, True, True, True, False, True, True, False, True, True]
    filter_nan = np.array(filter_nan)
    print(filter_nan)
    non_nan_indices = np.arange(len(filter_nan))[filter_nan]

    print('non_nan_indices', non_nan_indices)

    train_sequences = train_sequences.loc[non_nan_indices].reset_index(drop=True)
    all_xyz=[all_xyz[i] for i in non_nan_indices]


    data = {
        "sequence":train_sequences['sequence'].to_list(),
        "temporal_cutoff": train_sequences['temporal_cutoff'].to_list(),
        "description": train_sequences['description'].to_list(),
        "all_sequences": train_sequences['all_sequences'].to_list(),
        "xyz": all_xyz
    }

    