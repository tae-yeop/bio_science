import numpy as np
import pandas as pd

import rdkit
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, MACCSkeys, AllChem
from rdkit.Chem import rdFingerprintGenerator

class MEK1Processor:
    def __init__(self, csv_path):
        """
            prot_id	                                    seq	                canonical_smiles                                pIC50
        0	Q02750	MPKKKPTPIQLNPAPDGSAVNGTSSAETNLEALQKKLEELELDEQQ...	COc1cc(Nc2c(C#N)cnc3cc(-c4ccc(CN5CCOCC5)cc4)cc...	6.552842
        1	Q02750	MPKKKPTPIQLNPAPDGSAVNGTSSAETNLEALQKKLEELELDEQQ...	N#Cc1c(N)[nH]c(Sc2ccc3ccccc3c2)c1C#N	5.853872
        2	Q02750	MPKKKPTPIQLNPAPDGSAVNGTSSAETNLEALQKKLEELELDEQQ...	N#CC(=C(/N)Sc1ccccc1Cl)/C(C#N)=C(\N)Sc1ccccc1Cl	6.045757
        3	Q02750	MPKKKPTPIQLNPAPDGSAVNGTSSAETNLEALQKKLEELELDEQQ...	N#CC(=C(/N)Sc1ccccc1Br)/C(C#N)=C(\N)Sc1ccccc1Br	5.958607
        4	Q02750	MPKKKPTPIQLNPAPDGSAVNGTSSAETNLEALQKKLEELELDEQQ...	N#CC(=C(/N)Sc1ccccc1O)/C(C#N)=C(\N)Sc1ccccc1O	6.522879
        """
        self.df = pd.DataFrame(csv_path, index_col=0)
        self.smiles_list = list(self.df['canonical_smiles'])
        self.maccs_list = self.get_maccs_list()
    def mol_to_fingerprint(smiles, radius=2, nBits=1024):
        """
        Returns : (1024) 짜리 넘파이 어레이
        """

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES string: {smiles}")
            
            generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
            fp = generator.GetFingerprint(mol)
            return np.array(fp)

        except Exception as e:
            print(f"Error processing SMILES '{smiles}': {e}")
            return np.nan

    def mol_to_maccs(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES string: {smiles}")
            can_smi = Chem.MolToSmiles(mol, canonical=True) # rdkit.DataStructs.cDataStructs.ExplicitBitVect
            maccs = MACCSkeys.GenMACCSKeys(mol)
            return np.array(maccs)
        except Exception as e:
            print(f"Error processing SMILES '{smiles}': {e}")
            return np.nan

    def get_ecfp_list(self):
        errors =[]
        fps=[]

        for i, smi in enumerate(self.smiles_list):
            try:
                m = Chem.MolFromSmiles(smi)
                fp = np.array(AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=1024))
                fps.append(fp)
            except Exception as e:
                print(f"Error at index {i} for SMILES {smi}: {e}")
                errors.append(i)

        return fps
            
    def get_maccs_list(self):
        errors=[]
        maccs_keys=[]

        for i, smi in enumerate(self.smiles_list):
            try:
                m = Chem.MolFromSmiles(smi)
                can_smi = Chem.MolToSmiles(m,canonical=True)
                maccs = MACCSkeys.GenMACCSKeys(m)
                maccs_keys.append(maccs)
            except:
                print(i)
                errors.append(i)

        return maccs_keys
