import os
import pandas as pd
import pubchempy as pcp


class PubChem():
    def __init__(self, file_path):
        self.loda_data(file_path)
        

    def load_data(self, file_path):
        if file_path.endswith('xlsx'):
            self.df = pd.read_excel(file_path)
            print('파일속성 확인', self.df.columns)

        cid = self.df['PubChem_CID']
        cid_single = cid[0]
        
    
    def get_compound(self, cid_single):
        c = pcp.Compound.from_cid(str(cid_single)) #CID 값을 문자열로 입력해야 함
        c = pcp.get_compounds('Eupatilin', 'name') # 실제 이름으로 검색
        c = pcp.get_compounds('COC1=C(C=C(C=C1)C2=CC(=O)C3=C(O2)C=C(C(=C3O)OC)O)OC', 'smiles') #SMIELS로 검색
        c = pcp.get_compounds('Eupatilin', 'name', record_type='3d') # 보통은 2D로 불러오는데 3D 구조 불러올 수 있음 (atom의 좌표)

        return c
    
    def get_property(self, compound, cid):
        # compound에서도 기본적인 분자 특성은 가져올 수 있다
        c_mf = compound.molecular_formula # molecular formula (분자식)
        c_mw = compound.molecular_weight #molecular weight (분자량)
        c_smiles = compound.isomeric_smiles #SMILES
        c_xlogp = compound.xlogp #logP
        c_name = compound.iupac_name # iupac name
        c_synnonyms = compound.synonyms # synonyms

        pcp.get_properties('AtomStereoCount', '6623')
        c_tpsa = pcp.get_properties('TPSA', str(cid))
        ''' Pubchem properties (다음을 get_properties에 넣어서 확인 가능함)
        MolecularFormula, MolecularWeight, CanonicalSMILES, IsomericSMILES, 
        InChI, InChIKey, IUPACName, XLogP, ExactMass, MonoisotopicMass, TPSA, 
        Complexity, Charge, HBondDonorCount, HBondAcceptorCount, RotatableBondCount, 
        HeavyAtomCount, IsotopeAtomCount, AtomStereoCount, DefinedAtomStereoCount, 
        UndefinedAtomStereoCount, BondStereoCount, DefinedBondStereoCount, UndefinedBondStereoCount, 
        CovalentUnitCount, Volume3D, XStericQuadrupole3D, YStericQuadrupole3D, ZStericQuadrupole3D, 
        FeatureCount3D, FeatureAcceptorCount3D, FeatureDonorCount3D, FeatureAnionCount3D, FeatureCationCount3D, 
        FeatureRingCount3D, FeatureHydrophobeCount3D, ConformerModelRMSD3D, EffectiveRotorCount3D, ConformerCount3D.
        '''

        #사전형태로 데이터 불러오기 #get_properties('Property', 'identifier(CID)')
        p = pcp.get_properties('IsomericSMILES', 'CC', 'smiles', searchtype='superstructure')
        p = pcp.get_properties('XLogP', '6623')

        return p


if __name__ == '__main__':
    pubchem = PubChem('/home/bio_science/data/natural/NPASS_NPs.xlsx')
    cid = pubchem.df['PubChem_CID']
    cid_single = cid[0]
    c = pubchem.get_compound(cid_single)
    p = pubchem.get_property(c, cid_single)