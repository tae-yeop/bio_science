import os
import pandas as pd
from urllib.error import HTTPError
from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests


class PubChemCrwaler():
    def __init__(self,  file_path):
        self.df = pd.read_excel(file_path)
        #CAS 정보 추출하기
        cas = self.df['CAS']
        self.pubchem_cid = []
        self.smiles = []
        self.aids = []

    def pubchem_crawling(self, cas):
        """
        webcrawling을 쓰는 이유는 CAS 번호로 부터 추출하기 위해서
        """
        for cas_num in cas:
            try:
                label=True
                print(str("Input CAS number: ") + str(cas_num).strip())
                pubchem_url = "https://pubchem.ncbi.nlm.nih.gov/compound/" + str(cas_num).strip()
                cid_info = urlopen(pubchem_url, None, timeout=10000)

                while True:
                    # 한줄 한줄 읽다가 패턴 속에서 CID 값이 있는 걸 추출
                    # 불필요한 부분을 제거하고 CID만 뽑는다
                    pubchem_line = cid_info.readline()
                    if not pubchem_line: break

                    if (str(pubchem_line).__contains__('''b'    <meta name="pubchem_uid_value"''')):
                        self.pubchem_cid.append(str(pubchem_line).replace("b'", "").
                                            replace('"', '').
                                            replace("    <meta name=pubchem_uid_value content=", "").
                                            replace("'", "").
                                            replace(">", "").
                                            replace("\\n", "").strip())

                        label = False
                        break

                if(label):
                    self.pubchem_cid.append("None")

            except HTTPError as e:
                print(e)
                self.pubchem_cid.append("None")

    def get_smiles_from_cid(self, property='CanonicalSMILES'):
        """
        가능한 Properties
            MolecularFormula, MolecularWeight, CanonicalSMILES, IsomericSMILES, 
            InChI, InChIKey, IUPACName, XLogP, ExactMass, MonoisotopicMass, TPSA, 
            Complexity, Charge, HBondDonorCount, HBondAcceptorCount, RotatableBondCount, 
            HeavyAtomCount, IsotopeAtomCount, AtomStereoCount, DefinedAtomStereoCount, 
            UndefinedAtomStereoCount, BondStereoCount, DefinedBondStereoCount, UndefinedBondStereoCount, 
            CovalentUnitCount, Volume3D, XStericQuadrupole3D, YStericQuadrupole3D, ZStericQuadrupole3D, 
            FeatureCount3D, FeatureAcceptorCount3D, FeatureDonorCount3D, FeatureAnionCount3D, FeatureCationCount3D, 
            FeatureRingCount3D, FeatureHydrophobeCount3D, ConformerModelRMSD3D, EffectiveRotorCount3D, ConformerCount3D.
        """

        
        # CID에서 어떤 property를 텍스트로 가져오겠다는 적어둠
        # pubchem 라이브러리를 쓰던 지금같은 크롤링 방법을 쓰던지 똑같음
        for cid_num in self.pubchem_cid:
            try:
                label = True
                print(str("Input CID number: ") + str(cid_num).strip())
                url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/" + str(cid_num).strip() + f"/property/{property}/txt"
                url_info = urlopen(url, None, timeout=100000)

                
                bsObject = BeautifulSoup(url_info, "html.parser")
                All_txt = bsObject.text

                self.smiles.append(All_txt.strip("\n"))

                label = False

                if(label):
                    self.smiles.append("None")

            except HTTPError as e:
                print(e)
                self.smiles.append("None")


    def get_assay_from_cid(self, ):
        # pubchempy로는 biassay 데이터를 가져오는데 제한됨 -> 크롤링 방식을 써야함
        # aid = assay id

        # assay 하나마다 파일이 하나 있는 셈

        for cid_num in self.pubchem_cid:
            try:
                label = True
                print(str("Input CID number: ") + str(cid_num).strip())
                assay_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/" + str(cid_num).strip() + "/aids/txt"

                response = requests.get(assay_url, verify=False)
                self.aids.append(response.text.strip().split('\n'))

                label = False

                if(label):
                    self.aids.append("None")

            except HTTPError as e:
                print(e)
                self.aids.append("None")

        
    def fetch_data_from_aids(self, out_dir):
        for aid_list in self.aids:
            for aid_num in aid_list:
                try:
                    assay_download_url = "https://pubchem.ncbi.nlm.nih.gov/assay/pcget.cgi?query=download&record_type=datatable&actvty=all&response_type=save&aid=" + str(aid_num)
                    download_response = requests.get(assay_download_url, verify=False)
                    download_file_info = str(out_dir) + str(aid_num) + str(".csv")
                    with open(download_file_info, "wb") as file:
                        file.write(download_response.content)

                except HTTPError as e:
                    print(e)
                    self.aids.append("None")


    def fetch_one_sample_from_aids(self, filepath='./test.csv'):
        target_aid = self.aids[0][0] #위에서 수집했던 aid 중 한개만 추출
        assay_download_url = "https://pubchem.ncbi.nlm.nih.gov/assay/pcget.cgi?query=download&record_type=datatable&actvty=all&response_type=save&aid=" + str(target_aid)
        download_response = requests.get(assay_download_url, verify=False)
        with open(filepath, "wb") as file:
            file.write(download_response.content)

            