https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/


- 압축 풀면 6.25기가
```python
wget https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_35.sdf.gz

gunzip chembl_35.sdf.gz

```

https://docs.nvidia.com/bionemo-framework/1.10/notebooks/ZINC15-data-preprocessing.html

```python
wget https://files.docking.org/2D/AB/ABAC.txt 
```



zinc_250k_splits
- properties 없이 SMILES만 있음


## 데이터 소스

### 화합물 DB

[BindingDB](https://www.bindingdb.org/rwd/bind/) : 단백질-화합물 결합 친화도 데이터를 제공하는 데이터베이스

[PubChem BioAssay](https://pubchem.ncbi.nlm.nih.gov) : 화합물의 구조, 생물학적 활성, 독성 및 기타 생물학적 특성에 대한 정보를 
제공


PubChem DB를 이용한 분자구조 데이터 수집

pusbchem.ncbi.nlm.nih.gov/docs/pug-rest

[ChEMBL](https://www.ebi.ac.uk/chembl/)
- 화합물의 생리적 활성 및 약리학적 정보를 포함
- 사람이 일일이 바이오 활성을 가지는 화합물에 대한 정보를 DB 구축
- 2.4M개의 Compounds
- 1.5M개의 Assays(activity 실험 데이터)
    - binding assay : small molecule과 binding하는지


![Image](https://github.com/user-attachments/assets/10a58e37-8e3f-45fd-b9de-142a13eb763f)

- CSV를 클릭해서 다운받기
- 다큐먼트 ID마다 개별적인 실험 환경
- 일일이 클릭하기 힘들기 때문에 API를 제공한다


[ZINC](http://zinc.docking.org/)
[SciFinder](https://scifinder.cas.org/)
[Cambridge Crystallographic Data Center](https://www.ccdc.cam.ac.uk/structures)
[DrugBank](https://www.drugbank.ca/)
- 신약 개발과 관련된 생리활성 화합물의 구조와 생물학적 활성을 포함하는 데이터베이스이다.

[Moleculenet](https://moleculenet.org/datasets-1)
- 빨리 써볼 수 있는 분자 데이터셋
- Pytorch Geometric에서 받을 수 있는 데이터의 출처가 여기인듯


### 단백질 DB

Protein sequences are available in Uniprot (about 200M proteins) and MGnify databases.

Protein structures are available in PDB database (about 200K protein structures).

[Protein Data Bank, PDB](https://www.rcsb.org/)
- 단백질과 기타 생체 분자의 3차원 구조 정보를 제공하는 데이터베이스이다.



### 천연물 DB

[NPASS DB](https://bidd.group/NPASS-2018/downloadnpass.html)
- 제공하는 정보
1. 일반정보: 물질 ID, 이름, PubChem 및 ChEMBL ID 정보
2. 구조정보: SMILES, InChi, InChi-Key
3. 실험데이터: 천연물, 단백질, 실험데이터(IC50, PC50 등)
4. 천연물 유래 종: 천연물이 포함되어 있는 종 이름
5. 단백질 정보: 천연물에 연관성이 있는 단백질 리스트
6. 종 분류 정보: 분류 체계 정보

- NPASS-2023 버전이 최신이다
    - Natural Products-General Information : 일반 구조
        - num_of_organism : 천연물에 관련된 종 정보
        - num_of_target: 질병 정보
        - num_of_activity : 실험 데이터 갯수
    - Natural Products-Structure: 구조 정보
    - Natural Products-Activity Records: 활성 정보
        - np_id : 천연물 아이디
        - target_id : 단백질 아이디
        - activity_type_grouped: 실험 지표
        - activity_value : 실험값
        - assay_organism : 어떤 종으로 실험했는지 
        - ref_id_type: 근거 문헌
    -  Species Taxonomic Information: 유래종의 taxonomy


[CMAPUP DB](https://bidd.group/CMAUP/download.html)
- 식물종에 집중한 DB
- DB 제공 정보 (.txt 파일 제공)
1. 일반정보: 물질 ID, 이름, PubChem 및 ChEMBL ID 정보
2. 구조정보: SMILES, InChi, InChi-Key
3. 실험데이터: 천연물, 단백질, 실험데이터(IC50, PC50 등)
4. 식물 종: 천연물이 포함되어 있는 식물 종 정보
5. 활성물질 정보: 활성이 있다고 알려진 물질 목록

- KEGG Pathway
    - 생물학적 기능과 작용 기전을 체계적으로 정리
    - 대사경로(Metabolic Pathway)
        - 세포 내에서 이루어지는 다양한 화학반응(대사 과정)을 서로 연결된 네트워크(지도) 형태로 나타냅니다.
        - 예: 당대사(Glycolysis), 시트르산 회로(TCA cycle), 아미노산 생합성, 지질대사 경로 등
    - 신호전달경로(Signaling Pathway) 
        - 세포막 수용체, 세포 내 단백질 인산화 효소(kinase) 등 신호전달에 관여하는 분자들을 연결하여, 호르몬·성장인자·사이토카인 등에 대한 반응 메커니즘을 한눈에 볼 수 있게 해줍니다.
        - 예: MAPK 경로, PI3K-Akt 경로, Wnt 경로 등
    - 질병 관련 경로(Disease Pathway) 
    - 약물(Drug) 관련 경로


- `7. CMAUPv2.0_download_Ingredient_Target_Associations ActivityValues References`가 유용
    - 성분과 타겟간의 실험 정보
    - Ingredient ID : 식물
    - Target ID : 단백질
    - Reference ID : 문헌에 대한 정보

[KNApSAcK DB](https://www.knapsackfamily.com/KNApSAcK_Family/)
- 직접 검색해서 크롤링해야함
- url 앞부분 패턴은 고정이고 뒷부분 번호만 달라지는것을 활용

- [Biolgoical Activity](https://www.knapsackfamily.com/BiologicalActivity/top.php)
    - Biolgoical Activity에는 알려진 종의 활성 정보도 있음
    - 종, 관련된물질, 효능을 연결한것
    - 종에 대한 활성 정보를 얻기 쉽지 않는데 이걸 그나마 사용할 수 있음
    - CoreLink : 종에 포함된 성분물질
