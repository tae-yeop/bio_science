# 단백질 구조 예측


기존의 단백질 구조 예측 방식

(1) Template-based

- Comparative Modeling : Indeitity sequence가 50~30% 정도일시 사용
- Fold recognition : 25% 이하이거나 sequence alignment가 안될 때

핵심원리 : **homology implies structural similarities.**
- Homology가 있으면 유사성이 있다는 사실. 
- homologs : 진화적으로 관련있는 DNA/proteins을 말함


구조적 유사도가 높은지를 통해 단백질이 homologs인지 판단한다.

X 축: aling 맞춘 residue 갯수
이때 identitcal residue 비율이 30%보다 높으면 homologs라고 본다 -> Comparative 방법 사용
20% 이하이면 판단 어려움 -> Fold recognition 사용

순서
(1) Sequence alignment
하는 역할이 같은 두 protein에 대해서 alignment를 수행
BLOSUM-62 같은 것을 사용함
(substitution matrix)

remote(distant) homologs : 진화적 거리가 멈 (20~30%, GAP 큼)

(2) Sequence Search


PDB에서 가장 가까운 것 찾기

어떤 Template + Alignment가 잘 맞춰져 있느냐 => 이 두개가 큰 역할

framework = backbone

Model optimization에선 MD 시뮬레이션, Energy optimization을 수행함

(2) Phsyics-based
    - Ab inition prediction : 없거나 찾기 힘든 경우, free energy minimization, 어렵고 정확도 낮음



### Modeller
Spatial constraints : 특정 아미노산끼리 가까운지 등의 조건을 최대한 많이 끄집어 내기

PDB 전체 데이터를 봤을 때 특성 ($C\alpha-C\alpha$) rksdml xmrwjd


CHARMM : MD 시뮬레이션

backbone의 구조는 dihedral angles로 부터 결정됨.

템플릿은 missing이 최대한 적어야함. missing이 있을 수 있음 실험으로 얻은거라서

Ramachandran plot에서 벗어나는지 확인 (골격 backbone, diherdral angle )


BLAST로 돌렸는데 잘 나오지 않으면 close homologs가 없다


결과 해석

E-value : 신뢰도

점 갯수가 많으면 비슷한 아미노산

모노머
멀티머



실험 ground truth

(1) Cryo-EM structure : 분자 복합체 (리보솜 등) ~3옹스트롱까지
(2) Crystal structure : X-ray 결정학으로 단백질을 결정화 -> 회절 패턴으로 좌표 추정
(3) NMR structure : 용액 상태, 작은 단백질에 강점




## 생물학적 도메인 지식
### Co-evolution
- 미오글로빈은 사람, 동물 모두 유사함
- 비둘기와 사람은 전혀 다르지만 단백질을 유사한 부분이 있다

진화를 해오면서 산소를 옮기는 같은 똑같은 역할을 하는 것만 남아서

Co-evolution이 있으면 contact를 하고 있다.

MSA를 언어 모델로 바꾸어도 알파폴드와 그렇게 떨어지지 않는다는 결과가 있다.

Co-evolution은 Attention에 반영된다

참조한 것이 어떻게 연결될지에 대한 정보가 여기에 있어서

Attention map을 모두 concat해서 linear에 넣는다


# 모델링
- 다양한 데이터셋에서 피처를 얻고 각각에 대한 임베딩으로 부터 임베딩을 얻어서 concat후 인코더에 넣자

![Image](https://github.com/user-attachments/assets/d857be63-3b80-4935-8afc-f65f39d85f7d)




RoseTTAFold

- 알파폴드에 기반하여 더 효율적으로 예측
- 알파폴드는 마지막에만 3D 정보를 내놓고 3D 정보를 계속 만들어서 인풋으로 사용
- 반면에 피처를 뽑아낼때 3차원정보를 참여해서 사용
- 알파폴드의 refine step이 없어서 빠르다

고정확도 모델로 할 수 있는 것들

- 단백질 구조를 실험구조
- x-ray로 쓰는 경우 molecular replacement 테크닉이 자주 사용됨
    - 이 방법을 사용하기 위해선 실제 단백질과 상당히 유사한 모델 구조가 있어야 함
    - 과거에는 단백질 DB에서 가장 가까운 녀석의 homology를 이용해서 replacement 문제를 해결
    - 좋은 homology, 좋은 주형 단백질 없는 경우 ⇒ 답을 찾기 힘들었다
- 고정확도 모델을 통해 molecular replacement  문제를 해결하고 최종 결정 구조를 얻을 수 있다
- 파란색, 빨간색 : 예측한 부분, 회색은 실제

### 알파폴드

- CASP13에 등장


## ESM

신약 개발을 위한 타겟 단백질 구조 예측

GPCR : 신약 개발에 가장 많이 사용되는 타겟 단백질

- GPCR(G-Protein Coupled Receptors, G 단백질 결합 수용체)
- GPCR은 세포막을 가로지르는 단백질로, 외부 신호를 세포 내부로 전달하는 역할을 합니다. 이들은 다양한 생리적 과정에 관여
    - **신호 전달**: GPCR은 호르몬, 신경전달물질, 감각 신호 등 다양한 외부 자극을 인식하고 이를 세포 내부로 전달합니다.
    - **다양성**: 인간 게놈에서 약 800여 개의 GPCR 유전자가 확인되었으며, 이는 다양한 생리적 기능과 관련되어 있습니다.
    - **약물 타겟**: 현재 시장에 출시된 약물 중 약 30-50%가 GPCR을 타겟으로 합니다. 이는 고혈압, 심장질환, 정신질환 등 여러 질환의 치료에 사용됩니다.
- inactive vs active 두 가지 state를 가지는 것으로 알려짐
    - active가 선호되도록 하는 약인지 inactive가 선호되도록 하는 약인지에 따라 lead optimization에 사용되는 타겟 단백질이 달라져야 함
- RoesTTAFold를 통해 특정 state를 가지는 단백질을 만들어 낼 수 있었다
- 각 state에 맞는 단백질을 통해 버추어 스크리닝, lead optimization을 진행하면 신약 개발에 도움

복합체 구조도 가능

- 단백질 구조 예측은 co-evolution으로 부터 예측을 해왔다
- co-evolution이 기능을 유지하기 위해 일어난다
- co-evolution은 한 단백질에서만 일어나는게 아니다
- 어떤 단백질이 다른 단백질과 상호작용하는 것이 살아남는 데에 중요한 기능을 한다면 ⇒ 서로 다른 단백질 사이에도 co-evoltuon이 일어남

- 기존의 SWISS-MODEL으로 얻은 결과는 density map과 잘 맞지 않았다.
- 반면 RoseTTAFold는 좋은 arrangement를 보여줌
    - 기질 특이성을 가지는 부분을 예측


## trRosetta

- 알파폴드 이후로 CNN 구조를 많이 사용
- MSA의 서열 갯수가 적은 것도 Native (gt)와 유사한 예측을 수행
- trRosetta는 MSA 정보만 사용



## trRosetta2 (2020)
- 주형 단백질 정보 + 모델 평가 방법
- 더 나은 distance, orientation을 예측
- MSA 정보로 부터 주형 단백질을 먼저 찾고 관련 정보를 이미지 형태로 만들어서 인풋
    - 주형 단백질 정보가 존재하는 경우 ⇒ 이용하는게 도움이 된다
- 그런데 타겟 단백질에 여러 도메인이 존재하여 일부만 주형 단백질이 존재하고 아닌 경우도 있는 경우
    - 여러 모델과 각 모델의 예측 정보를 받아들이는 최종 네트워크가 있다
- Rebetta 웹에서 사용 가능
    - 예측에는 몇 시간~몇일이 걸린다
    - confidence값이 0.9 이상이면 실험 상에서 얻는 구조와 거의 동등함
    - 0.8정도면 지엽적인 부분을 제외하면 상당히 정확
    - 0.6정도면 전체적인건 맞는 수준



## AlphaFold v2(2020)

### 개요

Inpust seq를 통해서 Genetic DB에서 homoglog를 찾는다
homolog와 input seq를 이용해서 MSA을 얻고 3차원 텐서로 이뤄진 MSA represtation을 얻는다.

single representation은 homolog 정보들이 담긴 effective한 representation임.

Structure DB (PDB)에서 템플릿을 얻는다. 여기서 아미노산간의 거리 정보를 얻어서 pair representation을 만든다.


- CASP14에 등장
- 어텐션 기반의 단백질 구조 예측
    - MSA 상호 정보를 on the fly로 추출
    - 2차원 정보와 상호 정보를 attention으로 계속 refine
- 최종적은 pair 정보와 빨간색을 이용해서 3차원 equivalent한 strucutre moudle을 통해 3차원 좌표값을 예측
- 실제 3차원 구조를 예측하도록 최적화됨
- 알파폴드를 통해 얻어낸 모델들을 데이터베이스화 함 : ([https://alphafold.ebi.ac.uk](https://alphafold.ebi.ac.uk/))
- 위의 DB에 없다면 직접 예측 : ColabFold, 기존 알파폴드를 완벽히 따르지 않아 성능 차이가 있음 : https://github.com/sokrypton/ColabFold


### 파이프라인

(1) MSA 서치
`HMMER`를 써서 homolog를 찾음
`HH`를 써서 찾아내느 homologues가 PDB에서 3D structure를 가지는지 체크




Correlated mutation = coevolution (이유: 구조를 유지하기 위해)

Contact 정보 : 공간적으로 붙어있는지 알 수 있다.

Genomic DB(UniRef90, Unuclust30)
Meta Genomic DB(BFP, Mgnify)

PDB70 : 템플릿으로 부터 레지듀(아미노산 하나) 쌍


Triangular attention

Distance matrix까지 나오면 3차원 주고 모델링

(2) EvoFormer

48개의 Block으로 구성된 Transformer 모델




### 결과
- Free-modeling category (템플릿 없어도) 87.0 GDT 달성



## 오픈소스 알파폴드
- Chai-1 : https://github.com/chaidiscovery/chai-lab
- Boltz-1 : https://github.com/jwohlwend/boltz
- Protenix : https://github.com/bytedance/Protenix


- [Lessons from implementing AlphaFold3 in the wild]()
실제 스크래치에서 구현하신 분께서 배웠던 점에 대한 강의
구현한 [깃허브 레포](https://github.com/Ligo-Biosciences/AlphaFold3/tree/main)도 오픈소스
- [The Illustrated AlphaFold](https://elanapearl.github.io/blog/2024/the-illustrated-alphafold/)
상세하게 각 콤퍼넌트별로 정리해줌. 
여긴 좀 더 보충 설명이나 실제 코드 구현적인 부분을 위주로 설명



## Backbone Strucurue Prediction

$\phi_i, \psi_i, w_i$ 는 거의 고정되어있다고 가정
rigid body (N-Cα-C)라고 생각 => degree of freedom 줄임 => 예측 쉽게



# Protein Design

function -> structure -> seqeunce
sequence를 알고 나서 AF를 사용하여 structure를 확인하여 검증


small molecule = ligand

docking 성공 = RMSD < 2 옹스트롱

AF2의 binding site가 정확하지 않다

num_relax : 생성된 molecule dynamics 설정


Rosetta : 옛날 방버으로 안정된 구조를 설계할 수 있게 해줌
