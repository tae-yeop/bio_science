# RNA 특징


## RNA 염기의 기본 구성

1) 질소 염기(base) – A, U, G, C 중 하나
2) 오탄당 (pentose sugar) – 리보스(ribose)
3) 인산기 (phosphate group)


single-stranded이지만 실제로는 secondary structure를 가진다.

non-coding RNA
- 단백질로 번역되지 않는 RNA
- 다음이 대표적인 ncRNA

tRNA
- 아미노산을 리보솜에 운반
rRNA
- 리보솜(Ribosome)의 구성 성분
- 세포질(Rough Endoplasmic Reticulum)에 흩어져 있음

snRNA / snoRNA
- 스플라이싱, 리보솜 RNA 수정

miRNA / siRNA
- 유전자 발현 조절
lncRNA
- 다양한 기능 (구조, 조절 등)



## structured RNA
- 3차원 구조를 갖는다는 뜻
- 자체적으로 접힘
- 내부의 **염기쌍 결합(base pairing)**을 통해 이중나선 구조나 hairpin, stem-loop 같은 구조를 형성



strucuture ncRNA는 자체적으로 접혀서 double-stranded가 된다
항상은 아니지만 예측하기 쉬운 제약조건을 제공함

짧은 RNA는 가능한 형태가 제한되어 예측이 쉬움
긴 RNA는 degree-of-freedom이 너무 높아짐

기본적인 염기쌍 형성 원리 (Hydrogen bonding)
- 정상 염기쌍 (canonical): A-U, G-C
- 비정상 염기쌍 (wobble): G-U 가능


염기쌍의 거리 제약과 예측 이점
- C1'(C1-prime)은 리보스 당의 첫번째 탄소 원자
    - 이 원자는 각 RNA 염기(nucleotide)가 다른 염기와 연결
    - 염기(base)와도 결합하는 중심 지점
    - 3D 좌표 예측에서 매우 중요한 기준점
- paired nucleotide의 C1'-C1' 거리는 ~10.5 Å (표준편차 ~0.3)
- base pair를 잘 맞추는 것만으로도 구조 제약을 확정






hairpin 구조는 예측이 쉬운 기본 단위이며, 많은 구조 RNA가 이를 반복
base pair 간 sequence 거리(긴거리 pairing)는 구조 예측의 난이도 증가 요소

# Co-variation(MSA)

두 RNA가 **공변이(mutual mutation)**를 겪으면 base pair가 보존될 수 있음






# 전략

Base pair 예측 성공시 C1 조회 가능 → 구조 생성 간접 정보 확보
BPPM or dot-bracket structures을 추가 feature로 써서 sequence data를 augment하자
-> helices and loops 정보 반영


LLM + Unsupervised 으로도 구조적 시그널을 학습

embeddings encode long-range dependencies (base pairing)
external structure predictors (like RNAfold/EternaFold) can propagate errors


Supervised multi-task approaches are limited by the availability and accuracy of known RNA structures



host가 제공하는거
RibonanzaNet 파인튜닝
RibonanzaNet2 코드 제공




쓸만한 아이디어
DRfold2
- RNA Composite Language Model (RCLM) 사용



Protein 예측하는 모델 파인튜닝?
- Transfer learning을 기대하는건데 단백질 예측과 RNA는 좀 다를 듯



RNAformer

