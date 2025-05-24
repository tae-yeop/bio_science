펩타이드 약물 (Therapeutic Peptides)

- 아미노산 서열로 이뤄진 짧은 단백질
- 생체 내에서 다양한 기능
- 높이 특이성과 낮은 부작용
- poly-degradation(분해) -> 반감기가 너무 짧아 -> 도달하기전에 분해


다른 약물과 비교하면 small moleculde은 세포막 투과가 잘되긴 하지만 특정 타켓에 대한 특이성이 떨어진다. 예를 들어 카이나이제에 대해서만 바인딩하지 않음. 단백질 깁나 약물은 특이성이 높으나 안전성과 투과성이 떨어진다. 펩타이드 약물은 이 둘의 중간이다. 

분류
- linear peptide : 분해잘됨
- cyclic peptide : 분해 안되게
- conjugtaed peptide : 약물을 추가
- 어디서 왔는지
    - Native (체내에 있는 원서열) : 반감기
    - Analog (최적화 -> 안정적 + 반감기)
    - Heterologous (완전히 새로운)


펩타이드는 체내 물에 의해 계속 변화 -> structure prediction이 samll molecule과 달리 차원이 다르게 어렵다.

설계는 전통적 설계는 실험 데이터와 경험을 바탕으로 서열 변형을 하고 컴퓨터 기반 설계는 분다 도킹, 분자 동역학을 통해 서열 최적화 

딥러닝 기반 펩타이드 예측 모델의 분류
- 구조 예측
- 펩타이드 설계 모델
- 결합 및 상호작용 모델 (activity)

대규모 데이터는 어디서? : Protein-Protein Inetrcation을 활용



PLM : Protein Language Model
ESMFold
OmegaFold


PLM(Preotein Language Model)
아직까진 optimization 할때 physics-based method 필요

MIC (마이크로 그램 / mL)

PSSM : 얼마나 인접하는지를


펩타이드를 네이티브로 쓰지 않는다 -> modification을 거친다
-> 화학적 modification 이후 아미노산 시퀀스 x
-> all atom model들이 유용하지 않을까 하는 생각

trRosetta
