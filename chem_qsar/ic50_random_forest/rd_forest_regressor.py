import pandas as pd
import matplotlib.pyplot as plt

df_raw = pd.read_csv("jak2_data.csv")
print(df_raw.describe())

# 중복 row 제거
df = df_raw.drop_duplicates()
print(df.describe())

plt.figure()
df.hist("pIC50")
plt.savefig("pIC50_histogram.png")
plt.close()

# Decriptor 계산
# 1. Molecular Weight
# 2. Number of hydrogen bond acceptors. 
# 3. Number of hydrogen bond donors. 
# 4. logP
# 5. fraction of SP3 hybridized carbon (sp3 혼성을 가지는 탄소의 개수)
# 6. Number of rotatable bond. 
# 7. Number of rings. 
# 8. TPSA: polar surface area. (분자의 표면적 넓이, 단위: A^2)
# 9. Number of Aramatic Rings (방향성 고리의 개수)
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem

from rdkit.Chem.rdMolDescriptors import (
    CalcExactMolWt, # 분자량
    CalcCrippenDescriptors, # logP
    CalcNumLipinskiHBA, 
    CalcNumLipinskiHBD, 
    CalcFractionCSP3, 
    CalcNumRotatableBonds, 
    CalcNumRings, 
    CalcTPSA, 
    CalcNumAromaticRings
)

# empty dictionary for pandas
properties = {"MW":[], "LogP":[], "HBA":[], "HBD": [], 
              "CSP3": [], "NumRotBond": [], "NumRings": [], "TPSA": [], 
              "NumAromaticRings": [], "pIC50": []}

for idx, smiles in enumerate(df["SMILES"]): # Smiles 열에서 반복.
    mol = Chem.MolFromSmiles(smiles)
    if mol == None: # if molecule is not valid. mol안에 None이 들어있으면 문제가 있는 것!
        continue

    properties["MW"].append(CalcExactMolWt(mol))
    properties["LogP"].append(CalcCrippenDescriptors(mol)[0])
    properties["HBA"].append(CalcNumLipinskiHBA(mol))
    properties["HBD"].append(CalcNumLipinskiHBD(mol))
    properties["CSP3"].append(CalcFractionCSP3(mol))
    properties["NumRotBond"].append(CalcNumRotatableBonds(mol))
    properties["NumRings"].append(CalcNumRings(mol))
    properties["TPSA"].append(CalcTPSA(mol))
    properties["NumAromaticRings"].append(CalcNumAromaticRings(mol))
    properties["pIC50"].append(df["pIC50"].iloc[idx])


new_data = pd.DataFrame(properties)
print(new_data.describe())


X = new_data.iloc[:, :-1] # 전체행, 마지막열 직전까지를 입력 값으로 사용. 
y = new_data.iloc[:, -1]  # 전체행, 마지막열의 데이터를 목적 값으로 추출

print(X.head())
print(y.head())

import sklearn.model_selection
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.head())

# RandomForest
import sklearn.ensemble
from sklearn.ensemble import RandomForestRegressor

my_model = RandomForestRegressor()
my_model.fit(X_train, y_train)


y_pred = my_model.predict(X_test)
plt.scatter(y_test, y_pred, marker='.')
plt.xlabel("Experimental pIC50")
plt.ylabel("Predicted pIC50")
plt.grid()
plt.plot(range(4, 12), range(4, 12), "r--", label = "y=x")
plt.legend()
plt.savefig("scatter_plot_basic.png")


from sklearn.metrics import mean_squared_error

mse1 = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error (in pIC50 unit): {mse1:.3f}")