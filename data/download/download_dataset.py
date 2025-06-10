import argparse
import os
import requests
import pandas as pd
from multiprocessing import Pool

# 필요한 경우 생화학 데이터 파싱 라이브러리 임포트
from Bio.PDB import PDBParser, PPBuilder        # PDBbind 처리에 사용
from rdkit import Chem                          # 리간드 SMILES 추출에 사용 (RDKit)
from chembl_webresource_client.new_client import new_client  # ChEMBL API 클라이언트

# 개별 복합체(PDBbind)의 정보를 처리하는 함수
def process_pdbbind_complex(complex_id, base_dir):
    """하나의 PDBbind 복합체에 대해 (protein_sequence, ligand_smiles, affinity) 추출"""
    pdb_id = complex_id
    result = {"PDB_ID": pdb_id, "protein_sequence": None, "ligand_smiles": None, "affinity": None}
    try:
        # PDB 파일 경로 (예: base_dir/PDBbind/v2020/XXXX/XXXX_protein.pdb 등)
        pdb_file = os.path.join(base_dir, pdb_id, f"{pdb_id}_protein.pdb")
        ligand_file = os.path.join(base_dir, pdb_id, f"{pdb_id}_ligand.mol2")  # 또는 .sdf 등
        
        # 단백질 서열 추출 (Biopython 사용)
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, pdb_file)
        ppb = PPBuilder()
        seq = []
        for peptide in ppb.build_peptides(structure):
            seq.append(str(peptide.get_sequence()))
        result["protein_sequence"] = "".join(seq)
        
        # 리간드 SMILES 추출 (RDKit 사용)
        ligand_mol = None
        if os.path.exists(ligand_file):
            # mol2 파일을 RDKit으로 불러오기
            ligand_mol = Chem.MolFromMol2File(ligand_file, sanitize=True)
        else:
            # 만약 별도 리간드 파일이 없으면 PDB에서 HETATM 추출하여 파싱
            with open(pdb_file, 'r') as pf:
                lines = pf.readlines()
            hetatm_lines = [line for line in lines if line.startswith("HETATM") and " HOH" not in line]
            ligand_block = "".join(hetatm_lines)
            ligand_mol = Chem.MolFromPDBBlock(ligand_block, sanitize=True)
        if ligand_mol:
            result["ligand_smiles"] = Chem.MolToSmiles(ligand_mol)
    except Exception as e:
        print(f"[WARN] Failed to process {pdb_id}: {e}")
    return result

# ZINC 화합물 SMILES 정제 함수
def clean_smiles(smiles):
    """SMILES 문자열 정제 및 유효성 검사"""
    # 공백 제거 및 대문자 변환 등 간단한 정제
    s = smiles.strip()
    # RDKit으로 유효한 구조인지 검사 (선택사항)
    mol = Chem.MolFromSmiles(s)
    if mol:
        s = Chem.MolToSmiles(mol, canonical=True)  # 정규화된 canonical SMILES
        return s
    else:
        return None  # 유효하지 않은 SMILES

# ChEMBL 화합물 정보 조회 함수
def fetch_chembl_compound(chembl_id):
    """ChEMBL ID로부터 SMILES와 표준 활성값 추출"""
    try:
        mol = new_client.molecule.get(chembl_id)    # 화합물 정보 조회
        smiles = mol['molecule_structures']['canonical_smiles'] if mol else None
        return (chembl_id, smiles)
    except Exception as e:
        print(f"[WARN] Failed to fetch compound {chembl_id}: {e}")
        return (chembl_id, None)

def main():
    parser = argparse.ArgumentParser(description="Dataset download and preprocessing script")
    parser.add_argument('--dataset', '-d', nargs='+', choices=['PDBbind','ZINC','ChEMBL'],
                        help='다운로드/처리할 데이터셋 선택 (복수 선택 가능)', required=True)
    parser.add_argument('--output_dir', '-o', default='.', help='결과 CSV를 저장할 경로')
    parser.add_argument('--target', '-t', type=str, help='ChEMBL 데이터셋 선택 시 표적 단백질 (예: CHEMBL ID)')
    parser.add_argument('--zinc_sample_size', type=int, default=10000, help='ZINC 샘플링 개수 (기본 10000)')
    args = parser.parse_args()
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. PDBbind 처리
    if 'PDBbind' in args.dataset:
        print("Downloading PDBbind dataset index and files...")
        # (예시) PDBbind 인덱스 파일 다운로드
        index_url = "http://www.pdbbind.org.cn/download/INDEX_general_PL_data_2020"  # 가정된 URL
        idx_path = os.path.join(output_dir, "INDEX_general_PL_data_2020")
        if not os.path.exists(idx_path):
            r = requests.get(index_url)
            with open(idx_path, 'wb') as f:
                f.write(r.content)
        # 인덱스 파일 파싱
        pdb_list = []
        affinity_dict = {}
        with open(idx_path, 'r') as f:
            lines = f.readlines()[6:]  # 헤더 스킵 (첫 5줄 등)
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    pdb_id = parts[0]
                    aff_val = parts[4]   # (예시: "5.5" or "15000.0")
                    affinity_dict[pdb_id] = aff_val
                    pdb_list.append(pdb_id)
        # (참고: 실제 인덱스에는 단위 등이 포함되어 있을 수 있어 추가 파싱 필요)
        # PDBbind 구조 데이터 다운로드 (대용량이므로 실제 구현에서는 tar.gz 받아 풀도록 최적화 가능)
        base_dir = os.path.join(output_dir, "pdbbind_files")
        os.makedirs(base_dir, exist_ok=True)
        for pdb_id in pdb_list:
            pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            pdb_file_path = os.path.join(base_dir, pdb_id, f"{pdb_id}_protein.pdb")
            if not os.path.exists(pdb_file_path):
                os.makedirs(os.path.join(base_dir, pdb_id), exist_ok=True)
                try:
                    resp = requests.get(pdb_url)
                    with open(pdb_file_path, 'wb') as f:
                        f.write(resp.content)
                except Exception as e:
                    print(f"[WARN] Failed to download PDB {pdb_id}: {e}")
            # (리간드 구조 파일 다운로드/추출 로직 추가 가능)
        
        # 병렬로 각 복합체 파싱
        print(f"Processing {len(pdb_list)} PDBbind complexes with multiprocessing...")
        with Pool() as pool:
            records = pool.starmap(process_pdbbind_complex, [(pid, base_dir) for pid in pdb_list])
        # 친화도 값 매핑
        for rec in records:
            pid = rec["PDB_ID"]
            rec["affinity"] = affinity_dict.get(pid, None)
        # 데이터프레임으로 저장
        df_pdb = pd.DataFrame(records)
        df_pdb.to_csv(os.path.join(output_dir, "pdbbind_data.csv"), index=False)
        print(f"PDBbind 처리 완료: {len(df_pdb)}개 레코드 저장")
    
    # 2. ZINC 처리
    if 'ZINC' in args.dataset:
        print("Downloading ZINC dataset...")
        # (예시) ZINC SMILES 리스트 다운로드 - 여기서는 가정적으로 Kaggle 등에서 받은 파일 사용
        zinc_file = os.path.join(output_dir, "zinc_smiles.txt")
        if not os.path.exists(zinc_file):
            # 실제 구현에서는 requests나 Kaggle API 등을 통해 데이터 확보
            example_zinc_url = "https://.../ZINC_smiles.txt"  # 가상 URL
            r = requests.get(example_zinc_url)
            with open(zinc_file, 'wb') as f:
                f.write(r.content)
        # SMILES 로드
        with open(zinc_file, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
        print(f"Total SMILES downloaded: {len(smiles_list)}")
        # 중복 제거
        smiles_list = list(set(smiles_list))
        print(f"After duplicate removal: {len(smiles_list)}")
        # 멀티프로세싱으로 SMILES 정제
        print("Cleaning SMILES strings with multiprocessing...")
        with Pool() as pool:
            cleaned = pool.map(clean_smiles, smiles_list)
        # 유효한 SMILES만 필터링
        cleaned = [s for s in cleaned if s is not None]
        # 샘플링 (요청된 크기만큼)
        sample_size = min(args.zinc_sample_size, len(cleaned))
        if sample_size < len(cleaned):
            import random
            cleaned = random.sample(cleaned, sample_size)
            print(f"Sampled {sample_size} SMILES from cleaned data")
        # 데이터프레임 생성 및 저장
        df_zinc = pd.DataFrame(cleaned, columns=["smiles"])
        df_zinc.to_csv(os.path.join(output_dir, "zinc_data.csv"), index=False)
        print(f"ZINC 처리 완료: {len(df_zinc)}개 SMILES 저장")
    
    # 3. ChEMBL 처리
    if 'ChEMBL' in args.dataset:
        if not args.target:
            print("[ERROR] ChEMBL 데이터셋을 선택한 경우 --target 인자로 표적을 지정해야 합니다.")
            return
        target_id = args.target
        print(f"Querying ChEMBL for target {target_id}...")
        # 표적에 대한 bioactivity 검색 (예: IC50 값들만 필터링)
        activities = new_client.activity.filter(target_chembl_id=target_id, type="IC50", relation="=")
        # Pandas DataFrame으로 변환
        df_act = pd.DataFrame(list(activities))
        if df_act.empty:
            print(f"No bioactivity data found for target {target_id}")
            return
        # 활성값 관련 컬럼만 선택 (필요에 따라 단위 통일 등 처리 가능)
        df_act = df_act[['molecule_chembl_id','type','standard_value','standard_units']]
        # 중복 화합물 제거
        mol_ids = df_act['molecule_chembl_id'].unique().tolist()
        print(f"Found {len(mol_ids)} unique compounds with activities for target {target_id}.")
        # 멀티프로세싱으로 각 화합물의 SMILES 가져오기
        print("Fetching compound structures from ChEMBL with multiprocessing...")
        with Pool() as pool:
            mol_results = pool.map(fetch_chembl_compound, mol_ids)
        # 결과를 데이터프레임으로 정리
        df_mol = pd.DataFrame(mol_results, columns=['molecule_chembl_id','smiles'])
        # 병합하여 최종 데이터 구성
        df_chembl = pd.merge(df_act, df_mol, on='molecule_chembl_id', how='left')
        # 필요 없는 중간 컬럼(drop molecule_chembl_id, type 등) 조정 및 pIC50 계산 예시
        # 여기서는 molecule_chembl_id 유지하고, pIC50 추가
        df_chembl['pIC50'] = df_chembl.apply(
            lambda row: (9 - np.log10(float(row['standard_value'])*1e-9)) if row['standard_units']=='nM' and pd.notnull(row['standard_value']) else None,
            axis=1
        )
        df_chembl.to_csv(os.path.join(output_dir, "chembl_data.csv"), index=False)
        print(f"ChEMBL 처리 완료: {len(df_chembl)}개 활성 데이터 저장 (표적 {target_id})")

if __name__ == "__main__":
    main()
