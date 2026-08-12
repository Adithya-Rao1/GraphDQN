import torch
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
import re
import onnxruntime
import numpy as np
from typing import List, Dict, Union
from diskcache import Cache
from tqdm import tqdm
from contextlib import contextmanager, nullcontext
import os
from pathlib import Path

_DEFAULT_ONNX_PATH = str(Path(__file__).resolve().parent / "models" / "affinity_predictor.onnx")

class PredictionModule:
    def __init__(self, model_path: str = _DEFAULT_ONNX_PATH):
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.mean = 6.51286529169358
        self.scale = 1.5614094578916633

    def convert_to_affinity(self, normalized: float) -> Dict[str, float]:
        neg_log10_affinity_M = float((normalized * self.scale) + self.mean)
        affinity_uM = float((10**6) * (10**(-neg_log10_affinity_M)))
        return {
            "neg_log10_affinity_M": neg_log10_affinity_M,
            "affinity_uM": affinity_uM
        }

    def predict(self, batch_data: np.ndarray) -> List[Dict[str, float]]:
        affinities = []
        for feature in batch_data:
            affinity_normalized = self.session.run(None, {self.input_name: [feature], 'TrainingMode': np.array(False)})[0][0][0]
            affinities.append(self.convert_to_affinity(affinity_normalized))
        return affinities

class Plapt:
    def __init__(self, prediction_module_path: str = _DEFAULT_ONNX_PATH, device: str = 'cuda', cache_dir: str = './embedding_cache', use_tqdm: bool = False):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_tqdm = use_tqdm
        
        self.prot_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self.prot_encoder = BertModel.from_pretrained("Rostlab/prot_bert").to(self.device)
        
        self.mol_tokenizer = RobertaTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        self.mol_encoder = RobertaModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1").to(self.device)
        
        self.prediction_module = PredictionModule(prediction_module_path)
        self.cache = Cache(cache_dir)

    @contextmanager
    def progress_bar(self, total: int, desc: str):
        if self.use_tqdm:
            with tqdm(total=total, desc=desc) as pbar:
                yield pbar
        else:
            yield nullcontext()

    @staticmethod
    def preprocess_sequence(seq: str) -> str:
        return " ".join(re.sub(r"[UZOB]", "X", seq))

    def tokenize_molecule(self, mol_smiles: Union[str, List[str]]) -> torch.Tensor:
        return self.mol_tokenizer(mol_smiles, padding=True, max_length=278, truncation=True, return_tensors='pt')

    def tokenize_protein(self, prot_seq: Union[str, List[str]]) -> torch.Tensor:
        preprocessed = [self.preprocess_sequence(seq) if isinstance(seq, str) else self.preprocess_sequence(seq[0]) for seq in prot_seq]
        return self.prot_tokenizer(preprocessed, padding=True, max_length=3200, truncation=True, return_tensors='pt')

    def encode_molecules(self, mol_smiles: List[str], batch_size: int) -> torch.Tensor:
        embeddings = []
        with self.progress_bar(len(mol_smiles), "Encoding molecules") as pbar:
            for batch in self.make_batches(mol_smiles, batch_size):
                cached_embeddings = [self.cache.get(smiles) for smiles in batch]
                uncached_indices = [i for i, emb in enumerate(cached_embeddings) if emb is None]
                
                if uncached_indices:
                    uncached_smiles = [batch[i] for i in uncached_indices]
                    tokens = self.tokenize_molecule(uncached_smiles)
                    with torch.no_grad():
                        new_embeddings = self.mol_encoder(**tokens.to(self.device)).pooler_output.cpu()
                    for i, emb in zip(uncached_indices, new_embeddings):
                        cached_embeddings[i] = emb
                        self.cache[batch[i]] = emb
                
                embeddings.extend(cached_embeddings)
                if self.use_tqdm:
                    pbar.update(len(batch))
        
        return torch.stack(embeddings).to(self.device)

    def encode_proteins(self, prot_seqs: List[str], batch_size: int) -> torch.Tensor:
        embeddings = []
        with self.progress_bar(len(prot_seqs), "Encoding proteins") as pbar:
            for batch in self.make_batches(prot_seqs, batch_size):
                cached_embeddings = [self.cache.get(seq) for seq in batch]
                uncached_indices = [i for i, emb in enumerate(cached_embeddings) if emb is None]
                
                if uncached_indices:
                    uncached_seqs = [batch[i] for i in uncached_indices]
                    tokens = self.tokenize_protein(uncached_seqs)
                    with torch.no_grad():
                        new_embeddings = self.prot_encoder(**tokens.to(self.device)).pooler_output.cpu()
                    for i, emb in zip(uncached_indices, new_embeddings):
                        cached_embeddings[i] = emb
                        self.cache[batch[i]] = emb
                
                embeddings.extend(cached_embeddings)
                if self.use_tqdm:
                    pbar.update(len(batch))
        
        return torch.stack(embeddings).to(self.device)

    @staticmethod
    def make_batches(iterable: List, n: int = 1):
        length = len(iterable)
        for ndx in range(0, length, n):
            yield iterable[ndx:min(ndx + n, length)]

    def predict_affinity(self, prot_seqs: List[str], mol_smiles: List[str], prot_batch_size: int = 2, mol_batch_size: int = 16, affinity_batch_size: int = 128) -> List[Dict[str, float]]:
        if len(prot_seqs) != len(mol_smiles):
            raise ValueError("The number of proteins and molecules must be the same.")

        prot_encodings = self.encode_proteins(prot_seqs, prot_batch_size)
        mol_encodings = self.encode_molecules(mol_smiles, mol_batch_size)

        affinities = []
        with self.progress_bar(len(prot_seqs), "Predicting affinities") as pbar:
            for batch in self.make_batches(range(len(prot_seqs)), affinity_batch_size):
                prot_batch = prot_encodings[batch]
                mol_batch = mol_encodings[batch]
                features = torch.cat((prot_batch, mol_batch), dim=1).cpu().numpy()
                batch_affinities = self.prediction_module.predict(features)
                affinities.extend(batch_affinities)
                if self.use_tqdm:
                    pbar.update(len(batch))

        return affinities

    def score_candidates(self, target_protein: str, mol_smiles: List[str], mol_batch_size: int = 16, affinity_batch_size: int = 128) -> List[Dict[str, float]]:
        target_encoding = self.encode_proteins([target_protein], batch_size=1)
        mol_encodings = self.encode_molecules(mol_smiles, mol_batch_size)

        affinities = []
        with self.progress_bar(len(mol_smiles), "Scoring candidates") as pbar:
            for batch in self.make_batches(range(len(mol_smiles)), affinity_batch_size):
                mol_batch = mol_encodings[batch]
                repeated_target = target_encoding.repeat(len(batch), 1)
                features = torch.cat((repeated_target, mol_batch), dim=1).cpu().numpy()
                batch_affinities = self.prediction_module.predict(features)
                affinities.extend(batch_affinities)
                if self.use_tqdm:
                    pbar.update(len(batch))

        return affinities
    
def run_predictions(plapt, prot_seqs, mol_smiles, mol_batch_size=16, affinity_batch_size=128):
    scores = plapt.score_candidates(prot_seqs, mol_smiles, mol_batch_size, affinity_batch_size)
    
    affinities = []
    for score in scores:
        affinities.append(score['affinity_uM'])
    
    return affinities
    
if __name__ == "__main__":
    plapt = Plapt()
    
    # Example for predict_affinity
    proteins = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG", 
                "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
    molecules = ["CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F", 
                 "COC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F"]
    
    results = plapt.predict_affinity(proteins, molecules, prot_batch_size=2, mol_batch_size=16, affinity_batch_size=128)
    print("\nPredict Affinity Results:", results)
    
    # Example for score_candidates
    target_protein = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    candidate_molecules = ["CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F", 
                           "COC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",
                           "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F"]
    
    scores = plapt.score_candidates(target_protein, candidate_molecules, mol_batch_size=16, affinity_batch_size=128)
    print("\nScore Candidates Results:", scores)

    target_protein = "MAEPRQEFEVMEDHAGTYGLGDRKDQGGYTMHQDQEGDTDAGLKESPLQTPTEDGSEEPGSETSDAKSTPTAEDVTAPLVDEGAPGKQAAAQPHTEIPEGTTAEEAGIGDTPSLEDEAAGHVTQEPESGKVVQEGFLREPGPPGLSHQLMSGMPGAPLLPEGPREATRQPSGTGPEDTEGGRHAPELLKHQLLGDLHQEGPPLKGAGGKERPGSKEEVDEDRDVDESSPQDSPPSKASPAQDGRPPQTAAREATSIPGFPAEGAIPLPVDFLSKVSTEIPASEPDGPSVGRAKGQDAPLEFTFHVEITPNVQKEQAHSEEHLGRAAFPGAPGEGPEARGPSLGEDTKEADLPEPSEKQPAAAPRGKPVSRVPQLKARMVSKSKDGTGSDDKKAKTSTRSSAKTLKNRPCLSPKHPTPGSSDPLIQPSSPAVCPEPPSSPKYVSSVTSRTGSSGAKEMKLKGADGKTKIATPRGAAPPGQKGQANATRIPAKTPPAPKTPPSSGEPPKSGDRSGYSSPGSPGTPGSRSRTPSLPTPPTREPKKVAVVRTPPKSPSSAKSRLQTAPVPMPDLKNVKSKIGSTENLKHQPGGGKVQIINKKLDLSNVQSKCGSKDNIKHVPGGGSVQIVYKPVDLSKVTSKCGSLGNIHHKPGGGQVEVKSEKLDFKDRVQSKIGSLDNITHVPGGGNKKIETHKLTFRENAKAKTDHGAEIVYKSPVVSGDTSPRHLSNVSSTGSIDMVDSPQLATLADEVSASLAKQGL"
    candidate_molecules = ["N#CC1=C(C(F)(F)F)C([N+]([O-])=O)=C(C2=CC=CC([N+]([O-])=O)=C2)NC1=O", 
                           "CN1[C@@](N(C)CC2)([H])[C@@]2(C)C3=C1C=CC(OC(NC4=CC=CC=C4)=O)=C3",
                           "S=C(NC1=CC=CC2=C1C=CN2)NC3=CC=C(Cl)C=C3"]
    
    for candidate in candidate_molecules:
        affinities = run_predictions(Plapt(), target_protein, [candidate])
        print(affinities)