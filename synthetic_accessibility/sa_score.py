import math
import os.path as op
import pickle
from collections import defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
import sys

_fscores = None
_DATA_ROOT = Path(__file__).resolve().parent / "Metis_Data"

class SyntheticAccessibility(object):
  @staticmethod
  def readFragmentScores(name='fpscores'):
    import gzip
    global _fscores
    if name == "fpscores":
      name = str(_DATA_ROOT / "sa_score_data" / name)
    data = pickle.load(gzip.open('%s.pkl.gz' % name))
    outDict = {}
    for i in data:
      for j in range(1, len(i)):
        outDict[i[j]] = float(i[0])
    _fscores = outDict

  @staticmethod
  def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro

  @staticmethod
  def calculateScore(m):
    if _fscores is None:
      SyntheticAccessibility.readFragmentScores()

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = morgan_gen.GetFingerprint(mol)  
    fps = fp.GetNonzeroElements()

    score1 = 0.
    nf = 0
    for bitId, v in fps.items():
      nf += v
      sfp = bitId
      score1 += _fscores.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = SyntheticAccessibility.numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
      if len(x) > 8:
        nMacrocycles += 1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
      macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
      score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    # smooth the 10-end
    if sascore > 8.:
      sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
      sascore = 10.0
    elif sascore < 1.:
      sascore = 1.0

    return sascore

  @staticmethod
  def processSMILES(smiles):
      #print('smiles\tsa_score')
      sa_scores = []
      for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
          print('Invalid SMILES: ', smile)
        s = SyntheticAccessibility.calculateScore(mol)
        sa_scores.append(s)

      return sa_scores

if __name__ == "__main__":
    smiles = sys.argv[1]
    predictor = SyntheticAccessibility()
    score = predictor.processSMILES(smiles)
    print(score)