"""RDKit descriptor computation -> schema.Descriptors (+ alert hits)."""
from __future__ import annotations

import os
import sys
import threading
from collections import OrderedDict
from typing import List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors as D, QED, RDConfig, rdMolDescriptors

from studio.schema import AlertHit, Descriptors
from studio.scoring.alerts import find_alerts


def _load_sascorer():
    """Ertl SA scorer: RDKit Contrib SA_Score (needs its fpscores.pkl.gz), else repo fallback."""
    contrib = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if os.path.exists(os.path.join(contrib, "sascorer.py")) and os.path.exists(os.path.join(contrib, "fpscores.pkl.gz")):
        sys.path.append(contrib)
        try:
            import sascorer  # type: ignore

            return sascorer.calculateScore, "rdkit-contrib:fpscores.pkl.gz"
        except Exception:
            pass
    try:
        from synthetic_accessibility import sa_score  # repo copy; reads its own data file

        fn = getattr(sa_score, "calculateScore", None)
        if fn is not None:
            fn(Chem.MolFromSmiles("CCO"))
            return fn, "repo:synthetic_accessibility"
    except Exception:
        pass
    return None, "unavailable"


_SA_FN, SA_SOURCE = _load_sascorer()  # SA_SOURCE reports which scorer/data file is in use


def compute_descriptors(mol: Chem.Mol) -> Tuple[Descriptors, List[AlertHit]]:
    alerts = find_alerts(mol)
    mw = D.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    lip = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
    sa: Optional[float] = None
    if _SA_FN is not None:
        try:
            sa = float(_SA_FN(mol))
        except Exception:
            sa = None
    desc = Descriptors(
        qed=float(QED.qed(mol)), sa=sa, logp=float(logp), tpsa=float(rdMolDescriptors.CalcTPSA(mol)),
        mw=float(mw), hbd=hbd, hba=hba, rotb=rdMolDescriptors.CalcNumRotatableBonds(mol),
        fsp3=float(rdMolDescriptors.CalcFractionCSP3(mol)), heavy_atoms=mol.GetNumHeavyAtoms(),
        lipinski_violations=lip, alerts=len(alerts),
    )
    return desc, alerts


class LRU:
    """Minimal thread-safe LRU."""

    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._d: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                self.hits += 1
                return self._d[key]
            self.misses += 1
            return None

    def put(self, key, value):
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)

    def __len__(self):
        return len(self._d)
