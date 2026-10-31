"""python -m studio.registry.selfcheck  -- run every op on every site of ~20 diverse molecules."""
from __future__ import annotations

import sys
import time
from collections import defaultdict

from rdkit import Chem

from studio.registry import get_registry

MOLECULES = [
    "CCOC(=O)c1ccc(cc1)[N+](=O)[O-]",            # mock seed
    "CC(=O)Oc1ccccc1C(=O)O",                      # aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",                 # ibuprofen
    "CC(=O)Nc1ccc(O)cc1",                         # paracetamol
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",                 # caffeine
    "CN1CCC[C@H]1c1cccnc1",                       # nicotine
    "c1ccc2c(c1)cc[nH]2",                         # indole
    "OC(=O)c1ccccc1O",                            # salicylic acid
    "CCN(CC)CCNC(=O)c1ccc(N)cc1",                 # procainamide
    "Cc1ccc(cc1)S(=O)(=O)N",                      # tosylamide
    "O=C(O)CCCc1ccccc1",                          # 4-phenylbutyric acid
    "CC(C)NCC(O)COc1cccc2ccccc12",                # propranolol
    "CC1=C(C(=O)N(N1C)c1ccccc1)N(C)C",            # aminophenazone
    "O=C1CCCN1",                                  # pyrrolidinone
    "C1CCNCC1",                                   # piperidine
    "N#Cc1ccc(cc1)C=O",                           # cyano-benzaldehyde
    "Clc1ccc(cc1)C(=O)NCc1ccccc1",                # benzamide
    "COc1ccc(cc1)C(=O)CSc1nnc(o1)C",              # thioether/ketone/oxadiazole
    "C[C@H](N)C(=O)O",                            # alanine (stereo)
    "OCC1OC(O)C(O)C(O)C1O",                       # glucose
    "N=C(N)c1ccccc1",                             # benzamidine
    "CCCCCCCC(=O)N1CCN(CC1)c1ccc(F)cc1",          # long chain amide/piperazine
]


def main() -> int:
    reg = get_registry()
    stats = defaultdict(lambda: dict(sites=0, valid=0, empty=0, raw_invalid=0, exc=0, nondet=0, mols=0))
    t0 = time.time()
    for smi in MOLECULES:
        assert Chem.MolFromSmiles(smi) is not None, smi
        for op in reg.all():
            s = stats[op.id]
            try:
                sites = op.sites(smi)
            except Exception:
                s["exc"] += 1
                continue
            s["mols"] += bool(sites)
            for site in sites:
                s["sites"] += 1
                try:
                    out = op.apply(smi, site)
                    again = op.apply(smi, site)
                    raw = op._raw(smi, site)
                except Exception:
                    s["exc"] += 1
                    continue
                if out != again:
                    s["nondet"] += 1
                if out:
                    s["valid"] += 1
                else:
                    s["empty"] += 1
                    if raw:  # wrapped class returned something the registry rejected
                        s["raw_invalid"] += 1
    print(f"{len(reg.all())} ops x {len(MOLECULES)} molecules in {time.time() - t0:.1f}s\n")
    print(f"{'op':32} {'mols':>4} {'sites':>6} {'valid':>6} {'empty':>6} {'rawbad':>6} {'exc':>4} {'nondet':>6}")
    never, always_fail, exc = [], [], 0
    for op in reg.all():
        s = stats[op.id]
        print(f"{op.id:32} {s['mols']:4} {s['sites']:6} {s['valid']:6} {s['empty']:6} {s['raw_invalid']:6} {s['exc']:4} {s['nondet']:6}")
        exc += s["exc"] + s["nondet"]
        if s["sites"] == 0:
            never.append(op.id)
        elif s["valid"] == 0:
            always_fail.append(op.id)
    print(f"\nnever applicable (0 sites on all molecules): {never}")
    print(f"100% failure (sites but 0 valid products): {always_fail}")
    print(f"exceptions + nondeterminism: {exc}")
    return 1 if exc else 0


if __name__ == "__main__":
    sys.exit(main())
