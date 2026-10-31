#!/usr/bin/env python
"""Drop-in replacement for DynamicBind's esm/scripts/extract.py (ESM-2 650M, layer 33, per_tok).

SUBSTITUTION (disclosed): upstream pulls fair-esm weights from dl.fbaipublicfiles.com, which this box is not
allowed to reach. We load the same model, facebook/esm2_t33_650M_UR50D, from the official Hugging Face Hub and
write the identical output format ({"label", "representations": {33: [L, 1280]}} per fasta record).
HF's last_hidden_state already includes the final emb_layer_norm_after, matching fair-esm layer-33 output.

usage: hf_esm_extract.py fasta_file output_dir [--device cuda]
"""
import argparse
import pathlib

import torch
from transformers import AutoTokenizer, EsmModel

MODEL = "facebook/esm2_t33_650M_UR50D"


def read_fasta(path):
    label, seq, out = None, [], []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if label is not None:
                out.append((label, "".join(seq)))
            label, seq = line[1:].split()[0], []
        elif line:
            seq.append(line)
    if label is not None:
        out.append((label, "".join(seq)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fasta_file")
    ap.add_argument("output_dir")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--truncation_seq_length", type=int, default=10000)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = EsmModel.from_pretrained(MODEL, add_pooling_layer=False).eval().to(args.device)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = read_fasta(args.fasta_file)
    print(f"Read {args.fasta_file} with {len(recs)} sequences (HF {MODEL})")
    with torch.no_grad():
        for label, seq in recs:
            seq = seq[: args.truncation_seq_length]
            enc = tok(seq, return_tensors="pt").to(args.device)
            h = model(**enc).last_hidden_state[0, 1 : len(seq) + 1].float().cpu().clone()
            assert h.shape[0] == len(seq), (h.shape, len(seq))
            torch.save({"label": label, "representations": {33: h}}, out_dir / f"{label}.pt")
            print("saved", label, tuple(h.shape))


if __name__ == "__main__":
    main()
