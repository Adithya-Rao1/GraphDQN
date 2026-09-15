import argparse
import base64
import io
import json
import os
import time

import torch
from rdkit import Chem
from rdkit.Chem import Draw, rdFMCS

from experiments.data.targets import TARGETS, DEFAULT_TARGET
from experiments.data.starting_molecules import sample_pilot_molecules
from reward.multi_objective import compute_reward
from ADMET.model import ADMETModel
from binding_module.binding_affinity.plapt import Plapt
from synthetic_accessibility.sa_score import SyntheticAccessibility

import dqn.dqn_hyperparams as reward_hp


def mol_image_base64(smiles, highlight_atoms=None, size=(280, 280)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=size, highlightAtoms=highlight_atoms or [])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def new_atoms_since(prev_smiles, curr_smiles):
    if not prev_smiles:
        return []
    prev_mol = Chem.MolFromSmiles(prev_smiles)
    curr_mol = Chem.MolFromSmiles(curr_smiles)
    if prev_mol is None or curr_mol is None:
        return []
    try:
        mcs = rdFMCS.FindMCS([prev_mol, curr_mol], timeout=5, matchValences=True, ringMatchesRingOnly=True)
        if mcs.canceled or mcs.numAtoms == 0:
            return []
        patt = Chem.MolFromSmarts(mcs.smartsString)
        match = curr_mol.GetSubstructMatch(patt)
        return [i for i in range(curr_mol.GetNumAtoms()) if i not in match]
    except Exception:
        return []


def score_smiles(smiles, target_seq, device, off_target_seq, admet_model, binding_model, sa_model):
    return compute_reward(
        smiles=smiles,
        target_seq=target_seq,
        device=device,
        off_target_seq=off_target_seq,
        admet_weight=reward_hp.admet_weight,
        binding_weight=reward_hp.binding_weight,
        synthetic_weight=reward_hp.synthetic_weight,
        selectivity_weight=reward_hp.selectivity_weight,
        admet_model=admet_model,
        binding_model=binding_model,
        sa_model=sa_model,
    )


def rollout_dqn(checkpoint, start_smiles, target_seq, off_target_seq, device, max_steps,
                 admet_model, binding_model, sa_model):
    from dqn.dqn_network import DKDQNAgent
    from dqn.all_envs import MultiObjectiveRewardEnv
    from dqn.utils import create_graph

    agent = DKDQNAgent(output_dim=1, device=device)
    state_dict = torch.load(checkpoint, map_location=device)
    agent.qn.load_state_dict(state_dict)
    agent.target_qn.load_state_dict(state_dict)  # get_action reads target_qn

    env = MultiObjectiveRewardEnv(
        discount_factor=reward_hp.discount_factor,
        device=device,
        init_mol=start_smiles,
        max_steps=max_steps,
        target_seq=target_seq,
        off_target_seq=off_target_seq,
        admet_model=admet_model,
        binding_model=binding_model,
        sa_model=sa_model,
    )
    env.initialize()

    trajectory = [start_smiles]
    for _ in range(max_steps):
        all_actions = list(env.get_valid_actions())
        obs = create_graph(all_actions)
        chosen = agent.get_action(obs, epsilon_threshold=0.0)
        action_smiles = all_actions[chosen]
        result = env.step(action_smiles)
        trajectory.append(action_smiles)
        if result.terminated:
            break
    return trajectory


def rollout_ppo(checkpoint, start_smiles, target_seq, off_target_seq, device, max_steps):
    import ppo.ppo_hyperparams as hp
    from ppo.ppo_agent import PPO
    from ppo.mol_graph import GraphDataset

    state_dim = GraphDataset().node_feature_dim
    model = PPO(state_dim=state_dim, action_dim=hp.max_actions, max_action=1.0,
                target_seq=target_seq, off_target_seq=off_target_seq, device=device,
                base_mols=[start_smiles])
    model.load_checkpoint(checkpoint)

    obs, _ = model.env.reset()
    trajectory = [start_smiles]
    for _ in range(max_steps):
        with torch.no_grad():
            logits = model.actor(obs)
            action = torch.argmax(logits).item()
        obs, reward, terminated, truncated, info = model.env.step(action)
        trajectory.append(info["smiles"])
        if terminated or truncated:
            break
    return trajectory, model.env.admet_model, model.env.binding_model, model.env.sa_model


def build_steps(trajectory, target_seq, off_target_seq, device, admet_model, binding_model, sa_model):
    steps = []
    prev_smiles = None
    for i, smiles in enumerate(trajectory):
        canon = Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) if Chem.MolFromSmiles(smiles) else smiles
        score = score_smiles(canon, target_seq, device, off_target_seq, admet_model, binding_model, sa_model)
        highlight = new_atoms_since(prev_smiles, canon)
        steps.append({
            "index": i,
            "smiles": canon,
            "image_b64": mol_image_base64(canon, highlight_atoms=highlight),
            "admet": score["admet"],
            "binding_uM": score["binding_uM"],
            "sa_score": score["sa_score"],
            "selectivity": score["selectivity"],
            "reward": score["reward"],
        })
        prev_smiles = canon
    return steps


def render_html(algo, target_name, checkpoint, steps, output_path):
    def fmt(v, digits=3):
        return "-" if v is None else f"{v:.{digits}f}"

    first, last = steps[0], steps[-1]
    delta_binding = None
    if first["binding_uM"] and last["binding_uM"]:
        delta_binding = first["binding_uM"] - last["binding_uM"]

    cards = []
    for s in steps:
        img_tag = (f'<img src="data:image/png;base64,{s["image_b64"]}" alt="step {s["index"]}">'
                   if s["image_b64"] else '<div class="missing">invalid molecule</div>')
        cards.append(f"""
        <div class="card">
          <div class="step-num">Step {s['index']}</div>
          {img_tag}
          <div class="smiles">{s['smiles']}</div>
          <table class="metrics">
            <tr><td>Reward</td><td>{fmt(s['reward'])}</td></tr>
            <tr><td>ADMET score</td><td>{fmt(s['admet'])}</td></tr>
            <tr><td>Binding (uM)</td><td>{fmt(s['binding_uM'])}</td></tr>
            <tr><td>SA score</td><td>{fmt(s['sa_score'])}</td></tr>
            {f"<tr><td>Selectivity</td><td>{fmt(s['selectivity'])}</td></tr>" if s['selectivity'] is not None else ""}
          </table>
        </div>""")

    summary_delta = (
        f'<p>Binding affinity improved by {delta_binding:.3f} uM '
        f'({first["binding_uM"]:.3f} -&gt; {last["binding_uM"]:.3f}).</p>'
        if delta_binding is not None else ""
    )

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{algo.upper()} agent trajectory - {target_name}</title>
<style>
  body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; background: #0b0d12; color: #e8e8ec; margin: 0; padding: 24px; }}
  h1 {{ font-size: 20px; margin-bottom: 4px; }}
  .meta {{ color: #9aa0ac; font-size: 13px; margin-bottom: 20px; }}
  .summary {{ background: #161a22; border: 1px solid #262b36; border-radius: 10px; padding: 16px 20px; margin-bottom: 24px; }}
  .filmstrip {{ display: flex; flex-wrap: wrap; gap: 16px; }}
  .card {{ background: #161a22; border: 1px solid #262b36; border-radius: 10px; padding: 12px; width: 220px; }}
  .card img {{ width: 100%; border-radius: 6px; background: #fff; }}
  .step-num {{ font-weight: 600; margin-bottom: 6px; }}
  .smiles {{ font-family: monospace; font-size: 10px; color: #9aa0ac; word-break: break-all; margin: 6px 0; height: 28px; overflow: hidden; }}
  .metrics {{ width: 100%; font-size: 12px; border-collapse: collapse; }}
  .metrics td {{ padding: 2px 0; }}
  .metrics td:first-child {{ color: #9aa0ac; }}
  .metrics td:last-child {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .missing {{ height: 220px; display: flex; align-items: center; justify-content: center; color: #9aa0ac; }}
</style>
</head>
<body>
  <h1>{algo.upper()} agent trajectory &mdash; {target_name}</h1>
  <div class="meta">checkpoint: {checkpoint} &middot; {len(steps) - 1} edit steps &middot; generated {time.strftime('%Y-%m-%d %H:%M:%S')}</div>
  <div class="summary">
    <strong>Start:</strong> {first['smiles']}<br>
    <strong>Final:</strong> {last['smiles']}<br>
    {summary_delta}
  </div>
  <div class="filmstrip">
    {''.join(cards)}
  </div>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["dqn", "ppo"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-name", default=DEFAULT_TARGET, choices=list(TARGETS))
    parser.add_argument("--off-target-name", default=None, choices=list(TARGETS))
    parser.add_argument("--start-smiles", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_seq = TARGETS[args.target_name]
    off_target_seq = TARGETS[args.off_target_name] if args.off_target_name else None
    start_smiles = args.start_smiles or sample_pilot_molecules(n=1, seed=args.seed)[0]

    if args.algo == "dqn":
        import dqn.dqn_hyperparams as hyp
        max_steps = args.max_steps or hyp.max_steps
        admet_model = ADMETModel(device)
        binding_model = Plapt(device=str(device))
        sa_model = SyntheticAccessibility()
        trajectory = rollout_dqn(args.checkpoint, start_smiles, target_seq, off_target_seq,
                                  device, max_steps, admet_model, binding_model, sa_model)
    else:
        import ppo.ppo_hyperparams as hp
        max_steps = args.max_steps or hp.steps_per_episode
        trajectory, admet_model, binding_model, sa_model = rollout_ppo(
            args.checkpoint, start_smiles, target_seq, off_target_seq, device, max_steps)

    steps = build_steps(trajectory, target_seq, off_target_seq, device, admet_model, binding_model, sa_model)

    output_path = args.output or f"experiments/results/visualizations/{args.algo}_{args.target_name}_{time.strftime('%Y%m%d-%H%M%S')}.html"
    render_html(args.algo, args.target_name, args.checkpoint, steps, output_path)

    json_path = os.path.splitext(output_path)[0] + ".json"
    with open(json_path, "w") as f:
        json.dump({
            "algo": args.algo,
            "target_name": args.target_name,
            "checkpoint": args.checkpoint,
            "steps": steps,
        }, f, indent=2)

    print(f"Wrote {output_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()