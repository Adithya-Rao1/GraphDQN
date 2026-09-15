from typing import Optional

import streamlit as st


def _fmt(value, digits=3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def render_candidate_card(candidate: dict, api, allow_scoring: bool = True, allow_select: bool = False) -> Optional[bool]:
    """Renders one candidate as a card matching visualize_agent.py's styling.

    Returns True if this candidate's fine-tune "include" checkbox is checked
    (only when allow_select=True), else None.
    """
    img_tag = (
        f'<img class="gdqn-mol-img" src="data:image/png;base64,{candidate["image_b64"]}">'
        if candidate.get("image_b64") else '<div class="gdqn-meta">invalid molecule</div>'
    )
    selectivity_row = (
        f'<tr><td>Selectivity</td><td>{_fmt(candidate.get("selectivity"))}</td></tr>'
        if candidate.get("selectivity") is not None else ""
    )
    rating_row = (
        f'<tr><td>Your rating</td><td>{_fmt(candidate.get("user_rating"), 0)}</td></tr>'
        if candidate.get("user_rating") is not None else ""
    )

    protein_line = candidate.get("target_protein_name", "")
    if candidate.get("off_target_protein_name"):
        protein_line += f" (vs. off-target {candidate['off_target_protein_name']})"

    finetune_uses = candidate.get("used_in_finetune_runs") or []
    finetune_line = ""
    if finetune_uses:
        run_ids = ", ".join(f"#{u['training_run_id']}" for u in finetune_uses)
        finetune_line = f'<div class="gdqn-meta">Used in fine-tune run(s): {run_ids}</div>'

    st.markdown(f"""
    <div class="gdqn-card">
      {img_tag}
      <div class="gdqn-smiles">{candidate['smiles']}</div>
      <div class="gdqn-meta">
        From {candidate.get('starting_smiles', '?')} &middot; target: {protein_line}<br>
        Config: {candidate.get('config_name', '?')} &middot; run #{candidate.get('training_run_id', '?')}
      </div>
      {finetune_line}
      <table class="gdqn-metrics">
        <tr><td>Reward</td><td>{_fmt(candidate['reward'])}</td></tr>
        <tr><td>ADMET score</td><td>{_fmt(candidate.get('admet_score'))}</td></tr>
        <tr><td>Binding (uM)</td><td>{_fmt(candidate.get('binding_uM'))}</td></tr>
        <tr><td>SA score</td><td>{_fmt(candidate.get('sa_score'))}</td></tr>
        {selectivity_row}
        {rating_row}
      </table>
    </div>
    """, unsafe_allow_html=True)

    selected = None
    cols = st.columns([3, 1]) if allow_select else [st.container()]

    with cols[0]:
        if allow_scoring:
            default_rating = int(candidate["user_rating"]) if candidate.get("user_rating") is not None else 50
            rating = st.slider("Score", 0, 100, default_rating, key=f"rating_{candidate['id']}")
            if st.button("Save score", key=f"save_{candidate['id']}"):
                api.score_candidate(candidate["id"], float(rating))
                st.success("Saved")
                st.rerun()

    if allow_select:
        with cols[1]:
            selected = st.checkbox("Use for fine-tune", key=f"select_{candidate['id']}",
                                    disabled=candidate.get("user_rating") is None)

    return selected
