import streamlit as st

BG = "#0b0d12"
CARD_BG = "#161a22"
BORDER = "#262b36"
TEXT = "#e8e8ec"
MUTED = "#9aa0ac"

_CSS = f"""
<style>
  html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
    background: {BG} !important;
    color: {TEXT};
    font-family: -apple-system, Helvetica, Arial, sans-serif;
  }}
  [data-testid="stSidebar"] {{
    background: {CARD_BG} !important;
    border-right: 1px solid {BORDER};
  }}
  [data-testid="stHeader"] {{ background: transparent !important; }}

  h1, h2, h3, h4, h5, h6 {{ color: {TEXT}; font-weight: 600; }}
  p, label, span, div {{ color: {TEXT}; }}

  .gdqn-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 16px;
  }}
  .gdqn-meta {{ color: {MUTED}; font-size: 13px; }}
  .gdqn-smiles {{
    font-family: monospace;
    font-size: 12px;
    color: {MUTED};
    word-break: break-all;
  }}
  .gdqn-metrics {{ width: 100%; font-size: 13px; border-collapse: collapse; }}
  .gdqn-metrics td {{ padding: 3px 0; }}
  .gdqn-metrics td:first-child {{ color: {MUTED}; }}
  .gdqn-metrics td:last-child {{
    text-align: right;
    font-variant-numeric: tabular-nums;
    color: {TEXT};
  }}
  .gdqn-mol-img {{ width: 100%; border-radius: 6px; background: #fff; }}

  [data-testid="stButton"] > button {{
    background: {CARD_BG};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 6px;
  }}
  [data-testid="stButton"] > button:hover {{
    border-color: {TEXT};
    color: {TEXT};
  }}

  [data-testid="stTextInput"] input,
  [data-testid="stTextArea"] textarea,
  [data-testid="stNumberInput"] input,
  [data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
    background: {CARD_BG} !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 6px !important;
  }}

  [data-testid="stMetric"] {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 12px 16px;
  }}

  hr {{ border-color: {BORDER}; }}
</style>
"""


def inject_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
