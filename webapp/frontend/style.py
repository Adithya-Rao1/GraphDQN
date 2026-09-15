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

  /* Streamlit's built-in "running" indicator -- an animated figure icon
     (cycles through several pictograms) shown top-right during script
     execution/rerun. Our polling components (components/progress.py) rerun
     every couple seconds while a job is active, so this would otherwise be
     almost constantly animating. Hidden per user request. */
  [data-testid="stStatusWidget"] {{ display: none !important; }}

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
  .gdqn-mol-img {{
    display: block;
    width: 100%;
    max-width: 280px;
    margin: 0 auto;
    box-sizing: border-box;
    border-radius: 8px;
    border: 1px solid {BORDER};
    background: {CARD_BG};
    padding: 8px;
  }}

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
  [data-testid="stNumberInput"] input {{
    background: {CARD_BG} !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 6px !important;
  }}

  /* Selectbox / multiselect closed-state control -- descendant (not just
     direct-child) selectors since BaseWeb nests the value display a couple
     levels deep and a `>` combinator silently misses it. */
  [data-testid="stSelectbox"] div[data-baseweb="select"],
  [data-testid="stSelectbox"] div[data-baseweb="select"] div,
  [data-testid="stMultiSelect"] div[data-baseweb="select"],
  [data-testid="stMultiSelect"] div[data-baseweb="select"] div {{
    background: {CARD_BG} !important;
    color: {TEXT} !important;
    border-color: {BORDER} !important;
  }}

  /* Multiselect selected-item chips -- BaseWeb gives these their own light
     background regardless of theme, which combined with our forced light
     text color elsewhere made them render as invisible light-on-light. */
  span[data-baseweb="tag"] {{
    background: {BORDER} !important;
    color: {TEXT} !important;
  }}
  span[data-baseweb="tag"] * {{ color: {TEXT} !important; fill: {TEXT} !important; }}

  /* Newer Streamlit builds implement st.multiselect's tags via
     react-aria-components instead of BaseWeb (confirmed by inspecting the
     actual rendered DOM: <span data-tag="" ...>), so the data-baseweb rule
     above never matched them -- they kept their default light chip
     background with our forced light text on top, i.e. invisible. */
  [data-testid="stMultiSelectTagsContainer"] span[data-tag] {{
    background: {BORDER} !important;
    color: {TEXT} !important;
    border-radius: 6px !important;
  }}
  [data-testid="stMultiSelectTagsContainer"] span[data-tag] * {{
    color: {TEXT} !important;
  }}
  [data-testid="stMultiSelectTagsContainer"] span[data-tag] svg path {{
    stroke: {TEXT} !important;
  }}

  /* Dropdown/menu popovers (the open list of options) render in a portal
     attached directly to <body>, outside stAppViewContainer, so they need
     their own unscoped rules or they fall back to BaseWeb's light theme. */
  div[data-baseweb="popover"],
  div[data-baseweb="popover"] *,
  ul[data-baseweb="menu"],
  li[role="option"] {{
    background: {CARD_BG} !important;
    color: {TEXT} !important;
  }}
  li[role="option"]:hover,
  li[aria-selected="true"] {{
    background: {BORDER} !important;
  }}

  /* Tab bars (st.tabs) */
  button[data-baseweb="tab"] {{ color: {MUTED} !important; }}
  button[data-baseweb="tab"][aria-selected="true"] {{ color: {TEXT} !important; }}
  [data-baseweb="tab-highlight"] {{ background: {TEXT} !important; }}
  [data-baseweb="tab-border"] {{ background: {BORDER} !important; }}

  [data-testid="stWidgetLabel"] p,
  [data-testid="stWidgetLabel"] span {{ color: {TEXT} !important; }}

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
