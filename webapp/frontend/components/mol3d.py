import streamlit.components.v1 as components

from style import BORDER, CARD_BG, TEXT


def render_3d_viewer(molblock: str, height: int = 340, key: str = "viewer3d") -> None:
    if not molblock:
        components.html(
            f'<div style="height:{height}px; display:flex; align-items:center; justify-content:center; '
            f'color:{TEXT}; opacity:0.5; font-family:-apple-system,Helvetica,Arial,sans-serif; '
            f'border:1px solid {BORDER}; border-radius:8px; background:{CARD_BG};">'
            f"3D structure unavailable</div>",
            height=height,
        )
        return

    escaped = molblock.replace("\\", "\\\\").replace("`", "\\`")
    html = f"""
    <div id="{key}" style="height:{height}px; width:100%; box-sizing:border-box;
         border-radius:8px; border:1px solid {BORDER}; background:{CARD_BG};"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.1.0/3Dmol-min.js"></script>
    <script>
      const viewer = $3Dmol.createViewer("{key}", {{backgroundColor: "{CARD_BG}"}});
      viewer.addModel(`{escaped}`, "mol");
      viewer.setStyle({{}}, {{
        stick: {{radius: 0.15, colorscheme: "Jmol"}},
        sphere: {{scale: 0.25, colorscheme: "Jmol"}}
      }});
      viewer.zoomTo();
      viewer.render();
    </script>
    """
    components.html(html, height=height + 4)
