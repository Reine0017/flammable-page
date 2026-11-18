# The flammable page

A page transition: a lit match follows your cursor; touch it to the button and the cover burns away like real paper to reveal the page underneath.

**Play:** https://reine0017.github.io/flammable-page/ · July version: https://reine0017.github.io/flammable-page/v1/

## v1 → v2

- **v1 (July)** grows a sine-wobbled ring from the ignition point and strokes the char as outlines.
- **v2 (November)**, rebuilt with Claude Fable 5.2:
  - **The idea:** the cover is a WebGL shader that thresholds a burn-arrival map over time, so the hole grows fingers, leaves islands and opens spot burns ahead of the front.
  - **Fuel map:** a paper-fibre fuel map baked from Blender nodes (`tools/blender_bake_maps.py`).
  - **Spread:** a small fire-spread solver (`tools/spread.py`, Dijkstra over a 16-neighbour stencil with an upward draught) turns the fuel into per-pixel burn times.
  - **Rim and char:** the ember colours come from Blender's Blackbody node, and the crumpled char texture is rendered in Cycles.
  - **Flames:** a Blender-simulated flame flipbook exists (`tools/blender_render_flames.py`) but is switched off. Real paper smoulders more quietly.

Everything is inlined into one HTML file. WebGL and Canvas 2D, no libraries.
