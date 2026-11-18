"""Bake the burn-arrival map and the char ramp in Blender.

    blender -b --factory-startup --python tools/blender_bake_maps.py

What Blender does here:
  * the FUEL map (how fast each bit of paper burns) is a node material,
    baked to an image with Cycles;
  * the RIM RAMP (scorch -> glowing rim -> char) is three node materials,
    with ember colours from Blender's Blackbody node, baked the same way;
  * the CHAR TEXTURE (black crumpled burnt paper) is a lit material,
    rendered with Cycles.
What Blender does not do: the fire spread itself. That is tools/spread.py,
a plain solver that runs here in Blender's Python on the baked fuel map.

Saves blender/burn-maps.blend so the node trees can be opened and tweaked.
"""
import json
import os
import struct
import sys
import time
import zlib

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "assets")
sys.path.insert(0, HERE)
import spread  # noqa: E402

N = 512
RAMP_W, RAMP_ROWS = 256, 16
D_MIN, D_MAX = -24.0, 40.0  # px range the ramp covers; the front is d = 0


# ---------------------------------------------------------------- helpers
def write_png(path, rgb):
    """Minimal exact PNG writer (no colour management in the way)."""
    h, w, _ = rgb.shape
    raw = rgb.astype(np.int16)
    sub = raw.copy()
    sub[:, 1:] -= raw[:, :-1]
    up = raw.copy()
    up[1:] -= raw[:-1]
    rows = []
    for y in range(h):
        cands = [(0, raw[y]), (1, sub[y]), (2, up[y])]
        ft, data = min(cands, key=lambda c: int(np.abs(((c[1] + 128) % 256) - 128).sum()))
        rows.append(bytes([ft]) + (data % 256).astype(np.uint8).tobytes())

    def chunk(tag, body):
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(b"".join(rows), 9)))
        f.write(chunk(b"IEND", b""))


def linear_to_srgb(a):
    a = np.clip(a, 0, None)
    return np.where(a <= 0.0031308, a * 12.92, 1.055 * np.power(a, 1 / 2.4) - 0.055)


class Tree:
    """Tiny wrapper so the node graphs below read top to bottom."""

    def __init__(self, name):
        self.mat = bpy.data.materials.new(name)
        self.mat.use_nodes = True
        self.nt = self.mat.node_tree
        self.nt.nodes.clear()
        self.col = 0

    def node(self, kind, label=None, **props):
        n = self.nt.nodes.new(kind)
        n.location = (self.col * 190, -len(self.nt.nodes) % 5 * 150)
        self.col += 1
        if label:
            n.label = label
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def link(self, a, b):
        self.nt.links.new(a, b)

    def math(self, op, a, b=None, label=None, clamp=False):
        n = self.node("ShaderNodeMath", label, operation=op, use_clamp=clamp)
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = v
            else:
                self.link(v, n.inputs[i])
        return n.outputs[0]

    def sstep(self, value, lo, hi, label=None):
        """smoothstep(lo, hi, value); lo/hi may be numbers or sockets."""
        n = self.node("ShaderNodeMapRange", label, interpolation_type="SMOOTHSTEP")
        self.link(value, n.inputs["Value"])
        for name, v in (("From Min", lo), ("From Max", hi)):
            if isinstance(v, (int, float)):
                n.inputs[name].default_value = v
            else:
                self.link(v, n.inputs[name])
        return n.outputs[0]

    def emit(self, color=None, strength=None):
        e = self.node("ShaderNodeEmission")
        if color is not None:
            self.link(color, e.inputs["Color"])
        if strength is not None:
            self.link(strength, e.inputs["Strength"])
        out = self.node("ShaderNodeOutputMaterial")
        self.link(e.outputs[0], out.inputs["Surface"])


def bake(plane, tree, w, h):
    """Bake a material's emission to a float image; returns (h, w, 3), row 0 = top."""
    img = bpy.data.images.new(tree.mat.name + "_bake", w, h, alpha=False, float_buffer=True, is_data=True)
    tex = tree.nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tree.nt.nodes.active = tex
    tex.select = True
    plane.data.materials.clear()
    plane.data.materials.append(tree.mat)
    bpy.context.view_layer.objects.active = plane
    plane.select_set(True)
    bpy.ops.object.bake(type="EMIT", margin=0, use_clear=True)
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    return px.reshape(h, w, 4)[::-1, :, :3].astype(np.float64)


def uv_xy(t):
    uv = t.node("ShaderNodeTexCoord")
    sep = t.node("ShaderNodeSeparateXYZ")
    t.link(uv.outputs["UV"], sep.inputs[0])
    return uv.outputs["UV"], sep.outputs[0], sep.outputs[1]


def noise(t, uv, scale, detail, label, stretch=None):
    vec = uv
    if stretch:
        m = t.node("ShaderNodeMapping", label + " stretch")
        m.inputs["Scale"].default_value = stretch
        t.link(uv, m.inputs["Vector"])
        vec = m.outputs[0]
    n = t.node("ShaderNodeTexNoise", label)
    n.inputs["Scale"].default_value = scale
    n.inputs["Detail"].default_value = detail
    t.link(vec, n.inputs["Vector"])
    return n.outputs[0]


def ramp_node(t, fac, stops, label):
    """stops: [(d_px, (r, g, b))] -> a ColorRamp positioned along the d axis."""
    r = t.node("ShaderNodeValToRGB", label)
    t.link(fac, r.inputs[0])
    els = r.color_ramp.elements
    while len(els) < len(stops):
        els.new(0.5)
    for el, (d, c) in zip(els, stops):
        el.position = (d - D_MIN) / (D_MAX - D_MIN)
        el.color = (*c, 1.0)
    return r.outputs[0]


# ---------------------------------------------------------------- materials
def fuel_material():
    """Burn speed across the sheet: ~0.3 (damp) to ~1.6 (dry)."""
    t = Tree("Fuel")
    uv, _, _ = uv_xy(t)
    big = noise(t, uv, 3.4, 2.0, "big lobes")
    mid = noise(t, uv, 9.5, 3.0, "fingers")
    fibre = noise(t, uv, 46.0, 2.0, "paper fibres", stretch=(1.0, 0.22, 1.0))
    mix = t.math("ADD", t.math("MULTIPLY", big, 0.55), t.math("MULTIPLY", mid, 0.30))
    mix = t.math("ADD", mix, t.math("MULTIPLY", fibre, 0.15))
    rng = t.node("ShaderNodeMapRange", "spread to 0..1")
    rng.inputs["From Min"].default_value = 0.30
    rng.inputs["From Max"].default_value = 0.70
    t.link(mix, rng.inputs["Value"])
    speed = t.math("ADD", t.math("MULTIPLY", t.math("POWER", rng.outputs[0], 1.5), 1.25), 0.38,
                   label="dry = fast")
    # damp patches: islands of paper that hold out and burn last
    damp = noise(t, uv, 5.5, 1.0, "damp patches")
    patch = t.node("ShaderNodeMapRange", "patch mask")
    patch.inputs["From Min"].default_value = 0.60
    patch.inputs["From Max"].default_value = 0.66
    t.link(damp, patch.inputs["Value"])
    speed = t.math("MULTIPLY", speed, t.math("SUBTRACT", 1.0, t.math("MULTIPLY", patch.outputs[0], 0.58)),
                   label="damp = slow")
    t.emit(strength=speed)
    return t


def ramp_axes(t):
    """x -> d in px, y -> crackle v (0 calm, 1 cracked and hot)."""
    _, x, v = uv_xy(t)
    d = t.math("ADD", t.math("MULTIPLY", x, D_MAX - D_MIN), D_MIN, label="d (px)")
    return x, v, d


def tint_material():
    """The paper side. A smoulder barely scorches: the sheet stays clean
    until a few pixels from the rim, then goes tan, brown, black."""
    t = Tree("Scorch")
    x, _, _ = ramp_axes(t)
    col = ramp_node(t, x, [
        (-24, (1, 1, 1)), (-11, (0.96, 0.90, 0.78)), (-6, (0.66, 0.47, 0.26)),
        (-2.8, (0.20, 0.10, 0.04)), (-1.0, (0.03, 0.015, 0.008)), (0.5, (0.008, 0.006, 0.005)),
        (40, (0.008, 0.006, 0.005))], "paper to scorch")
    t.emit(color=col)
    return t


def emission_material():
    """The glowing rim. v sets how thick and hot it is: a hairline of dull
    red where v = 0, a fat orange bead where v = 1. Colour is a Blackbody
    node and brightness follows T^4, so cooler really is dimmer and redder."""
    t = Tree("EmberRim")
    _, v, d = ramp_axes(t)
    width = t.math("ADD", t.math("MULTIPLY", t.math("POWER", v, 1.6), 8.0), 1.7, label="rim width (px)")
    lead = t.sstep(d, -1.2, -0.2, "rim starts")
    trail = t.math("SUBTRACT", 1.0,
                   t.sstep(d, t.math("SUBTRACT", width, 1.0), t.math("ADD", width, 0.8), "rim ends"))
    rim = t.math("MULTIPLY", lead, trail, label="rim")
    past = t.math("MAXIMUM", t.math("SUBTRACT", d, width), 0.0, label="px behind the rim")
    after = t.math("MULTIPLY", t.math("EXPONENT", t.math("DIVIDE", past, -6.0)),
                   t.math("GREATER_THAN", d, 0.0), label="afterglow on the char")
    kelvin = t.math("ADD", t.math("MULTIPLY", v, 350.0), 1450.0, label="kelvin at the rim")
    kelvin = t.math("MAXIMUM", t.math("SUBTRACT", kelvin, t.math("MULTIPLY", past, 45.0)), 900.0)
    bb = t.node("ShaderNodeBlackbody")
    t.link(kelvin, bb.inputs[0])
    radiance = t.math("POWER", t.math("DIVIDE", kelvin, 1800.0), 4.0, label="T^4 falloff")
    mask = t.math("MAXIMUM", rim, t.math("MULTIPLY", after, 0.35))
    strength = t.math("MULTIPLY", t.math("MULTIPLY", radiance, mask), 0.55, label="intensity")
    t.emit(color=bb.outputs[0], strength=strength)
    return t


def alpha_material():
    """Where paper turns to char: 0 on the paper side, 1 just past the rim."""
    t = Tree("CharMix")
    _, _, d = ramp_axes(t)
    t.emit(strength=t.sstep(d, -1.0, 0.8, "paper to char"))
    return t


def char_material():
    """Burnt paper: black and crumpled, with a faint grey sheen where a fold
    catches the light. Lit and rendered, not baked, so the folds are real."""
    t = Tree("Char")
    uv, _, _ = uv_xy(t)
    # bend the coordinates so the folds wander instead of forming tidy cells
    warp = t.node("ShaderNodeTexNoise", "warp")
    warp.inputs["Scale"].default_value = 2.6
    warp.inputs["Detail"].default_value = 2.0
    t.link(uv, warp.inputs["Vector"])
    bent = t.node("ShaderNodeVectorMath", "bend", operation="MULTIPLY_ADD")
    t.link(warp.outputs[1], bent.inputs[0])
    bent.inputs[1].default_value = (0.42, 0.42, 0.0)
    t.link(uv, bent.inputs[2])
    bent = bent.outputs[0]
    folds = t.node("ShaderNodeTexVoronoi", "big folds", feature="SMOOTH_F1")
    folds.inputs["Scale"].default_value = 5.0
    folds.inputs["Smoothness"].default_value = 0.7
    t.link(bent, folds.inputs["Vector"])
    crumple = t.node("ShaderNodeTexNoise", "crumple")
    crumple.inputs["Scale"].default_value = 7.0
    crumple.inputs["Detail"].default_value = 5.0
    crumple.inputs["Roughness"].default_value = 0.6
    crumple.inputs["Distortion"].default_value = 1.6
    t.link(bent, crumple.inputs["Vector"])
    grit = noise(t, uv, 120.0, 2.0, "ash grit")
    height = t.math("ADD", t.math("MULTIPLY", folds.outputs["Distance"], 1.0),
                    t.math("MULTIPLY", crumple.outputs[0], 0.9))
    height = t.math("ADD", height, t.math("MULTIPLY", grit, 0.04), label="height")
    bump = t.node("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.85
    bump.inputs["Distance"].default_value = 0.09
    t.link(height, bump.inputs["Height"])
    bsdf = t.node("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.006, 0.006, 0.006, 1)
    t.link(t.math("ADD", t.math("MULTIPLY", grit, 0.2), 0.42, label="roughness"), bsdf.inputs["Roughness"])
    t.link(bump.outputs[0], bsdf.inputs["Normal"])
    out = t.node("ShaderNodeOutputMaterial")
    t.link(bsdf.outputs[0], out.inputs["Surface"])
    return t


def render_char(plane, tree, size):
    """Light the char from a low angle and render it straight down."""
    scene = bpy.context.scene
    plane.data.materials.clear()
    plane.data.materials.append(tree.mat)
    cam_data = bpy.data.cameras.new("CharCam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = 2.0
    cam = bpy.data.objects.new("CharCam", cam_data)
    cam.location = (0, 0, 5)
    scene.collection.objects.link(cam)
    scene.camera = cam
    for name, rot, power in (("Key", (1.25, 0, -0.9), 3.2), ("Fill", (1.05, 0, 2.3), 0.5)):
        sun = bpy.data.lights.new(name, "SUN")
        sun.energy = power
        sun.angle = 0.25
        ob = bpy.data.objects.new(name, sun)
        ob.rotation_euler = rot
        scene.collection.objects.link(ob)
    world = bpy.data.worlds.new("Dim")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.006, 0.006, 0.007, 1)
    scene.world = world
    scene.cycles.samples = 96
    scene.cycles.use_denoising = True
    scene.render.resolution_x = scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Standard"
    ims = scene.render.image_settings
    if hasattr(ims, "media_type"):
        ims.media_type = "IMAGE"
    ims.file_format = "JPEG"
    ims.color_mode = "RGB"
    ims.quality = 86
    scene.render.filepath = os.path.join(OUT, "char-texture.jpg")
    bpy.ops.render.render(write_still=True)


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 16
    scene.cycles.use_denoising = False

    bpy.ops.mesh.primitive_plane_add(size=2)  # comes with a 0..1 UV map
    plane = bpy.context.object
    plane.name = "Sheet"

    fuel = fuel_material()
    tint, emis, alph = tint_material(), emission_material(), alpha_material()
    char = char_material()

    # --- arrival map
    fuel_px = bake(plane, fuel, N, N)[:, :, 0]
    print("fuel range %.2f..%.2f" % (fuel_px.min(), fuel_px.max()))
    T, grad = spread.solve(fuel_px)
    rgb, meta = spread.pack(T, grad)
    write_png(os.path.join(OUT, "burn-arrival.png"), rgb)
    meta["source"] = "Blender %s: fuel baked from node material, spread solved by tools/spread.py" % bpy.app.version_string
    with open(os.path.join(OUT, "burn-meta.json"), "w") as f:
        json.dump(meta, f)
    bands = (((T * 24) % 1 < 0.5) * 60 + T * 195).astype(np.uint8)
    write_png(os.path.join(OUT, "preview-arrival-bands.png"), np.dstack([bands] * 3))
    fp = np.clip(fuel_px / 1.7, 0, 1)
    write_png(os.path.join(OUT, "preview-fuel.png"), np.dstack([(fp * 255).astype(np.uint8)] * 3))

    # --- rim ramp: three strips stacked (scorch tint, emission, paper-to-char)
    strips = []
    for tree in (tint, emis, alph):
        px = bake(plane, tree, RAMP_W, RAMP_ROWS)
        # baked rows run v = 1 at the top; the page wants v = 0 at the top
        strips.append(linear_to_srgb(px[::-1]) if tree is not alph else px[::-1])
    ramp = np.clip(np.vstack(strips), 0, 1)
    write_png(os.path.join(OUT, "char-ramp.png"), (ramp * 255 + 0.5).astype(np.uint8))
    big = np.repeat(np.repeat((ramp * 255 + 0.5).astype(np.uint8), 3, axis=0), 3, axis=1)
    write_png(os.path.join(OUT, "preview-char-ramp.png"), big)

    render_char(plane, char, 512)

    plane.data.materials.clear()
    for tree in (char, fuel, tint, emis, alph):
        plane.data.materials.append(tree.mat)
        tree.mat.use_fake_user = True
    os.makedirs(os.path.join(ROOT, "blender"), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(ROOT, "blender", "burn-maps.blend"))
    print("BAKE OK in %.1fs" % (time.time() - t0), meta)


main()
