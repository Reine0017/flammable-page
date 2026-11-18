"""Simulate a small flame in Blender (Mantaflow) and render it as frames
for a flipbook.

    blender -b --factory-startup --python tools/blender_render_flames.py
    python3 tools/pack_flames.py        # frames -> assets/flames.webp

Set FLAME_QUICK=1 for a fast low-quality pass that only checks the pipeline.

The flame is rendered on pure black, looking straight at it, with the
burning strip near the bottom of the frame. The page draws these frames
along the burn front.
"""
import json
import os
import random
import shutil
import time

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
WORK = os.path.join(ROOT, "blender")
CACHE = os.path.join(WORK, "cache_flames")
FRAMES_DIR = os.path.join(WORK, "flame_frames")

QUICK = os.environ.get("FLAME_QUICK") == "1"
RES = 40 if QUICK else 112          # voxels along the tall axis
WARMUP = 10 if QUICK else 26        # frames to let the fire establish
FRAMES = 4 if QUICK else 64
FPS = 24
CELL = (96, 144)                    # final flipbook cell, px
SUPER = 2                           # render at 2x, the packer scales down
DOM = (1.6, 1.0, 2.4)               # domain size: x, y (depth), z (up)
BASE_Z = 0.22                       # height of the burning strip


def main():
    t0 = time.time()
    for d in (CACHE, FRAMES_DIR):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.frame_start = 1
    scene.frame_end = WARMUP + FRAMES

    # ---- domain: an open box of air
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, DOM[2] / 2))
    dom = bpy.context.object
    dom.name = "FireDomain"
    dom.scale = DOM
    bpy.ops.object.transform_apply(scale=True)
    ds = dom.modifiers.new("Fluid", "FLUID")
    ds.fluid_type = "DOMAIN"
    ds = ds.domain_settings
    ds.domain_type = "GAS"
    ds.resolution_max = RES
    ds.use_adaptive_domain = False
    ds.vorticity = 0.35          # curl in the air: makes the tongues wander
    ds.flame_vorticity = 1.1     # extra curl inside the flame
    ds.burning_rate = 0.45       # lower = fuel burns slower = taller tongues
    ds.flame_smoke = 0.0         # no smoke; the page has its own
    ds.beta = 1.6                # heat buoyancy: how hard hot gas rises
    ds.cache_type = "ALL"
    ds.cache_directory = CACHE
    ds.cache_frame_start = 1
    ds.cache_frame_end = WARMUP + FRAMES

    # ---- flow: a thin burning strip, like a length of paper edge
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, BASE_Z))
    flow = bpy.context.object
    flow.name = "BurningEdge"
    flow.scale = (0.46, 0.10, 0.05)
    bpy.ops.object.transform_apply(scale=True)
    fs = flow.modifiers.new("Fluid", "FLUID")
    fs.fluid_type = "FLOW"
    fs = fs.flow_settings
    fs.flow_type = "FIRE"
    fs.flow_behavior = "INFLOW"
    fs.fuel_amount = 1.0
    flow.hide_render = True      # only the fire shows, not the strip itself
    # the fuel supply gutters and the edge shifts, so the flame never settles
    rng = random.Random(7)
    for f in range(1, WARMUP + FRAMES + 1, 3):
        fs.fuel_amount = rng.uniform(0.45, 1.5)
        fs.keyframe_insert("fuel_amount", frame=f)
        flow.location = (rng.uniform(-0.07, 0.07), 0, BASE_Z)
        flow.keyframe_insert("location", frame=f)

    # ---- material: glow straight from the simulated flame field
    mat = bpy.data.materials.new("Fire")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "flame"
    kelvin = nt.nodes.new("ShaderNodeMapRange")
    kelvin.label = "flame -> kelvin"
    kelvin.inputs["To Min"].default_value = 1000.0
    kelvin.inputs["To Max"].default_value = 2500.0
    bb = nt.nodes.new("ShaderNodeBlackbody")
    power = nt.nodes.new("ShaderNodeMath")
    power.operation = "POWER"
    power.inputs[1].default_value = 1.4
    gain = nt.nodes.new("ShaderNodeMath")
    gain.operation = "MULTIPLY"
    gain.label = "brightness"
    gain.inputs[1].default_value = 15.0
    emit = nt.nodes.new("ShaderNodeEmission")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(attr.outputs["Fac"], kelvin.inputs["Value"])
    nt.links.new(kelvin.outputs[0], bb.inputs[0])
    nt.links.new(attr.outputs["Fac"], power.inputs[0])
    nt.links.new(power.outputs[0], gain.inputs[0])
    nt.links.new(bb.outputs[0], emit.inputs["Color"])
    nt.links.new(gain.outputs[0], emit.inputs["Strength"])
    nt.links.new(emit.outputs[0], out.inputs["Volume"])
    for i, n in enumerate((attr, kelvin, bb, power, gain, emit, out)):
        n.location = (i * 200, 0)
    dom.data.materials.append(mat)

    # ---- camera: flat-on, framing the domain exactly
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.type = "ORTHO"
    cam_data.sensor_fit = "VERTICAL"
    cam_data.ortho_scale = DOM[2]
    cam = bpy.data.objects.new("Cam", cam_data)
    cam.location = (0, -6, DOM[2] / 2)
    cam.rotation_euler = (1.5707963, 0, 0)
    scene.collection.objects.link(cam)
    scene.camera = cam

    world = bpy.data.worlds.new("Black")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)
    scene.world = world

    # ---- render settings
    scene.render.engine = "CYCLES"
    # CPU by default: the frames are tiny, and the first GPU render on a Mac
    # stalls for minutes while Metal compiles Cycles' kernels.
    scene.cycles.device = "CPU"
    if os.environ.get("FLAME_GPU") == "1":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type == "METAL"
        scene.cycles.device = "GPU"
    scene.cycles.samples = 12 if QUICK else 64
    scene.cycles.use_denoising = True
    scene.render.resolution_x = CELL[0] * SUPER
    scene.render.resolution_y = CELL[1] * SUPER
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    # Standard, not AgX: let hot cores clip from orange through yellow to
    # white, the way fire overexposes on camera, and keep the edges saturated.
    scene.view_settings.view_transform = "Standard"
    ims = scene.render.image_settings
    if hasattr(ims, "media_type"):
        ims.media_type = "IMAGE"
    ims.file_format = "PNG"
    ims.color_mode = "RGB"

    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(WORK, "flames.blend"))

    # ---- simulate
    bpy.ops.object.select_all(action="DESELECT")
    dom.select_set(True)
    bpy.context.view_layer.objects.active = dom
    print("baking fluid, res", RES, "frames", WARMUP + FRAMES)
    tb = time.time()
    bpy.ops.fluid.bake_all()
    print("fluid bake %.1fs, cache files: %d" % (
        time.time() - tb, sum(len(fs_) for _, _, fs_ in os.walk(CACHE))))

    # ---- render the flipbook frames
    tr = time.time()
    for i in range(FRAMES):
        scene.frame_set(WARMUP + 1 + i)
        scene.render.filepath = os.path.join(FRAMES_DIR, "flame_%03d.png" % i)
        bpy.ops.render.render(write_still=True)
    print("render %.1fs" % (time.time() - tr))

    with open(os.path.join(FRAMES_DIR, "frames.json"), "w") as f:
        json.dump({"frames": FRAMES, "fps": FPS, "cw": CELL[0], "ch": CELL[1],
                   "base": round(1 - BASE_Z / DOM[2], 4),
                   "source": "Blender %s Mantaflow fire, Cycles" % bpy.app.version_string}, f)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(WORK, "flames.blend"))
    print("FLAMES OK in %.1fs" % (time.time() - t0))


main()
