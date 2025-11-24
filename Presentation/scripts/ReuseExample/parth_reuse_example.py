# If you want to call the toolbox the old way with:
# blender -b -P demo_XXX.py
# then uncomment these two lines:
# import sys, os
# sys.path.append("../../BlenderToolbox/")

import blendertoolbox as bt
import bpy
import os
import numpy as np
import mathutils
import math

# -------------------
# Material: image texture via UVs
# -------------------

def add_image_texture_material(mesh_obj, image_path, material_name="ImageTextureMaterial"):
    """Create/assign a material that uses an image texture with the mesh UVs."""
    # Load image (reuses existing if already loaded)
    img = bpy.data.images.load(image_path, check_existing=True)

    # Make material
    mat = bpy.data.materials.new(material_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Nodes
    out = nodes.new("ShaderNodeOutputMaterial"); out.location = (400, 0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = (100, 0)

    tex = nodes.new("ShaderNodeTexImage"); tex.location = (-200, 0)
    tex.image = img
    # Color space default "sRGB" is fine for base color

    # Use active UV map from the mesh
    uv = nodes.new("ShaderNodeUVMap"); uv.location = (-400, 0)
    # Ensure an active UV map exists; OBJ importer should create one
    if mesh_obj.data.uv_layers:
        uv.uv_map = mesh_obj.data.uv_layers.active.name

    # Links
    links.new(uv.outputs["UV"], tex.inputs["Vector"])
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    # Assign
    if mesh_obj.data.materials:
        mesh_obj.data.materials[0] = mat
    else:
        mesh_obj.data.materials.append(mat)

    # Opaque shading (no alpha cutout)
    mat.blend_method = 'OPAQUE'
    return mat

# -------------------
# Main rendering loop
# -------------------

if __name__ == "__main__":
    file_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(file_path)

    obj_dir     = os.path.join(base_dir, 'aggressive')
    result_dir  = os.path.join(base_dir, 'images')
    texture_dir = os.path.join(base_dir, 'BlenderTexture')

    # Texture file to use
    texture_file = 'SoftCyan_Contact.png'
    texture_path = os.path.join(texture_dir, texture_file)
    if not os.path.isfile(texture_path):
        raise FileNotFoundError(f"Texture not found: {texture_path}")

    # pick which OBJ(s) to render
    for cnt in range(60, 200):
        obj_file    = f"frame_{cnt}.obj"
        mesh_path   = os.path.join(obj_dir, obj_file)
        result_path = os.path.join(result_dir, obj_file.replace('.obj', '.png'))

        if not os.path.isfile(mesh_path):
            print(f"Skipping missing OBJ: {mesh_path}")
            continue

        # --- Blender scene init ---
        imgRes_x = 1500
        imgRes_y = 1500
        numSamples = 128
        exposure = 1.5
        bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure)

        # --- Load mesh ---
        location = (1.5,-0.5, 1.11)
        rotation = (63, 0, 90)  # degrees; BlenderToolbox handles conversion
        scale = (1, 1, 1)
        mesh = bt.readMesh(mesh_path, location, rotation, scale)

        # --- Apply image texture material (uses OBJ UVs) ---
        add_image_texture_material(mesh, texture_path)

        # --- Invisible plane for shadows ---
        bt.invisibleGround(shadowBrightness=0.9)

        # --- Camera ---
        camLocation = (3, 0, 2)
        lookAtLocation = (0, 0, 0.5)
        focalLength = 45
        cam = bt.setCamera(camLocation, lookAtLocation, focalLength)

        # --- Lights ---
        lightAngle = (6, -30, -155)
        strength = 4
        shadowSoftness = 0.3
        sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)
        bt.setLight_ambient(color=(0.1, 0.1, 0.1, 1))

        # --- Shadow threshold ---
        bt.shadowThreshold(alphaThreshold=0.05, interpolationMode='CARDINAL')

        # --- Save blend file ---
        # bpy.ops.wm.save_mainfile(filepath=os.path.join(
        #     base_dir, 'reuse_render_' + obj_file.replace('.obj', '') + '.blend'))

        # --- Render image ---
        bt.renderImage(result_path, cam)
