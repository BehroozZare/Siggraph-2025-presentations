# # if you want to call the toolbox the old way with `blender -b -P demo_XXX.py`, then uncomment these two lines
# import sys, os
# sys.path.append("../../BlenderToolbox/")
import blendertoolbox as bt 
import bpy
import os
import numpy as np
from matplotlib import cm
import mathutils
import math
cwd = os.getcwd()

def setCamera(camLocation, lookAtLocation=(0,0,0), focalLength=35, rotation_euler=(0,0,0)):
    # Initialize camera
    bpy.ops.object.camera_add(location=camLocation)
    cam = bpy.context.object
    cam.data.lens = focalLength
    
    # Make camera look at the target
    direction = mathutils.Vector(lookAtLocation) - cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()

    # Apply additional rotation if provided
    cam.rotation_euler.rotate_axis('X', rotation_euler[0])
    cam.rotation_euler.rotate_axis('Y', rotation_euler[1])
    cam.rotation_euler.rotate_axis('Z', rotation_euler[2])
    
    return cam


## Separate the mesh by loose parts
def separate_mesh_by_loose_parts(obj):
    """
    Separates the given mesh object into individual objects based on loose parts.

    Inputs:
    - obj: bpy.object of the mesh to separate

    Returns:
    - List of separated mesh objects
    """
    # Ensure the object is selected and active
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Enter Edit Mode
    bpy.ops.object.mode_set(mode='EDIT')

    # Select all geometry
    bpy.ops.mesh.select_all(action='SELECT')

    # Separate by loose parts
    bpy.ops.mesh.separate(type='LOOSE')

    # Return to Object Mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Collect all new objects
    separated_objects = [o for o in bpy.context.selected_objects]

    return separated_objects

def add_image_texture_material(mesh_obj, image_path,
                            material_name="ImageTextureMaterial"):
    # Create a new material
    mat = bpy.data.materials.new(name=material_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear existing nodes
    nodes.clear()

    # Add UV Map node
    uv_map = nodes.new(type='ShaderNodeUVMap')
    uv_map.location = (-800, 0)
    uv_map.uv_map = "UVMap"  # Ensure the mesh has a UVMap

    # Add Image Texture node
    img_tex = nodes.new(type='ShaderNodeTexImage')
    img_tex.location = (-600, 0)
    img_tex.image = bpy.data.images.load(os.path.abspath(image_path))
    img_tex.interpolation = 'Linear'
    img_tex.extension = 'EXTEND'

    # Add Principled BSDF node
    principled_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled_bsdf.location = (-400, 0)

    # Add Material Output node
    material_output = nodes.new(type='ShaderNodeOutputMaterial')
    material_output.location = (0, 0)

    # Link UV Map to Image Texture
    links.new(uv_map.outputs['UV'], img_tex.inputs['Vector'])

    # Link Image Texture Color to Principled BSDF Base Color
    links.new(img_tex.outputs['Color'], principled_bsdf.inputs['Base Color'])

    # Link Principled BSDF to Material Output
    links.new(principled_bsdf.outputs['BSDF'], material_output.inputs['Surface'])

    # Assign the material to the mesh object
    if mesh_obj.data.materials:
        mesh_obj.data.materials[0] = mat
    else:
        mesh_obj.data.materials.append(mat)

    return mat

def add_color_material(mesh_obj, color=(1.0, 0.0, 0.0, 1.0), material_name="ColorMaterial"):
    # Create a new material
    mat = bpy.data.materials.new(name=material_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear existing nodes
    nodes.clear()

    # Add Principled BSDF node
    principled_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled_bsdf.location = (-200, 0)
    principled_bsdf.inputs['Base Color'].default_value = color

    # Add Material Output node
    material_output = nodes.new(type='ShaderNodeOutputMaterial')
    material_output.location = (0, 0)

    # Link nodes
    links.new(principled_bsdf.outputs['BSDF'], material_output.inputs['Surface'])

    # Assign the material to the mesh object
    if mesh_obj.data.materials:
        mesh_obj.data.materials[0] = mat
    else:
        mesh_obj.data.materials.append(mat)

    return mat

#main code
if __name__ == "__main__":
    file_path = os.path.abspath(__file__)
    obj_dir = os.path.join(os.path.dirname(file_path), 'data')
    result_dir = os.path.join(os.path.dirname(file_path), 'images')
    texture_dir = os.path.join(os.path.dirname(file_path), 'BlenderTexture')
    list_of_obj_files = []
    #List all the files in the obj_dir
    files = os.listdir(obj_dir)
    for file in files:
        if file.endswith('.obj'):
            list_of_obj_files.append(file)

    for cnt in range(len(list_of_obj_files)):
        obj_file = str(cnt) + '.obj'
        mesh_path = os.path.join(obj_dir, obj_file)
        result_path = os.path.join(result_dir, obj_file.replace('.obj', '.png'))
        ## initialize blender
        # imgRes_x = 1500 
        # imgRes_y = 1500
        imgRes_x = 400 
        imgRes_y = 400 
        numSamples = 256 
        exposure = 1.5 
        bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure)
        ## read mesh (choose either readPLY or readOBJ)
        location = (1.55, -0.05, 1.29) # (UI: click mesh > Transform > Location)
        rotation = (70, 0, 90) # (UI: click mesh > Transform > Rotation)
        scale = (1,1,1) # (UI: click mesh > Transform > Scale)
        mesh = bt.readMesh(mesh_path, location, rotation, scale)



        # Load the vertex color from a text with values at each row
        # -------- load vertex colours ---------
        color_data = np.loadtxt(os.path.join(obj_dir, "color.txt"))

        if color_data.ndim == 1 or color_data.shape[1] == 1:
            # ── the file has scalar values ──
            scaler = color_data.astype(np.float32)
            scaler = (scaler - scaler.min()) / (scaler.ptp() + 1e-8)   # normalise 0-1

            cmap = cm.get_cmap("plasma")
            vertex_colors = cmap(scaler)[:, :3]        # Nx3 RGB (drop alpha)
        else:
            # ── the file already has RGB ──
            vertex_colors = color_data[:, :3].astype(np.float32)

            # If values are in 0-255 instead of 0-1, renormalise:
            if vertex_colors.max() > 1.0:
                vertex_colors /= 255.0
        mesh = bt.setMeshColors(mesh, vertex_colors, "vertex")

        meshVColor = bt.colorObj([], 0.5, 1.0, 1.0, 0.0, 0.0)
        bt.setMat_VColor(mesh, meshVColor)
        
        ## set invisible plane (shadow catcher)
        bt.invisibleGround(shadowBrightness=0.9)

        camLocation = (3, 0, 2)
        lookAtLocation = (0,0,0.5)
        focalLength = 45 # (UI: click camera > Object Data > Focal Length)
        cam = bt.setCamera(camLocation, lookAtLocation, focalLength)

        ## set light
        lightAngle = (6, -30, -155) 
        strength = 4
        shadowSoftness = 0.3
        sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)

        ## set ambient light
        bt.setLight_ambient(color=(0.1,0.1,0.1,1)) 

        # set gray shadow to completely white with a threshold 
        bt.shadowThreshold(alphaThreshold = 0.05, interpolationMode = 'CARDINAL')

        ## save blender file so that you can adjust parameters in the UI
        
        # bpy.ops.wm.save_mainfile(filepath=os.path.join(os.path.dirname(file_path), 'mat_on_board_' + obj_file.replace('.obj', '') + '.blend'))
        
        ## save rendering
        bt.renderImage(result_path, cam)