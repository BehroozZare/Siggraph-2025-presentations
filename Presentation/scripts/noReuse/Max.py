# # if you want to call the toolbox the old way with `blender -b -P demo_XXX.py`, then uncomment these two lines
# import sys, os
# sys.path.append("../../BlenderToolbox/")
import blendertoolbox as bt 
import bpy
import os
import numpy as np
from matplotlib import cm

cwd = os.getcwd()
print(cwd)

def setMat_vertexColorWithWireframe(mesh, edgeThickness, edgeRGBA, baseRGBA=(0.9, 0.9, 0.9, 1.0)):
    """
    Sets a material that combines a uniform base color and wireframe rendering.

    Inputs:
    - mesh: bpy.object of the mesh
    - edgeThickness: Float value for wireframe thickness
    - edgeRGBA: Tuple of 4 floats for wireframe color (R, G, B, A)
    - baseRGBA: Tuple of 4 floats for base color (R, G, B, A)
    """
    # Initialize material node graph
    mat = bpy.data.materials.new('CombinedMaterial')
    mesh.data.materials.append(mat)
    mesh.active_material = mat
    mat.use_nodes = True
    tree = mat.node_tree
    nodes = tree.nodes
    links = tree.links

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Add necessary nodes
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (400, 0)

    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled.location = (0, 0)

    # Base Color RGB Node
    base_rgb = nodes.new(type='ShaderNodeRGB')
    base_rgb.outputs[0].default_value = baseRGBA
    base_rgb.location = (-600, 0)

    # Wireframe Node
    wireframe = nodes.new(type='ShaderNodeWireframe')
    wireframe.inputs['Size'].default_value = edgeThickness
    wireframe.location = (-600, -200)

    # Wireframe Color
    wire_rgb = nodes.new(type='ShaderNodeRGB')
    wire_rgb.outputs[0].default_value = edgeRGBA
    wire_rgb.location = (-800, -200)

    # Mix Shader
    mix_shader = nodes.new(type='ShaderNodeMixShader')
    mix_shader.location = (-200, 0)

    # Emission Shader for Wireframe
    emission = nodes.new(type='ShaderNodeEmission')
    emission.inputs['Color'].default_value = edgeRGBA
    emission.inputs['Strength'].default_value = 5.0
    emission.location = (-400, -200)

    # Links
    links.new(base_rgb.outputs['Color'], principled.inputs['Base Color'])
    links.new(wireframe.outputs['Fac'], mix_shader.inputs['Fac'])
    links.new(principled.outputs['BSDF'], mix_shader.inputs[1])
    links.new(emission.outputs['Emission'], mix_shader.inputs[2])
    links.new(mix_shader.outputs['Shader'], output.inputs['Surface'])






for mesh_id in ['0', '1']:
    outputPath = os.path.join(cwd, 'Max_' + mesh_id + '.png') # make it abs path for windows

    ## initialize blender
    imgRes_x = 1500 
    imgRes_y = 1500
    numSamples = 100 
    exposure = 1.5 
    bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure)


    ## read mesh (choose either readPLY or readOBJ)
    meshPath = os.path.join(cwd, 'meshes', '{}.obj'.format(mesh_id))
    location = (1.12, 0.16, 1.02) # (UI: click mesh > Transform > Location)
    rotation = (100, 0, 270) # (UI: click mesh > Transform > Rotation)
    scale = (0.005,0.005,0.005) # (UI: click mesh > Transform > Scale)
    mesh = bt.readMesh(meshPath, location, rotation, scale)


    ## Set both vertex color and wireframe material
    edgeThickness = 0.001
    edgeRGBA = (0, 0, 0, 1)
    baseRGBA = (0.95, 0.95, 0.95, 1)
    setMat_vertexColorWithWireframe(mesh, edgeThickness, edgeRGBA, baseRGBA)


    ## set invisible plane (shadow catcher)
    bt.invisibleGround(shadowBrightness=0.9)

    ## set camera (recommend to change mesh instead of camera, unless you want to adjust the Elevation)
    camLocation = (3, 0, 2)
    lookAtLocation = (0,0,0.5)
    focalLength = 45 # (UI: click camera > Object Data > Focal Length)
    cam = bt.setCamera(camLocation, lookAtLocation, focalLength)

    ## set light
    lightAngle = (6, -30, -155) 
    strength = 2
    shadowSoftness = 0.3
    sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)

    ## set ambient light
    bt.setLight_ambient(color=(0.1,0.1,0.1,1)) 

    ## set gray shadow to completely white with a threshold 
    bt.shadowThreshold(alphaThreshold = 0.05, interpolationMode = 'CARDINAL')

    ## save blender file so that you can adjust parameters in the UI
    bpy.ops.wm.save_mainfile(filepath=os.path.join(os.getcwd() + '/test.blend'))

    ## save rendering
    bt.renderImage(outputPath, cam)