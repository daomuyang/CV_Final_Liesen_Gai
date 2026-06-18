import bpy


VIEW_SHADING = "MATERIAL"
MATERIAL_ROUGHNESS = 0.78
MATERIAL_METALLIC = 0.0
MATERIAL_SPECULAR = 0.25

RENDER_SCALE = 100
FILTER_SIZE = 1.0 
RENDER_ENGINE = 'BLENDER_EEVEE'


def set_principled_value(node, names, value):
    for name in names:
        if name in node.inputs:
            node.inputs[name].default_value = value
            return


def bind_vertex_color_material(mesh_object):
    if mesh_object.type != "MESH":
        return

    mesh_data = mesh_object.data
    if len(mesh_data.color_attributes) == 0:
        print(f"{mesh_object.name}: no vertex color attributes, skipped")
        return

    color_attr = "Col" if "Col" in mesh_data.color_attributes else mesh_data.color_attributes[0].name

    color_attr_data = mesh_data.color_attributes[color_attr]
    if hasattr(color_attr_data, 'color_space'):
        color_attr_data.color_space = 'SRGB'

    material = bpy.data.materials.new(mesh_object.name + "_2dgs_mat")
    material.use_nodes = True

    nodes = material.node_tree.nodes
    links = material.node_tree.links

    for node in list(nodes):
        nodes.remove(node)

    output_node = nodes.new(type="ShaderNodeOutputMaterial")
    emission_node = nodes.new(type="ShaderNodeEmission")
    attribute_node = nodes.new(type="ShaderNodeAttribute")
    attribute_node.attribute_name = color_attr

    links.new(attribute_node.outputs["Color"], emission_node.inputs["Color"])
    emission_node.inputs["Strength"].default_value = 1.0
    links.new(emission_node.outputs["Emission"], output_node.inputs["Surface"])

    _ = MATERIAL_ROUGHNESS
    _ = MATERIAL_METALLIC
    _ = MATERIAL_SPECULAR

    mesh_data.materials.clear()
    mesh_data.materials.append(material)


def setup_2dgs_background():
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("2DGS_Default_World")
        bpy.context.scene.world = world
    world.use_nodes = True

    nodes = world.node_tree.nodes
    links = world.node_tree.links

    for node in list(nodes):
        nodes.remove(node)

    bg_node = nodes.new(type="ShaderNodeBackground")
    world_output = nodes.new(type="ShaderNodeOutputWorld")

    bg_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg_node.inputs["Strength"].default_value = 1.0

    links.new(bg_node.outputs["Background"], world_output.inputs["Surface"])


def apply_scene_render_defaults():
    scene = bpy.context.scene

    scene.render.fps = 24
    scene.render.resolution_x = max(scene.render.resolution_x, 1920)
    scene.render.resolution_y = max(scene.render.resolution_y, 1080)

    scene.render.resolution_percentage = RENDER_SCALE
    scene.render.filter_size = FILTER_SIZE
    bpy.context.preferences.system.gl_texture_limit = 'CLAMP_OFF'

    scene.render.engine = RENDER_ENGINE
    if RENDER_ENGINE == 'BLENDER_EEVEE':
        scene.eevee.taa_render_samples = 64
        scene.eevee.taa_samples = 32

        for attr in ['use_ssr', 'use_gtao', 'use_ao', 'use_gi', 'use_bloom']:
            try:
                setattr(scene.eevee, attr, False)
            except AttributeError:
                continue

    if hasattr(scene, "view_settings"):
        scene.view_settings.view_transform = 'Standard'
        scene.view_settings.look = 'None'
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0

    setup_2dgs_background()


apply_scene_render_defaults()

for scene_object in bpy.context.scene.objects:
    bind_vertex_color_material(scene_object)

for area in bpy.context.screen.areas:
    if area.type == "VIEW_3D":
        for space in area.spaces:
            if space.type == "VIEW_3D":
                space.shading.type = VIEW_SHADING
                space.shading.use_scene_lights = False
                space.shading.use_scene_world = True