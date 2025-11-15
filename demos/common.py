import vlearn as v


def import_ant_definition(env_def):
    filename = "assets/vsim/ant.vsim"
    env_def.import_definitions(filename)

    ant_def_handle = env_def.get_articulation_def_handle_by_name("torso")

    # Instantiate ant
    ant_rot = v.shortest_rotation(v.Vec3(0, 0, 1), v.Vec3(0, 1, 0))
    ant_pos = v.Vec3(0, 0.75, 0)

    ant_transform = v.Transform(ant_rot, ant_pos)
    ant_handle = env_def.create_articulation(ant_def_handle, ant_transform)

    return ant_def_handle, ant_handle
