from sys import argv
import vlearn as v


def print_usage():
    print("Usage: {} URDF_IN_FILE VSIM_OUT_FILE".format(argv[0]))
    exit(2)


if len(argv) != 3:
    print_usage()

urdf_file = argv[1]
vsim_file = argv[2]

v.create_gym(with_render=False, treat_warning_as_error=True)
gym = v.get_gym()

print("Converting URDF file '{}' to VSIM file '{}'".format(urdf_file, vsim_file))
gym.convert_urdf_to_vsim(urdf_file, vsim_file)
print("Done!")
