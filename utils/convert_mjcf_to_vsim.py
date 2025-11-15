from sys import argv
import vlearn as v


def print_usage():
    print("Usage: {} MJCF_IN_FILE VSIM_OUT_FILE MJCF_MODEL_INDEX".format(argv[0]))
    exit(2)


if len(argv) != 4:
    print_usage()

mjcf_file = argv[1]
vsim_file = argv[2]
model_index = int(argv[3])

v.create_gym(with_render=False, treat_warning_as_error=True)
gym = v.get_gym()

print("Converting MJCF file '{}' to VSIM file '{}'".format(mjcf_file, vsim_file))
gym.convert_mjcf_to_vsim(mjcf_file, vsim_file, model_index)
print("Done!")
