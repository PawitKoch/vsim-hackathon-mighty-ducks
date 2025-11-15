from sys import argv
import vlearn as v


def print_usage():
    print("Usage: {} VSIM_IN_FILE VSIM_OUT_FILE".format(argv[0]))
    exit(2)


if len(argv) != 3:
    print_usage()

in_file = argv[1]
out_file = argv[2]

v.create_gym(with_render=False, treat_warning_as_error=True)
gym = v.get_gym()

print("Converting VSIM file '{}' to VSIM file '{}'".format(in_file, out_file))
gym.convert_vsim_to_vsim(in_file, out_file)
print("Done!")
