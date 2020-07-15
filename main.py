import argparse
import json
import detection
import classification


def create_parser():
    argument_parser = argparse.ArgumentParser()

    argument_parser.add_argument('-t', '--temp', required=True,
                                 help="temporary solution")
    argument_parser.add_argument('-f', '--file', nargs='+',
                                 help="results description file")
    argument_parser.add_argument('-d', '--directory', required=True,
                                 help="directory containing data")

    return argument_parser


def read_data(json_file):
    with open(json_file, "r") as read_file:
        return json.load(read_file)


if __name__ == '__main__':
    parser = create_parser()
    namespace = parser.parse_args()

    data = read_data(namespace.file[0])

    if namespace.temp == '0':
        classification.visual(data, namespace.directory)
    if namespace.temp == '1':
        detection.visual(data, namespace.directory)
