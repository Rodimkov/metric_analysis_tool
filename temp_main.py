import argparse
import os
import task_factory
import json
import orjson
import GUI.main_gui as mg
import GUI_main
# This file is a temporary solution and contains inaccuracies.
# As a result, it is not written optimally


def create_parser():
    argument_parser = argparse.ArgumentParser()

    argument_parser.add_argument('-f', '--file', nargs='+', type=file_path, default=[None, None],
                                 help="results description file")

    argument_parser.add_argument('-d', '--directory', required=True, type=dir_path,
                                 help="directory containing data")

    argument_parser.add_argument('-m', '--mask', nargs='+', default=[None, None],
                                 help="directory containing data")

    return argument_parser


def file_path(path):
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError("{} is not a valid file".format(path))

    _, file_extension = os.path.splitext(path)

    if file_extension != '.json':
        raise argparse.ArgumentTypeError("{} is not a json format file".format(path))
    return path


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError("{} is not a valid path".format(path))


def read_data(json_file):
    with open(json_file, "r") as read_file:
        data = json.load(read_file)

    #with open(json_file, "rb") as read_file:
    #   data = orjson.loads(read_file.read())
    try:
        if not 'report_type' in data:
            raise Exception('report_type')
    except Exception as e:
        print("no key '{}' in file <json>".format(e))
        raise SystemExit(1)

    type_task = data.get("report_type")

    return data, type_task

def main():
    parser = create_parser()
    namespace = parser.parse_args()

    task = GUI_main.MainMenu(namespace.file, namespace.directory, namespace.mask)


        #task.visualize_data()
        #task.metrics()
        #task.top_n()




if __name__ == '__main__':
    main()