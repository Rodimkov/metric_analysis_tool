import argparse
import os
import task_factory


def create_parser():
    argument_parser = argparse.ArgumentParser()

    argument_parser.add_argument('-t', '--temp', required=True,
                                 help="temporary solution")
    argument_parser.add_argument('-f', '--file', nargs='+', type=file_path,
                                 help="results description file")
    argument_parser.add_argument('-d', '--directory', required=True, type=dir_path,
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


def main():
    parser = create_parser()
    namespace = parser.parse_args()

    task = None

    if namespace.temp == '0':
        factory = task_factory.MetricAnalysisFactory()
        task = factory.create_task("Classification", namespace.file[0], namespace.directory)

    task.visualize_data()
    task.metrics()


if __name__ == '__main__':
    main()