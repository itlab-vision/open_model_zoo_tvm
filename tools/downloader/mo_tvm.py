import argparse
import sys
from functools import partial
import fnmatch

import tvm
from tvm import relay
from tvm.driver.tvmc.frontends import load_model

import common
from accuracy_checker.utils import get_path, cast_to_bool, check_file_existence, validate_print_interval, string_to_list


def add_common_args(parser):
    common_args = parser.add_argument_group('Common arguments')
    common_args.add_argument('--framework',
                             help='Name of the framework used to train the input model.',
                             type=str,
                             choices=['tf', 'caffe', 'mxnet', 'kaldi', 'onnx'])
    common_args.add_argument('--data_type', type=str)
    common_args.add_argument('--output_dir', type=str)
    common_args.add_argument('--model_name', type=str)
    common_args.add_argument('--input', type=str)
    common_args.add_argument('--output', type=str)
    common_args.add_argument('--input_model', type=str)
    common_args.add_argument('--target', type=str)
    common_args.add_argument('--mean_values', '-ms',
                              help='Mean values to be used for the input image per channel. ' +
                                   'Values to be provided in the (R,G,B) or [R,G,B] format. ' +
                                   'Can be defined for desired input of the model, for example: ' +
                                   '"--mean_values data[255,255,255],info[255,255,255]". ' +
                                   'The exact meaning and order ' +
                                   'of channels depend on how the original model was trained.',
                              default=())
    common_args.add_argument('--scale_values',
                              help='Scale values to be used for the input image per channel. ' +
                                   'Values are provided in the (R,G,B) or [R,G,B] format. ' +
                                   'Can be defined for desired input of the model, for example: ' +
                                   '"--scale_values data[255,255,255],info[255,255,255]". ' +
                                   'The exact meaning and order ' +
                                   'of channels depend on how the original model was trained.',
                              default=())
    common_args.add_argument('--reverse_input_channels',
                              help='Switch the input channels order from RGB to BGR (or vice versa). Applied to '
                                   'original inputs of the model if and only if a number of channels equals 3. Applied '
                                   'after application of --mean_values and --scale_values options, so numbers in '
                                   '--mean_values and --scale_values go in the order of channels used in the original '
                                   'model.',
                              action='store_true')


def expand_arguments(parser):
    argv = parser.parse_args()
    model_info = get_model_info(argv)
    shape = get_shape_from_model_info(model_info)

    expanded_group = parser.add_argument_group('Expanded arguments')
    expanded_group.add_argument('--shape_dict', type=dict, default={argv.input: shape}, required=False)


def build_arguments_parser():
    parser = argparse.ArgumentParser(description='Deep Learning accuracy validation framework', allow_abbrev=False)
    add_common_args(parser)
    expand_arguments(parser)

    return parser


def get_model_info(args):
    return [model for model in common.load_models(args)
            if fnmatch.fnmatchcase(model.name, args.model_name)][0]


def get_shape_from_string(string):
    processed = string.replace(' ', '')
    processed = processed.replace('[', '')
    processed = processed.replace(']', '')
    processed = processed.split(',')
    return list(int(entry) for entry in processed)


def get_shape_from_model_info(model_info):
    if model_info.conversion_to_onnx_args:
        args_with_shape = model_info.conversion_to_onnx_args
        shape_pattern = '--input-shape='
    else:
        args_with_shape = model_info.mo_args
        shape_pattern = '--input_shape='

    str_shape = [arg for arg in args_with_shape if shape_pattern in arg][0]
    str_shape = str_shape.replace(shape_pattern, '')
    return get_shape_from_string(str_shape)


def main(cli_parser: argparse.ArgumentParser):
    argv = cli_parser.parse_args()
    print(argv)

    model = load_model(argv.input_model, argv.framework, argv.shape_dict)
    print(model.mod)
    model.export_classic_format()

    target = 'llvm'
    target_host = target
    arch = "x86_64"
    sdk = "macosx"
    from tvm.contrib import xcode

    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(model.mod, target=target, target_host=target_host, params=model.params)
        lib.export_library("/Users/agladyshev/workspace/open_model_zoo/public/resnet-50-pytorch/resnet-v1-50.dylib", fcompile=xcode.create_dylib, arch=arch, sdk=sdk)

    from tvm.contrib import graph_executor
    import numpy as np
    #
    dtype = "float32"
    dev = tvm.cpu(0)
    m = graph_executor.GraphModule(lib["default"](dev))
    m.set_input(argv.input, tvm.nd.array(np.zeros((1, 224, 224, 3), dtype=dtype)))
    m.run()
    tvm_output = m.get_output(0).asnumpy()
    print(tvm_output.shape)


if __name__ == '__main__':
    # from mo.utils.cli_parser import get_all_cli_parser
    # sys.exit(main(get_all_cli_parser()))
    sys.exit(main(build_arguments_parser()))
