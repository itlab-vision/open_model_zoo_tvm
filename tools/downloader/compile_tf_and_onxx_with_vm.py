import os
import onnx
import copy

import numpy as np
import sys
import json

import tvm
from tvm import te
from tvm import relay
from tvm.relay import vm
import tvm.contrib.utils
from ctypes import *
from tvm.contrib.download import download_testdata

from tvm.contrib import utils, ndk


def main():
    model_name = "ssd-resnet34-1200-onnx"
    path_to_model = "/Users/admin/workspace/scripts/models/OMZ_compiled/ssd-resnet34-1200-onnx/resnet34-ssd1200.onnx"
    compiled_model_save_dir = "/Users/admin/workspace/scripts/models/OMZ_compiled/ssd-resnet34-1200-onnx/"


    if path_to_model.endswith('.onnx'): 
        shape_dict = {"image": [1, 3, 1200, 1200]}
        onnx_model = onnx.load(path_to_model)
        mod, params = relay.frontend.from_onnx(onnx_model, shape=shape_dict, freeze_params=True)
        mod = relay.transform.DynamicToStatic()(mod)
    elif path_to_model.endswith('.pb'):
        graph_def = None
        tf_model_file = path_to_model

        import tensorflow as tf
        try:
            tf_compat_v1 = tf.compat.v1
        except ImportError:
            tf_compat_v1 = tf

        import tvm.relay.testing.tf as tf_testing

        with tf_compat_v1.gfile.GFile(tf_model_file, "rb") as f:
            graph_def = tf_compat_v1.GraphDef()
            graph_def.ParseFromString(f.read())
            graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    
        shape_dict = {"input_1": (1, 416, 416, 3)}
        mod, params = relay.frontend.from_tensorflow(graph_def, shape=shape_dict,
                                        outputs=["conv2d_59/BiasAdd","conv2d_67/BiasAdd","conv2d_75/BiasAdd"])


    target = "llvm"
    target_host = "llvm -mtriple=arm64-linux-android"

    if target=="opencl" or target=="cuda":
        gpu = "_gpu"
        
        with tvm.transform.PassContext(opt_level=3):
            graph_module = vm.compile(copy.deepcopy(mod), target=target, target_host=target_host, params=params)
        code, lib = graph_module.save()

        lib_path = os.path.join(compiled_model_save_dir, f"{model_name}{gpu}.so")
        code_path = os.path.join(compiled_model_save_dir, f"{model_name}{gpu}.ro")
        serialized_exec_path = os.path.join(compiled_model_save_dir, f"exec_{model_name}{gpu}.so")
        graph_module.mod.export_library(serialized_exec_path, ndk.create_shared)
        lib.export_library(lib_path, ndk.create_shared)
    else:   
        gpu = "_cpu"

        with tvm.transform.PassContext(opt_level=3):
            graph_module = vm.compile(copy.deepcopy(mod), target=target, params=params)
        code, lib = graph_module.save()

        lib_path = os.path.join(compiled_model_save_dir, f"{model_name}{gpu}.so")
        code_path = os.path.join(compiled_model_save_dir, f"{model_name}{gpu}.ro")
        serialized_exec_path = os.path.join(compiled_model_save_dir, f"exec_{model_name}{gpu}.so")
        graph_module.mod.export_library(serialized_exec_path)
        lib.export_library(lib_path)


    with open(code_path, "wb") as fo:
        fo.write(code)

    with open(compiled_model_save_dir + model_name + gpu + "_data.txt", 'w+') as f:
                f.write(json.dumps(
                {
                    "Shape_dict": shape_dict,
                    "Target": target,
                }
            ))


if __name__ == '__main__':
    main()