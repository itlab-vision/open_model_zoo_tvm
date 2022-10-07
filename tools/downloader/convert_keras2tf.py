#!/usr/bin/env python3

# Copyright (c) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import tensorflow as tf
import os

from pathlib import Path

def main():

    h5_filename = 'densenet201_weights_tf_dim_ordering_tf_kernels.h5'
    model_name = "densenet-201-tf"
    h5_dir = f'/Users/admin/workspace/open_model_zoo/tools/downloader/public/{model_name}/'
    savedmodel_output_dir = h5_dir + f'{model_name}.savedmodel'
    onnx_output_dir = "/Users/admin/workspace/scripts/models/OMZ_compiled/"
    if not os.path.exists(onnx_output_dir):
        os.mkdir(onnx_output_dir)

    tf.keras.backend.set_image_data_format('channels_last')

    model = tf.keras.applications.DenseNet201(
        weights=h5_dir+h5_filename
    )
    model.save(filepath=savedmodel_output_dir)

    os.system(f"python -m tf2onnx.convert --saved-model {savedmodel_output_dir} --output {onnx_output_dir}{model_name}.onnx")


if __name__ == '__main__':
    main()