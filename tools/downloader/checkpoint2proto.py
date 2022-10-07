import os, argparse
import tensorflow as tf
from tensorflow.python.tools import freeze_graph as freeze_tool

def freeze_graph(sess, input_checkpoint_path):
    saver = tf.train.Saver() # or your own Saver
    saver.restore(sess, input_checkpoint_path)

    absolute_model_dir = '/Users/admin/workspace/scripts/ckpt_converted'
    graph_file_name = 'saved_model.pb'
    graph_def = sess.graph.as_graph_def()
    tf.train.write_graph(graph_def, absolute_model_dir, graph_file_name)

    input_graph_path = absolute_model_dir + '/' + graph_file_name
    input_saver_def_path = ""
    input_binary = False
    
    graph = sess.graph
    nodes = [node.name for node in graph.as_graph_def().node]
    output_node_names = 'resnet_v2_50/predictions/Reshape_1:0'
    print("out node names:\n %s" % str(nodes))
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_graph_path = absolute_model_dir + "/tf-frozen_model.pb"
    clear_devices = True

    freeze_tool.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, input_checkpoint_path,
                              output_node_names, restore_op_name,
                              filename_tensor_name, output_graph_path,
                              clear_devices, "")