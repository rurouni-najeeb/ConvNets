## Python code for numerically optimising an input image to yield a maximum class score
import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import os
import scipy.io
from scipy import misc

sys.path.insert(0, '../')
import visualising_vgg as vgg

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_DEPTH = 3
path = '../../Assignment_2/style_transfer/imagenet-vgg-verydeep-19.mat'

def _fully_connected(vgg_layers, prev_layer, layer, expected_layer_name):
    
    with tf.variable_scope(expected_layer_name) as scope:

        W, b = vgg._weights(vgg_layers, layer, expected_layer_name)
        W = tf.constant(W)
        b = tf.constant(b)
        try:
            w, x, y, z = W.shape
            W = tf.reshape(W, [int(w*x*y), int(z)])
        except:
            pass
        try:
            w, x, y, z = prev_layer.shape
            flattened = int(x*y*z)
        except:
            flattened = int(prev_layer.shape[1])
            
        flat = tf.reshape(prev_layer, [-1,flattened])
        full = tf.matmul(flat, W)
        relu = tf.nn.relu(full + b)
        return relu

def _dropout(prev_layer, keep_probs, expected_layer_name):
    
    with tf.variable_scope(expected_layer_name) as scope:
        return tf.nn.dropout(prev_layer, keep_probs)

def main():
    
    ## Loading the graph
    print 'Loading the graph..'
    vgg = scipy.io.loadmat(path)
    vgg_layers = vgg['layers']
    graph = {}
    with tf.variable_scope("placeholders") as scope:
        image = tf.placeholder(shape=[None,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH],dtype=tf.float32,name="Input")
        keep_probs = tf.placeholder(dtype=tf.float32,name="keep_probs")
    
    ## Assembling the graph
    graph = {} 
    graph['conv1_1']  = vgg._conv_relu(vgg_layers, image, 0, 'conv1_1')
    graph['conv1_2']  = vgg._conv_relu(vgg_layers, graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = vgg._avg_pool(graph['conv1_2'],'avgpool1')
    graph['conv2_1']  = vgg._conv_relu(vgg_layers, graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2']  = vgg._conv_relu(vgg_layers, graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = vgg._avg_pool(graph['conv2_2'],'avgpool2')
    graph['conv3_1']  = vgg._conv_relu(vgg_layers, graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2']  = vgg._conv_relu(vgg_layers, graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3']  = vgg._conv_relu(vgg_layers, graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4']  = vgg._conv_relu(vgg_layers, graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = vgg._avg_pool(graph['conv3_4'],'avgpool3')
    graph['conv4_1']  = vgg._conv_relu(vgg_layers, graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2']  = vgg._conv_relu(vgg_layers, graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3']  = vgg._conv_relu(vgg_layers, graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4']  = vgg._conv_relu(vgg_layers, graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = vgg._avg_pool(graph['conv4_4'],'avgpool4')
    graph['conv5_1']  = vgg._conv_relu(vgg_layers, graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2']  = vgg._conv_relu(vgg_layers, graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3']  = vgg._conv_relu(vgg_layers, graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4']  = vgg._conv_relu(vgg_layers, graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = vgg._avg_pool(graph['conv5_4'],'avgpool5')
    graph['fc6'] = _fully_connected(vgg_layers, graph['avgpool5'], 37, 'fc6')
    graph['dropout1'] = _dropout(graph['fc6'],keep_probs,'dropout1')
    graph['fc7'] = _fully_connected(vgg_layers, graph['dropout1'], 39, 'fc7')
    graph['dropout2'] = _dropout(graph['fc7'], keep_probs, 'dropout2')
    graph['fc8'] = _fully_connected(vgg_layers, graph['dropout2'], 41, 'fc8')