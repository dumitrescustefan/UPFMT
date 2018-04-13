# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

from keras.engine import Layer
from keras import backend as k
import numpy as np
import tensorflow as tf

class Cartezian(Layer):
    
    def __init__(self, ** kwargs):
        super(Cartezian, self).__init__(** kwargs)
    
    def compute_output_shape(self, input_shape):
        print "input_shape =", input_shape
        return (input_shape[0], None, input_shape[2])
    
    def build(self, input_shape):
        super(Cartezian, self).build(input_shape) 
        
    def call(self, x):
        
        return tf.tensordot(x,x, axes = [[1], [0]])