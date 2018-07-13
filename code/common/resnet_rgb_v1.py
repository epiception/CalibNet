import numpy as np
import tensorflow as tf
import json
from cnn_utils import *
import config_res

with open(config_res.paths['resnet_params_path']) as f_in:
    parameters = json.load(f_in)

    class ResNet:
    
        def __init__(self, input_x, phase, parameters = parameters):

            self.input_x = input_x
            self.phase = phase
            self.parameters = parameters
            self.layer_zero
            self.normal_res_block
            self.shortcut_res_block
            self.Net

        def Net(self):
            out = self.layer_zero(self.input_x)
            
            out = self.normal_res_block(out, 1, 1)
            out = self.normal_res_block(out, 1, 2)
            
            for layer_x in range(2,5):
                
                out = self.shortcut_res_block(out, layer_x, 1)
                out = self.normal_res_block(out, layer_x, 2)
            
            return out

        def layer_zero(self, layer_input):

            layer_dict =self.parameters['layer0']
            bl_str = "block_1"

            W = np.array(layer_dict[bl_str]['conv1']['weight'], dtype = np.float32)
            bn_mov_mean = np.array(layer_dict[bl_str]['bn1']['running_mean'], dtype = np.float32)
            bn_mov_var = np.array(layer_dict[bl_str]['bn1']['running_var'], dtype = np.float32)
            bn_gamma = np.array(layer_dict[bl_str]['bn1']['weight'], dtype = np.float32)
            bn_beta = np.array(layer_dict[bl_str]['bn1']['bias'], dtype = np.float32)

            with tf.variable_scope('layer0'):
                x = conv2d(layer_input, W, name = "W",to_train = False)
                x = batchnorm(x, self.phase, "0", bn_beta, bn_gamma, bn_mov_mean, bn_mov_var)
                x = tf.nn.relu(x)
                x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

            return x

        def normal_res_block(self, layer_input, layer_no, block_no):
            
            layer_dict = self.parameters['layer%d'%layer_no]
            bl_str = "block_%d"%block_no

            W1 = np.array(layer_dict[bl_str]['conv1']['weight'], dtype = np.float32)
            bn_mov_mean1 = np.array(layer_dict[bl_str]['bn1']['running_mean'], dtype = np.float32)
            bn_mov_var1 = np.array(layer_dict[bl_str]['bn1']['running_var'], dtype = np.float32)
            bn_gamma1 = np.array(layer_dict[bl_str]['bn1']['weight'], dtype = np.float32)
            bn_beta1 = np.array(layer_dict[bl_str]['bn1']['bias'], dtype = np.float32)

            W2 = np.array(layer_dict[bl_str]['conv2']['weight'], dtype = np.float32)
            bn_mov_mean2 = np.array(layer_dict[bl_str]['bn2']['running_mean'], dtype = np.float32)
            bn_mov_var2 = np.array(layer_dict[bl_str]['bn2']['running_var'], dtype = np.float32)
            bn_gamma2 = np.array(layer_dict[bl_str]['bn2']['weight'], dtype = np.float32)
            bn_beta2 = np.array(layer_dict[bl_str]['bn2']['bias'], dtype = np.float32)

            with tf.variable_scope("normal_res_%d"%layer_no + "_block_%d"%block_no):

                residual = layer_input

                x = conv2d(layer_input, W1, name = "W1", to_train = False)
                x = batchnorm(x, self.phase, "1", bn_beta1, bn_gamma1, bn_mov_mean1, bn_mov_var1)
                x = tf.nn.relu(x)
                x = conv2d(x, W2, name = "W2", to_train = False)
                x = batchnorm(x, self.phase, "2", bn_beta2, bn_gamma2, bn_mov_mean2, bn_mov_var2)
                x = residual + x

                x =  tf.nn.relu(x)

            return x

        def shortcut_res_block(self, layer_input, layer_no, block_no):
            
            layer_dict = self.parameters['layer%d'%layer_no]
            bl_str = "block_%d"%block_no

            W1 = np.array(layer_dict[bl_str]['conv1']['weight'], dtype = np.float32)
            bn_mov_mean1 = np.array(layer_dict[bl_str]['bn1']['running_mean'], dtype = np.float32)
            bn_mov_var1 = np.array(layer_dict[bl_str]['bn1']['running_var'], dtype = np.float32)
            bn_gamma1 = np.array(layer_dict[bl_str]['bn1']['weight'], dtype = np.float32)
            bn_beta1 = np.array(layer_dict[bl_str]['bn1']['bias'], dtype = np.float32)

            W2 = np.array(layer_dict[bl_str]['conv2']['weight'], dtype = np.float32)
            bn_mov_mean2 = np.array(layer_dict[bl_str]['bn2']['running_mean'], dtype = np.float32)
            bn_mov_var2 = np.array(layer_dict[bl_str]['bn2']['running_var'], dtype = np.float32)
            bn_gamma2 = np.array(layer_dict[bl_str]['bn2']['weight'], dtype = np.float32)
            bn_beta2 = np.array(layer_dict[bl_str]['bn2']['bias'], dtype = np.float32)

            downsample_dict = self.parameters['layer%d_downsample'%layer_no]
            W_dn = np.array(downsample_dict['block_1']['conv']['weight'], dtype = np.float32)
            bn_mov_mean_dn = np.array(downsample_dict['block_1']['bn']['running_mean'], dtype = np.float32)
            bn_mov_var_dn = np.array(downsample_dict['block_1']['bn']['running_var'], dtype = np.float32)
            bn_gamma_dn = np.array(downsample_dict['block_1']['bn']['weight'], dtype = np.float32)
            bn_beta_dn = np.array(downsample_dict['block_1']['bn']['bias'], dtype = np.float32)

            with tf.variable_scope("shortcut_res_%d"%layer_no + "_block_%d"%block_no):

                residual = conv2d(layer_input, W_dn, name = "Wdn", to_train = False, stride=[1,2,2,1])
                residual = batchnorm(residual, self.phase, "res", bn_beta_dn, bn_gamma_dn, bn_mov_mean_dn, bn_mov_var_dn)

                x = conv2d(layer_input, W1, name = "W1", to_train = False, stride = [1,2,2,1])
                x = batchnorm(x, self.phase, "1", bn_beta1, bn_gamma1, bn_mov_mean1, bn_mov_var1)
                x = tf.nn.relu(x)
                x = conv2d(x, W2, name = "W2", to_train = False)
                x = batchnorm(x, self.phase, "2",bn_beta2, bn_gamma2, bn_mov_mean2, bn_mov_var2)
                x = residual + x

                x =  tf.nn.relu(x)

            return x



    



"""
    
    def layer(self, layer_input, layer_no):
        layer_dict = self.parameters['layer%d'%layer_no]

        cur = layer_input
        res = layer_input

        for b_no in range(1,3):
            bl_str = "block_%d"%b_no

            stride = [0,0]
            if(b_no == 1):
                stride = [2,1]
            else:
                stride = [1,1]

            # for in_bno in range(1,3):

            W1 = np.array(layer_dict[bl_str]['conv1']['weight'], dtype = np.float32)
            bn_mov_mean1 = np.array(layer_dict[bl_str]['bn1']['running_mean'], dtype = np.float32)
            bn_mov_var1 = np.array(layer_dict[bl_str]['bn1']['running_var'], dtype = np.float32)
            bn_gamma1 = np.array(layer_dict[bl_str]['bn1']['weight'], dtype = np.float32)
            bn_beta1 = np.array(layer_dict[bl_str]['bn1']['bias'], dtype = np.float32)

            W2 = np.array(layer_dict[bl_str]['conv2']['weight'], dtype = np.float32)
            bn_mov_mean2 = np.array(layer_dict[bl_str]['bn2']['running_mean'], dtype = np.float32)
            bn_mov_var2 = np.array(layer_dict[bl_str]['bn2']['running_var'], dtype = np.float32)
            bn_gamma2 = np.array(layer_dict[bl_str]['bn2']['weight'], dtype = np.float32)
            bn_beta2 = np.array(layer_dict[bl_str]['bn2']['bias'], dtype = np.float32)

            # W_conv1 = init_weights(W1, "_l_%d_bl_%d_no_%d"%(layer_no,b_no, 1), False)
            # W_conv2 = init_weights(W2, "_l_%d_bl_%d_no_%d"%(layer_no,b_no, 2), False)

            # with tf.name_scope("layer_%d_%d_1"%(layer_no,b_no)):
            out1 = conv2d_batchnorm(cur, W_conv1, "layer_%d_%d_1"%(layer_no,b_no), self.phase, bn_beta1, bn_gamma1, bn_mov_mean1, bn_mov_var1, [1,stride[0],stride[0],1], False)

            print("layer_%d_%d_1"%(layer_no,b_no), out1.shape)

            #  if layer1 no downsample, so stride 2,1 then 1,1 
            # else stride 2,1 then downsample then 1,1 

            if(layer_no > 1 and b_no == 1):
                

                W_conv_dn = init_weights(W_dn, "downsample_%d"%(layer_no), False)
                # with tf.name_scope("downsample_layer_%d_%d"%(layer_no,b_no)):
                res = conv2d_batchnorm(res, W_conv_dn, "layer_dn_%d"%(layer_no), self.phase, bn_beta_dn, bn_gamma_dn, bn_mov_mean_dn, bn_mov_var_dn, [1,2,2,1], False)

                print("downsample_layer_%d_%d_1"%(layer_no,b_no), res.shape)

                out1 = tf.nn.relu(out1 + res)

            else:
                out1 = tf.nn.relu(out1)

            # with tf.name_scope("layer_%d_%d_2"%(layer_no,b_no)):
            out2 = conv2d_batchnorm(out1, W_conv2, "layer_%d_%d_2"%(layer_no,b_no), self.phase, bn_beta2, bn_gamma2, bn_mov_mean2, bn_mov_var2, [1,stride[1],stride[1],1], True)
            print("layer_%d_%d_2"%(layer_no,b_no), out2.shape)
            cur = out2

        return cur




def _residual_block_first(self, x, out_channel, strides, name="unit"):
        in_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)

            # Shortcut connection
            if in_channel == out_channel:
                if strides == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
            else:
                shortcut = self._conv(x, 1, out_channel, strides, name='shortcut')
            # Residual
            x = self._conv(x, 3, out_channel, strides, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, out_channel, 1, name='conv_2')
            x = self._bn(x, name='bn_2')
            # Merge
            x = x + shortcut
            x = self._relu(x, name='relu_2')
        return x


def _residual_block(self, x, input_q=None, output_q=None, name="unit"):
    num_channel = x.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        print('\tBuilding residual unit: %s' % scope.name)
        # Shortcut connection
        shortcut = x
        # Residual
        x = self._conv(x, 3, num_channel, 1, input_q=input_q, output_q=output_q, name='conv_1')
        x = self._bn(x, name='bn_1')
        x = self._relu(x, name='relu_1')
        x = self._conv(x, 3, num_channel, 1, input_q=output_q, output_q=output_q, name='conv_2')
        x = self._bn(x, name='bn_2')

        x = x + shortcut
        x = self._relu(x, name='relu_2')
    return x

"""