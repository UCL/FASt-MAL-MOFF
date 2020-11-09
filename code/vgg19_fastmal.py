import tensorflow as tf

import numpy as np
from functools import reduce

VGG_MEAN = [103.939, 116.779, 123.68]

#IMSIZE = 512

class Vgg19:
    """
    A trainable version VGG19.
    """

    def __init__(self, vgg19_npy_path=None, trainable=True, dropout=0.5, imsize=512):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None
        self.imsize=imsize    
        self.var_dict = {}
        self.trainable = trainable
        self.dropout = dropout
    def conv_layers(self, input_batch):
        self.conv1_1 = self.conv_layer(input_batch, 3, 64, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4")
        pool5 = self.max_pool(self.conv5_4, 'pool5')
        return pool5        
        
    def build(self, rgb, train_mode=None):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [self.imsize, self.imsize, 1]
        assert green.get_shape().as_list()[1:] == [self.imsize, self.imsize, 1]
        assert blue.get_shape().as_list()[1:] == [self.imsize, self.imsize, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [self.imsize, self.imsize, 3]

        self.pool5 = self.conv_layers(bgr)
        self.avg_pool5 = tf.reduce_mean(self.conv_layers(bgr), axis=0)

        self.fc6 = self.fc_layer(self.pool5, 25088, 4096, "fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
        self.relu6 = tf.nn.relu(self.fc6)
        if train_mode is not None:
            self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, self.dropout), lambda: self.relu6)
        elif self.trainable:
            self.relu6 = tf.nn.dropout(self.relu6, self.dropout)

        self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        if train_mode is not None:
            self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, self.dropout), lambda: self.relu7)
        elif self.trainable:
            self.relu7 = tf.nn.dropout(self.relu7, self.dropout)

        self.fc8 = self.fc_layer(self.relu7, 4096, 1000, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None

    def build_simple(self, rgb, train_mode=None):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [self.imsize, self.imsize, 1]
        assert green.get_shape().as_list()[1:] == [self.imsize, self.imsize, 1]
        assert blue.get_shape().as_list()[1:] == [self.imsize, self.imsize, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [self.imsize, self.imsize, 3]

        self.pool5 = self.conv_layers(bgr)
        #self.avg_pool5 = tf.reduce_mean(self.conv_layers(bgr), axis=0)

        self.new_fc6 = self.fc_layer(self.pool5, 131072, 4096, "new_fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
        self.new_relu6 = tf.nn.relu(self.new_fc6)
        if train_mode is not None:
            self.new_relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.new_relu6, self.dropout), lambda: self.new_relu6)
        elif self.trainable:
            self.new_relu6 = tf.nn.dropout(self.new_relu6, self.dropout)

        self.new_fc7 = self.fc_layer(self.new_relu6, 4096, 2048, "new_fc7")
        self.new_relu7 = tf.nn.relu(self.new_fc7)
        if train_mode is not None:
            self.new_relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.new_relu7, self.dropout), lambda: self.new_relu7)
        elif self.trainable:
            self.new_relu7 = tf.nn.dropout(self.new_relu7, self.dropout)

        self.new_fc8 = self.fc_layer(self.new_relu7, 2048, 2, "new_fc8")

        self.prob = tf.nn.softmax(self.new_fc8, name="new_prob")

        self.data_dict = None
        
    def build_2classes(self, rgb, train_mode=None):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [self.imsize, self.imsize, 1]
        assert green.get_shape().as_list()[1:] == [self.imsize, self.imsize, 1]
        assert blue.get_shape().as_list()[1:] == [self.imsize, self.imsize, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [self.imsize, self.imsize, 3]

        self.pool5 = self.conv_layers(bgr)
        #self.avg_pool5 = tf.reduce_mean(self.conv_layers(bgr), axis=0)

        self.fc6 = self.fc_layer(self.pool5, 25088, 4096, "fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
        self.relu6 = tf.nn.relu(self.fc6)
        if train_mode is not None:
            self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, self.dropout), lambda: self.relu6)
        elif self.trainable:
            self.relu6 = tf.nn.dropout(self.relu6, self.dropout)

        self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        if train_mode is not None:
            self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, self.dropout), lambda: self.relu7)
        elif self.trainable:
            self.relu7 = tf.nn.dropout(self.relu7, self.dropout)

        self.new_fc8 = self.fc_layer(self.relu7, 4096, 2, "new_fc8")

        self.prob = tf.nn.softmax(self.new_fc8, name="prob")

        self.data_dict = None

        
        
    def build_avg_pool(self, rgb, train_mode=None):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [self.imsize, self.imsize, 1]
        assert green.get_shape().as_list()[1:] == [self.imsize, self.imsize, 1]
        assert blue.get_shape().as_list()[1:] == [self.imsize, self.imsize, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [self.imsize, self.imsize, 3]

        self.pool5 = self.conv_layers(bgr)
        self.avg_pool5 = tf.reduce_mean(self.conv_layers(bgr), axis=0)
        #self.avg_pool5 = tf.reduce_max(self.conv_layers(bgr), axis=0)

        #self.new_fc6 = self.fc_layer(self.avg_pool5, 2048, 1024, "new_fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
        self.new_fc6 = self.fc_layer(self.avg_pool5, 8192, 1024, "new_fc6")
        self.new_relu6 = tf.nn.relu(self.new_fc6)
        if train_mode is not None:
            self.new_relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.new_relu6, self.dropout), lambda: self.new_relu6)
        elif self.trainable:
            self.new_relu6 = tf.nn.dropout(self.new_relu6, self.dropout)

        self.new_fc7 = self.fc_layer(self.new_relu6, 1024, 512, "new_fc7")
        self.new_relu7 = tf.nn.relu(self.new_fc7)
        if train_mode is not None:
            self.new_relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.new_relu7, self.dropout), lambda: self.new_relu7)
        elif self.trainable:
            self.new_relu7 = tf.nn.dropout(self.new_relu7, self.dropout)

        self.new_fc8 = self.fc_layer(self.new_relu7, 512, 2, "new_fc8")

        self.new_prob = tf.nn.softmax(self.new_fc8, name="new_prob")  
        
        
        
    def build_avg_pool_fov(self, rgb, train_mode=None):
            """
            load variable from npy to build the VGG
    
            :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
            :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
            """
    
            rgb_scaled = rgb * 255.0
    
            # Convert RGB to BGR
            red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
            assert red.get_shape().as_list()[1:] == [self.imsize, self.imsize, 1]
            assert green.get_shape().as_list()[1:] == [self.imsize, self.imsize, 1]
            assert blue.get_shape().as_list()[1:] == [self.imsize, self.imsize, 1]
            bgr = tf.concat(axis=3, values=[
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])
            assert bgr.get_shape().as_list()[1:] == [self.imsize, self.imsize, 3]
    
            self.pool5 = self.conv_layers(bgr)
            self.avg_pool5 = tf.reduce_mean(self.conv_layers(bgr), axis=0)
            #self.avg_pool5 = tf.reduce_sum(self.conv_layers(bgr), axis=0)
    
            self.new_fc6 = self.fc_layer(self.avg_pool5, 8192, 1024, "new_fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
            self.new_relu6 = tf.nn.relu(self.new_fc6)
            if train_mode is not None:
                self.new_relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.new_relu6, self.dropout), lambda: self.new_relu6)
            elif self.trainable:
                self.new_relu6 = tf.nn.dropout(self.new_relu6, self.dropout)
    
            self.new_fc7 = self.fc_layer(self.new_relu6, 1024, 512, "new_fc7")
            self.new_relu7 = tf.nn.relu(self.new_fc7)
            if train_mode is not None:
                self.new_relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.new_relu7, self.dropout), lambda: self.new_relu7)
            elif self.trainable:
                self.new_relu7 = tf.nn.dropout(self.new_relu7, self.dropout)
    
            self.new_fc8 = self.fc_layer(self.new_relu7, 512, 2, "new_fc8")
    
            self.new_prob = tf.nn.softmax(self.new_fc8, name="new_prob")         

        #self.data_dict = None        
#        
#    def build_loop(self, rgb_list, no_images=10, train_mode=None, reuse=False):
#        """
#        load variable from npy to build the VGG
#
#        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
#        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
#        """
##        with tf.variable_scope('NewVGG'):
##        # image is 256 x 256 x input_c_dim
##            if reuse:
##                tf.get_variable_scope().reuse_variables()
##            else:
##                assert tf.get_variable_scope().reuse is False
#        rgb_list_scaled = rgb_list * 255.00
#
#        # Convert RGB to BGR
#        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_list_scaled)
#        assert red.get_shape().as_list()[1:] == [self.imsize, self.imsize, 1]
#        assert green.get_shape().as_list()[1:] == [self.imsize, self.imsize, 1]
#        assert blue.get_shape().as_list()[1:] == [self.imsize, self.imsize, 1]
#        bgr_list = tf.concat(axis=3, values=[
#            blue - VGG_MEAN[0],
#            green - VGG_MEAN[1],
#            red - VGG_MEAN[2],
#        ])
#        assert bgr_list.get_shape().as_list()[1:] == [self.imsize, self.imsize, 3]
#        
#        bgr = tf.expand_dims(bgr_list[0,:,:,:],0)
#        self.temp_pool5 = self.conv_layers(bgr)
#        stack_pool5 = self.temp_pool5 
#        
#        
#        ntens = tf.Variable(1.0)
#        for o in range(1, 10):
#            #print(o)
#            bgri=tf.expand_dims(bgr_list[o,:,:,:],0)
#            self.temp_pool5 = tf.math.add(self.temp_pool5, self.conv_layers(bgri))#, axis=0)
#            ntens.assign(ntens+1)
#            #stack_pool6 = tf.stack(stack_pool5, self.conv_layers(bgri))
#                #self.avg_pool5 = tf.reduce_mean(self.conv_layers(bgr), axis=0)    
#        self.temp_pool5 = self.temp_pool5/ntens         
#        #self.temp_pool5 = tf.reduce_mean(stack_pool5, axis=0) 
#        self.new_fc6 = self.fc_layer(self.temp_pool5, 131072, 4096, "new_fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
#        self.new_relu6 = tf.nn.relu(self.new_fc6)
#        if train_mode is not None:
#            self.new_relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.new_relu6, self.dropout), lambda: self.new_relu6)
#        elif self.trainable:
#            self.new_relu6 = tf.nn.dropout(self.new_relu6, self.dropout)
#
#        self.new_fc7 = self.fc_layer(self.new_relu6, 4096, 4096, "new_fc7")
#        self.new_relu7 = tf.nn.relu(self.new_fc7)
#        if train_mode is not None:
#            self.new_relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.new_relu7, self.dropout), lambda: self.new_relu7)
#        elif self.trainable:
#            self.new_relu7 = tf.nn.dropout(self.new_relu7, self.dropout)
#
#        self.new_fc8 = self.fc_layer(self.new_relu7, 4096, 2, "new_fc8")
#
#        self.new_prob = tf.nn.softmax(self.new_fc8, name="new_prob")        
        
        
        
        
        
        
#    def build_loop_tbnails(self, rgb_list, no_images=10, train_mode=None, reuse=False):
#        """
#        load variable from npy to build the VGG
#
#        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
#        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
#        """
##        with tf.variable_scope('NewVGG'):
##        # image is 256 x 256 x input_c_dim
##            if reuse:
##                tf.get_variable_scope().reuse_variables()
##            else:
##                assert tf.get_variable_scope().reuse is False
#        rgb_list_scaled = rgb_list * 255.00
#
#        # Convert RGB to BGR
#        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_list_scaled)
#        assert red.get_shape().as_list()[1:] == [self.imsize, self.imsize, 1]
#        assert green.get_shape().as_list()[1:] == [self.imsize, self.imsize, 1]
#        assert blue.get_shape().as_list()[1:] == [self.imsize, self.imsize, 1]
#        bgr_list = tf.concat(axis=3, values=[
#            blue - VGG_MEAN[0],
#            green - VGG_MEAN[1],
#            red - VGG_MEAN[2],
#        ])
#        assert bgr_list.get_shape().as_list()[1:] == [self.imsize, self.imsize, 3]
#        
#        bgr = tf.expand_dims(bgr_list[0,:,:,:],0)
#        self.temp_pool5 = self.conv_layers(bgr)
#        stack_pool5 = self.temp_pool5 
#        
#        
#        ntens = tf.Variable(1.0)
#        for o in range(1, no_images):
#            #print(o)
#            bgri=tf.expand_dims(bgr_list[o,:,:,:],0)
#            self.temp_pool5 = tf.math.add(self.temp_pool5, self.conv_layers(bgri))#, axis=0)
#            ntens.assign(ntens+1)
#            #stack_pool6 = tf.stack(stack_pool5, self.conv_layers(bgri))
#                #self.avg_pool5 = tf.reduce_mean(self.conv_layers(bgr), axis=0)    
#        self.temp_pool5 = self.temp_pool5/ntens         
#        #self.temp_pool5 = tf.reduce_mean(stack_pool5, axis=0) 
#        self.new_fc6 = self.fc_layer(self.temp_pool5, 2048, 2048, "new_fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
#        self.new_relu6 = tf.nn.relu(self.new_fc6)
#        if train_mode is not None:
#            self.new_relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.new_relu6, self.dropout), lambda: self.new_relu6)
#        elif self.trainable:
#            self.new_relu6 = tf.nn.dropout(self.new_relu6, self.dropout)
#
#        self.new_fc7 = self.fc_layer(self.new_relu6, 2048, 2048, "new_fc7")
#        self.new_relu7 = tf.nn.relu(self.new_fc7)
#        if train_mode is not None:
#            self.new_relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.new_relu7, self.dropout), lambda: self.new_relu7)
#        elif self.trainable:
#            self.new_relu7 = tf.nn.dropout(self.new_relu7, self.dropout)
#
#        self.new_fc8 = self.fc_layer(self.new_relu7, 2048, 2, "new_fc8")
#
#        self.new_prob = tf.nn.softmax(self.new_fc8, name="new_prob")            
        
        
        
        
        
        
        
        
        
        
        
        
        
        

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count
