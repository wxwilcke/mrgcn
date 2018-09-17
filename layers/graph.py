from __future__ import print_function

from keras import activations, initializers
from keras import regularizers
from keras.engine import Layer
from keras.layers import Dropout

import keras.backend as K
import tensorflow as tf


class GraphConvolution(Layer):
    def __init__(self, output_dim, support=1, featureless=False,
                 input_layer=False, init='glorot_uniform', activation='linear',
                 weights=None, W_regularizer=None, num_bases=-1,
                 b_regularizer=None, bias=False, dropout=0., **kwargs):
        self.init = initializers.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim  # number of features per node
        self.support = support  # filter support / number of weights
        self.featureless = featureless  # use/ignore input features
        self.input_layer = input_layer  # adjust for input
        self.dropout = dropout

        assert support >= 1

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.bias = bias
        self.initial_weights = weights
        self.num_bases = num_bases

        # these will be defined during build()
        self.input_dim = None
        self.W_I = None
        self.W_F = None
        self.W_I_comp = None
        self.W_F_comp = None
        self.b = None
        self.num_nodes = None

        super().__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes[0]
        output_shape = (features_shape[0], self.output_dim)
        return output_shape  # (batch_size, output_dim)

    def build(self, input_shapes):
        assert len(input_shapes[0]) == 2
        self.input_dim = input_shapes[0][1]  # number of features
        self.num_nodes = int(input_shapes[1][1]/self.support)  # assume A = n x Rn

        # generate weights
        if self.num_bases > 0:
            if self.input_layer:
                # Bn x h  // B := number of basis functions
                self.W_I = tf.concat([self.add_weight((self.input_dim, self.output_dim),
                                                       initializer=self.init,
                                                       name='{}_W_I'.format(self.name),
                                                       regularizer=self.W_regularizer) for _ in range(self.num_bases)],
                                       axis=0)

                self.W_I_comp = self.add_weight((self.support, self.num_bases),
                                                 initializer=self.init,
                                                 name='{}_W_I_comp'.format(self.name),
                                                 regularizer=self.W_regularizer)

            if not self.featureless:
                # B x f x h  // B := number of basis functions
                self.W_F = tf.concat([[self.add_weight((self.input_dim, self.output_dim),
                                                        initializer=self.init,
                                                        name='{}_W_F'.format(self.name),
                                                        regularizer=self.W_regularizer)] for _ in range(self.num_bases)],
                                       axis=0)

                self.W_F_comp = self.add_weight((self.support, self.num_bases),
                                                 initializer=self.init,
                                                 name='{}_W_F_comp'.format(self.name),
                                                 regularizer=self.W_regularizer)

        else:
            if self.input_layer:
                # Rn x h  // R := number of relations
                self.W_I = tf.concat([self.add_weight((self.input_dim, self.output_dim),
                                                       initializer=self.init,
                                                       name='{}_W_I'.format(self.name),
                                                       regularizer=self.W_regularizer) for _ in range(self.support)],
                                       axis=0)

            if not self.featureless:
                # R x f x h  // R := number of relations
                self.W_F = tf.concat([[self.add_weight((self.input_dim, self.output_dim),
                                                        initializer=self.init,
                                                        name='{}_W_F'.format(self.name),
                                                        regularizer=self.W_regularizer)] for _ in range(self.support)],
                                       axis=0)

        if self.bias:
            self.b = self.add_weight((self.output_dim,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        # set self.built = True
        #super().build(input_shapes)

    def call(self, inputs, mask=None):
        # inputs = [X, A]
        X, A = inputs[0], inputs[1]


        ## compute graph features #############################################
        # AIW_I = AW_I

        AIW_I = tf.Variable(0.0, tf.float32)
        if self.input_layer:
            W_I = self.W_I
            # reduce weight matrix if basis functions are used
            if self.num_bases > 0:
                W_I = tf.reshape(W_I, [self.num_bases,
                                       self.num_nodes,
                                       self.output_dim])
                W_I = tf.transpose(W_I, perm=[1, 0, 2])
                W_I = tf.einsum('ij,bjk->bik', self.W_I_comp, W_I)
                W_I = tf.reshape(W_I, [self.support*self.num_nodes,
                                       self.output_dim])

            # convolve
            AIW_I = tf.sparse_tensor_dense_matmul(A, W_I)


        ## compute entity features ###########################################
        # AFW_F

        AFW_F = tf.Variable(0.0, tf.float32)
        if not self.featureless:
            F = X
            if type(X) is tf.SparseTensor:
                X = tf.sparse_tensor_to_dense(X)

            W_F = self.W_F
            # reduce weight matrix if basis functions are used
            if self.num_bases > 0:
                W_F = tf.transpose(W_F, perm=[1, 0, 2])
                W_F = tf.einsum('ij,bjk->bik', self.W_F_comp, W_F)

            # convolve
            FW_F = tf.einsum('ij,bjk->bik', F, W_F) # R x n x y
            FW_F = tf.reshape(FW_F, [self.support*self.num_nodes, self.output_dim])
            AFW_F = tf.sparse_tensor_dense_matmul(A, FW_F)


        ## compute output ####################################################
        # iff input layer: AXW = A[I F]W = AIW_I + AFW_F
        # else:            AXW = AHW
        AXW = tf.add(AIW_I, AFW_F)


        # if featureless add dropout to output, by elementwise multiplying with column vector of ones,
        # with dropout applied to the vector of ones.
        if self.featureless:
            tmp = K.ones(self.num_nodes)
            tmp_do = Dropout(self.dropout)(tmp)
            AXW = K.transpose(K.transpose(AXW) * tmp_do)

        if self.bias:
            AXW += self.b
        return self.activation(AXW)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'num_bases': self.num_bases,
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
