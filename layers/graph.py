from __future__ import print_function

from keras import activations, initializers
from keras import regularizers
from keras.engine import Layer
from keras.layers import Dropout

import keras.backend as K
import tensorflow as tf


class GraphConvolution(Layer):
    def __init__(self, output_dim, support=1, featureless=False,
                 init='glorot_uniform', activation='linear',
                 weights=None, W_regularizer=None, num_bases=-1,
                 b_regularizer=None, bias=False, dropout=0., **kwargs):
        self.init = initializers.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim  # number of features per node
        self.support = support  # filter support / number of weights
        self.featureless = featureless  # use/ignore input features
        self.dropout = dropout

        assert support >= 1

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.bias = bias
        self.initial_weights = weights
        self.num_bases = num_bases

        # these will be defined during build()
        self.input_dim = None
        self.W = None
        self.W_comp = None
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
        self.num_nodes = input_shapes[1][1]  # assume A = n x n

        # generate weights
        if self.num_bases > 0:
            # B x f x h  // B := number of basis functions
            self.W = tf.concat([[self.add_weight((self.input_dim, self.output_dim),
                                                 initializer=self.init,
                                                 name='{}_W'.format(self.name),
                                                 regularizer=self.W_regularizer)] for _ in range(self.num_bases)],
                                   axis=0)

            self.W_comp = self.add_weight((self.support, self.num_bases),
                                          initializer=self.init,
                                          name='{}_W_comp'.format(self.name),
                                          regularizer=self.W_regularizer)
        else:
            # R x f x h  // R := number of relations
            self.W = tf.concat([[self.add_weight((self.input_dim, self.output_dim),
                                                 initializer=self.init,
                                                 name='{}_W'.format(self.name),
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
        # inputs = [X, A_0, A_1, ..., A_r]
        X = inputs[0]  # feature matrix (n x f)
        if type(X) is tf.SparseTensor:
            # let X be dense
            X = tf.sparse_tensor_to_dense(X)

        # merge adjacency matrices to (n x nR)
        A = tf.sparse_concat(sp_inputs=inputs[1:], axis=1)

        # reduce weight matrix if bases are used
        if self.num_bases > 0: # TODO: check:
            self.W = tf.transpose(self.W, perm=[1, 0, 2])  # transpose to f x B x h
            self.W = tf.einsum('ij,bjk->bik', self.W_comp, self.W)  # (R x B)*(f x B x h) = f x R x h
            self.W = tf.transpose(self.W, perm=[1, 0, 2])  # transpose to R x f x h

        # graph convolve
        XW = tf.einsum('ij,bjk->bik', X, self.W)  # R x n x h
        XW = tf.reshape(XW, [self.support*self.num_nodes, self.output_dim])  # Rn x h
        AXW = tf.sparse_tensor_dense_matmul(A, XW)  # n x h

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
