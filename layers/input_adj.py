from keras.engine.topology import Node, InputLayer
import keras.backend as K


def InputAdj(name=None, dtype=K.floatx(), sparse=False,
          tensor=None):
    shape = (None, None)
    input_layer = InputLayer(batch_input_shape=shape,
                             name=name, sparse=sparse, dtype=dtype)
    outputs = input_layer._inbound_nodes[0].output_tensors
    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs


class InputLayerAdj(InputLayer):
    def __init__(self, input_shape=None, batch_size=None,
                 batch_input_shape=None, dtype=None,
                 input_tensor=None, sparse=False, name=None):

        if not name:
            prefix = 'input'
            name = prefix + '_' + str(K.get_uid(prefix))
        super(InputLayer, self).__init__(dtype=dtype, name=name)

        self.trainable = False
        self.built = True
        self.sparse = sparse

        if input_shape and batch_input_shape:
            raise ValueError('Only provide the input_shape OR '
                             'batch_input_shape argument to '
                             'InputLayer, not both at the same time.')
        if input_tensor is not None and batch_input_shape is None:
            # If input_tensor is set, and batch_input_shape is not set:
            # Attempt automatic input shape inference.
            try:
                batch_input_shape = K.int_shape(input_tensor)
            except TypeError:
                if not input_shape and not batch_input_shape:
                    raise ValueError('InputLayer was provided '
                                     'an input_tensor argument, '
                                     'but its input shape cannot be '
                                     'automatically inferred. '
                                     'You should pass an input_shape or '
                                     'batch_input_shape argument.')
        if not batch_input_shape:
            if not input_shape:
                raise ValueError('An Input layer should be passed either '
                                 'a `batch_input_shape` or an `input_shape`.')
            else:
                batch_input_shape = (batch_size,) + tuple(input_shape)
        else:
            batch_input_shape = tuple(batch_input_shape)

        if not dtype:
            if input_tensor is None:
                dtype = K.floatx()
            else:
                dtype = K.dtype(input_tensor)

        self.batch_input_shape = batch_input_shape
        self.dtype = dtype

        if input_tensor is None:
            self.is_placeholder = True
            input_tensor = K.placeholder(shape=batch_input_shape,
                                         dtype=dtype,
                                         sparse=self.sparse,
                                         name=self.name)
        else:
            self.is_placeholder = False
            input_tensor._keras_shape = batch_input_shape

        # Create an input node to add to self.outbound_node
        # and set output_tensors' _keras_history.
        input_tensor._uses_learning_phase = False
        input_tensor._keras_history = (self, 0, 0)
        Node(self,
             inbound_layers=[],
             node_indices=[],
             tensor_indices=[],
             input_tensors=[input_tensor],
             output_tensors=[input_tensor],
             input_masks=[None],
             output_masks=[None],
             input_shapes=[batch_input_shape],
             output_shapes=[batch_input_shape])


    def get_config(self):
        config = {'batch_input_shape': self.batch_input_shape,
                  'input_dtype': self.dtype,
                  'sparse': self.sparse,
                  'name': self.name}
        return config
