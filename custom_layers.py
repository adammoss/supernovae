from keras import backend as K
from keras.engine import Layer
from keras.layers.core import Lambda
from keras.layers import Merge
import theano

import copy
import inspect
import types as python_types
import marshal
import sys
import warnings

class MaskLambda(Layer):
  
    def __init__(self, function, output_shape=None, arguments={}, **kwargs):
        self.function = function
        self.arguments = arguments
        self.supports_masking = True
        if output_shape is None:
            self._output_shape = None
        elif type(output_shape) in {tuple, list}:
            self._output_shape = tuple(output_shape)
        else:
            if not hasattr(output_shape, '__call__'):
                raise Exception('In Lambda, `output_shape` '
                                'must be a list, a tuple, or a function.')
            self._output_shape = output_shape
        super(MaskLambda, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        if self._output_shape is None:
            # if TensorFlow, we can infer the output shape directly:
            if K._BACKEND == 'tensorflow':
                if type(input_shape) is list:
                    xs = [K.placeholder(shape=shape) for shape in input_shape]
                    x = self.call(xs)
                else:
                    x = K.placeholder(shape=input_shape)
                    x = self.call(x)
                if type(x) is list:
                    return [K.int_shape(x_elem) for x_elem in x]
                else:
                    return K.int_shape(x)
            # otherwise, we default to the input shape
            return input_shape
        elif type(self._output_shape) in {tuple, list}:
            nb_samples = input_shape[0] if input_shape else None
            return (nb_samples,) + tuple(self._output_shape)
        else:
            shape = self._output_shape(input_shape)
            if type(shape) not in {list, tuple}:
                raise Exception('output_shape function must return a tuple')
            return tuple(shape)

    def call(self, x, mask=None):
        arguments = self.arguments
        arg_spec = inspect.getargspec(self.function)
        if 'mask' in arg_spec.args:
            arguments['mask'] = mask
        return self.function(x, **arguments)

    def get_config(self):
        py3 = sys.version_info[0] == 3

        if isinstance(self.function, python_types.LambdaType):
            if py3:
                function = marshal.dumps(self.function.__code__).decode('raw_unicode_escape')
            else:
                function = marshal.dumps(self.function.func_code).decode('raw_unicode_escape')
            function_type = 'lambda'
        else:
            function = self.function.__name__
            function_type = 'function'

        if isinstance(self._output_shape, python_types.LambdaType):
            if py3:
                output_shape = marshal.dumps(self._output_shape.__code__)
            else:
                output_shape = marshal.dumps(self._output_shape.func_code)
            output_shape_type = 'lambda'
        elif callable(self._output_shape):
            output_shape = self._output_shape.__name__
            output_shape_type = 'function'
        else:
            output_shape = self._output_shape
            output_shape_type = 'raw'

        config = {'function': function,
                  'function_type': function_type,
                  'output_shape': output_shape,
                  'output_shape_type': output_shape_type,
                  'arguments': self.arguments}
        base_config = super(MaskLambda, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self,x,input_mask=None):
        return None

    @classmethod
    def from_config(cls, config):
        function_type = config.pop('function_type')
        if function_type == 'function':
            function = globals()[config['function']]
        elif function_type == 'lambda':
            function = marshal.loads(config['function'].encode('raw_unicode_escape'))
            function = python_types.FunctionType(function, globals())
        else:
            raise Exception('Unknown function type: ' + function_type)

        output_shape_type = config.pop('output_shape_type')
        if output_shape_type == 'function':
            output_shape = globals()[config['output_shape']]
        elif output_shape_type == 'lambda':
            output_shape = marshal.loads(config['output_shape'])
            output_shape = python_types.FunctionType(output_shape, globals())
        else:
            output_shape = config['output_shape']

        config['function'] = function
        config['output_shape'] = output_shape
        return cls(**config)

def lambda_mask_average(x,mask=None):
    return K.batch_dot(x,mask,axes=1) / K.sum(mask, axis=-1, keepdims=True)

class MaskMerge(Layer):
   
    def __init__(self, layers=None, mode='sum', concat_axis=-1,
                 dot_axes=-1, output_shape=None, output_mask=None,
                 node_indices=None, tensor_indices=None, name=None):
        self.layers = layers
        self.mode = mode
        self.concat_axis = concat_axis
        self.dot_axes = dot_axes
        if type(self.dot_axes) == int:
            self.dot_axes = [self.dot_axes, ] * 2
        self._output_shape = output_shape
        self.node_indices = node_indices
        self._output_mask = output_mask

        # layer parameters
        self.inbound_nodes = []
        self.outbound_nodes = []
        self.constraints = {}
        self.regularizers = []
        self.trainable_weights = []
        self.non_trainable_weights = []
        self.supports_masking = True
        self.uses_learning_phase = False
        self.input_spec = None  # compatible with whatever
        if not name:
            prefix = self.__class__.__name__.lower()
            name = prefix + '_' + str(K.get_uid(prefix))
        self.name = name

        if layers:
            # this exists for backwards compatibility.
            # equivalent to:
            # merge = Merge(layers=None)
            # output = merge([input_tensor_1, input_tensor_2])
            if not node_indices:
                # by default we connect to
                # the 1st output stream in the input layer
                node_indices = [0 for _ in range(len(layers))]
            self._arguments_validation(layers, mode,
                                       concat_axis, dot_axes,
                                       node_indices, tensor_indices)
            self.built = True
            self.add_inbound_node(layers, node_indices, tensor_indices)
        else:
            self.built = False

    def _arguments_validation(self, layers, mode, concat_axis, dot_axes,
                              node_indices, tensor_indices):
        '''Validates user-passed arguments and raises exceptions
        as appropriate.
        '''
        if not hasattr(mode, '__call__'):
            if mode not in {'sum', 'mul', 'concat', 'ave', 'cos', 'dot'}:
                raise Exception('Invalid merge mode: ' + str(mode))
        if type(layers) not in {list, tuple} or len(layers) < 2:
            raise Exception('A Merge should only be applied to a list of '
                            'layers with at least 2 elements. Found: ' + str(layers))

        if tensor_indices is None:
            tensor_indices = [None for _ in range(len(layers))]

        input_shapes = []
        for i, layer in enumerate(layers):
            layer_output_shape = layer.get_output_shape_at(node_indices[i])
            if type(layer_output_shape) is list:
                # case: the layer has multiple output tensors
                # and we only need a specific one
                layer_output_shape = layer_output_shape[tensor_indices[i]]
            input_shapes.append(layer_output_shape)

        if mode in {'sum', 'mul', 'ave', 'cos'}:
            input_shapes_set = set(input_shapes)
            if len(input_shapes_set) > 1:
                raise Exception('Only layers of same output shape can '
                                'be merged using ' + mode + ' mode. ' +
                                'Layer shapes: %s' % input_shapes)
        if mode in {'cos', 'dot'}:
            if len(layers) > 2:
                raise Exception(mode + ' merge takes exactly 2 layers')
            shape1 = input_shapes[0]
            shape2 = input_shapes[1]
            n1 = len(shape1)
            n2 = len(shape2)
            if mode == 'dot':
                if type(dot_axes) == int:
                    if dot_axes < 0:
                        dot_axes = [dot_axes % n1, dot_axes % n2]
                    else:
                        dot_axes = [n1 - dot_axes, n2-dot_axes]
                if type(dot_axes) not in [list, tuple]:
                    raise Exception('Invalid type for dot_axes - should be a list.')
                if len(dot_axes) != 2:
                    raise Exception('Invalid format for dot_axes - should contain two elements.')
                if type(dot_axes[0]) is not int or type(dot_axes[1]) is not int:
                    raise Exception('Invalid format for dot_axes - list elements should be "int".')
                if shape1[dot_axes[0]] != shape2[dot_axes[1]]:
                    raise Exception('Dimension incompatibility using dot mode: ' +
                                    '%s != %s. ' % (shape1[dot_axes[0]], shape2[dot_axes[1]]) +
                                    'Layer shapes: %s, %s' % (shape1, shape2))
        elif mode == 'concat':
            reduced_inputs_shapes = [list(shape) for shape in input_shapes]
            shape_set = set()
            for i in range(len(reduced_inputs_shapes)):
                del reduced_inputs_shapes[i][self.concat_axis]
                shape_set.add(tuple(reduced_inputs_shapes[i]))
            if len(shape_set) > 1:
                raise Exception('"concat" mode can only merge layers with matching ' +
                                'output shapes except for the concat axis. ' +
                                'Layer shapes: %s' % (input_shapes))

    def call(self, inputs, mask=None):

        if type(inputs) is not list or len(inputs) <= 1:
            raise Exception('Merge must be called on a list of tensors '
                            '(at least 2). Got: ' + str(inputs))
        # case: "mode" is a lambda or function.
        if hasattr(self.mode, '__call__'):
            # TODO: consider making it possible to
            # pass custom arguments to lambda.
            arguments = {}
            return self.mode(inputs, **arguments)

        if self.mode == 'sum' or self.mode == 'ave':
            s = inputs[0]
            for i in range(1, len(inputs)):
                s += inputs[i]
            if self.mode == 'ave':
                s /= len(inputs)
            return s

        elif self.mode == 'concat':
            return K.concatenate(inputs, axis=self.concat_axis)

        elif self.mode == 'mul':
            s = inputs[0]
            for i in range(1, len(inputs)):
                s *= inputs[i]
            return s

        elif self.mode == 'dot':
            l1 = inputs[0]
            l2 = inputs[1]
            output = K.batch_dot(l1, l2, self.dot_axes)
            return output

        elif self.mode == 'cos':
            l1 = inputs[0]
            l2 = inputs[1]
            denominator = K.sqrt(K.batch_dot(l1, l1, self.dot_axes) *
                                 K.batch_dot(l2, l2, self.dot_axes))
            denominator = K.maximum(denominator, K.epsilon())
            output = K.batch_dot(l1, l2, self.dot_axes) / denominator
            output = K.expand_dims(output, 1)
            return output
        else:
            raise Exception('Unknown merge mode.')

    def __call__(self, inputs, mask=None):
        '''We disable successive calls to __call__ for Merge layers.
        Although there is no technical obstacle to
        making it possible to __call__ a Merge instance many times
        (it is just a layer), it would make for a rather inelegant API.
        '''
        if type(inputs) is not list:
            raise Exception('Merge can only be called on a list of tensors, '
                            'not a single tensor. Received: ' + str(inputs))
        if self.built:
            raise Exception('A Merge layer cannot be used more than once, '
                            'please use ' +
                            'the "merge" function instead: ' +
                            '`merged_tensor = merge([tensor_1, tensor2])`.')

        all_keras_tensors = True
        for x in inputs:
            if not hasattr(x, '_keras_history'):
                all_keras_tensors = False
                break

        if all_keras_tensors:
            layers = []
            node_indices = []
            tensor_indices = []
            for x in inputs:
                layer, node_index, tensor_index = x._keras_history
                layers.append(layer)
                node_indices.append(node_index)
                tensor_indices.append(tensor_index)
            self._arguments_validation(layers, self.mode,
                                       self.concat_axis, self.dot_axes,
                                       node_indices, tensor_indices)
            self.built = True
            self.add_inbound_node(layers, node_indices, tensor_indices)

            outputs = self.inbound_nodes[-1].output_tensors
            return outputs[0]  # merge only returns a single tensor
        else:
            return self.call(inputs, mask)

    def get_output_shape_for(self, input_shape):
        assert type(input_shape) is list  # must have multiple input shape tuples
        # case: callable self._output_shape
        if hasattr(self.mode, '__call__'):
            if hasattr(self._output_shape, '__call__'):
                output_shape = self._output_shape(input_shape)
                return output_shape
            elif self._output_shape is not None:
                return (input_shape[0],) + tuple(self._output_shape)
            else:
                # TODO: consider shape auto-inference with TF
                raise Exception('The Merge layer ' + self.name +
                                ' has a callable `mode` argument, ' +
                                'and we cannot infer its output shape because ' +
                                'no `output_shape` argument was provided.' +
                                'Make sure to pass a shape tuple (or a callable) ' +
                                '`output_shape` to Merge.')
        # pre-defined merge modes
        input_shapes = input_shape
        if self.mode in ['sum', 'mul', 'ave']:
            # all tuples in input_shapes should be the same
            return input_shapes[0]
        elif self.mode == 'concat':
            output_shape = list(input_shapes[0])
            for shape in input_shapes[1:]:
                if output_shape[self.concat_axis] is None or shape[self.concat_axis] is None:
                    output_shape[self.concat_axis] = None
                    break
                output_shape[self.concat_axis] += shape[self.concat_axis]
            return tuple(output_shape)
        elif self.mode == 'dot':
            shape1 = list(input_shapes[0])
            shape2 = list(input_shapes[1])
            dot_axes = [a - 1 for a in self.dot_axes]
            tensordot_output = np.tensordot(np.zeros(tuple(shape1[1:])),
                                            np.zeros(tuple(shape2[1:])),
                                            axes=dot_axes)
            if len(tensordot_output.shape) == 0:
                shape = (1,)
            else:
                shape = tensordot_output.shape
            return (shape1[0],) + shape
        elif self.mode == 'cos':
            return (input_shapes[0][0], 1)

    def compute_mask(self, inputs, mask=None):

        if mask is None or not any([m is not None for m in mask]):
            return None

        assert hasattr(mask, '__len__') and len(mask) == len(inputs)

        if self.mode in ['sum', 'mul', 'ave']:
            bool_type = 'bool' if K._BACKEND == 'tensorflow' else 'int32'
            masks = [K.cast(m, bool_type) for m in mask if m is not None]
            mask = masks[0]
            for m in masks[1:]:
                mask = mask & m
            return mask
        elif self.mode in ['concat']:
            masks = [K.ones_like(inputs[i][:-1]) if m is None else m for i, m in zip(inputs, mask)]
            expanded_dims = [K.expand_dims(m) for m in masks]
            concatenated = K.concatenate(expanded_dims, axis=self.concat_axis)
            return K.all(concatenated, axis=-1, keepdims=False)
        elif self.mode in ['cos', 'dot']:
            return None
        elif hasattr(self.mode, '__call__'):
            if hasattr(self._output_mask, '__call__'):
                return self._output_mask(mask)
            else:
                return self._output_mask
        else:
            # this should have been caught earlier
            raise Exception('Invalid merge mode: {}'.format(self.mode))

    def get_config(self):
        py3 = sys.version_info[0] == 3

        if isinstance(self.mode, python_types.LambdaType):
            if py3:
                mode = marshal.dumps(self.mode.__code__).decode('raw_unicode_escape')
            else:
                mode = marshal.dumps(self.mode.func_code).decode('raw_unicode_escape')
            mode_type = 'lambda'
        elif callable(self.mode):
            mode = self.mode.__name__
            mode_type = 'function'
        else:
            mode = self.mode
            mode_type = 'raw'

        if isinstance(self._output_shape, python_types.LambdaType):
            if py3:
                output_shape = marshal.dumps(self._output_shape.__code__)
            else:
                output_shape = marshal.dumps(self._output_shape.func_code)
            output_shape_type = 'lambda'
        elif callable(self._output_shape):
            output_shape = self._output_shape.__name__
            output_shape_type = 'function'
        else:
            output_shape = self._output_shape
            output_shape_type = 'raw'

        return {'name': self.name,
                'mode': mode,
                'mode_type': mode_type,
                'concat_axis': self.concat_axis,
                'dot_axes': self.dot_axes,
                'output_shape': output_shape,
                'output_shape_type': output_shape_type}

    @classmethod
    def from_config(cls, config):
        mode_type = config.pop('mode_type')
        if mode_type == 'function':
            mode = globals()[config['mode']]
        elif mode_type == 'lambda':
            mode = marshal.loads(config['mode'].encode('raw_unicode_escape'))
            mode = python_types.FunctionType(mode, globals())
        else:
            mode = config['mode']

        output_shape_type = config.pop('output_shape_type')
        if output_shape_type == 'function':
            output_shape = globals()[config['output_shape']]
        elif output_shape_type == 'lambda':
            output_shape = marshal.loads(config['output_shape'])
            output_shape = python_types.FunctionType(output_shape, globals())
        else:
            output_shape = config['output_shape']

        config['mode'] = mode
        config['output_shape'] = output_shape
        return super(MaskMerge, cls).from_config(config)

def mask_merge(inputs, mode='sum', concat_axis=-1,
          dot_axes=-1, output_shape=None, output_mask=None, name=None):
    '''Functional merge, to apply to Keras tensors (NOT layers).
    Returns a Keras tensor.

    # Example usage:

    ```python
    tensor_a = Input(shape=(32,))
    tensor_b = Input(shape=(32,))
    merged_tensor = merge([tensor_a, tensor_b], mode='concat', concat_axis=1)
    ```

    # Arguments
        mode: string or lambda/function. If string, must be one
            of: 'sum', 'mul', 'concat', 'ave', 'cos', 'dot'.
            If lambda/function, it should take as input a list of tensors
            and return a single tensor.
        concat_axis: integer, axis to use in mode `concat`.
        dot_axes: integer or tuple of integers, axes to use in mode `dot`.
        output_shape: shape tuple (tuple of integers), or lambda/function
            to compute output_shape (only if merge mode is a lambda/function).
            If the latter case, it should take as input a list of shape tuples
            (1:1 mapping to input tensors) and return a single shape tuple, including the
            batch size (same convention as the `get_output_shape_for` method of layers).
        node_indices: optional list of integers containing
            the output node index for each input layer
            (in case some input layers have multiple output nodes).
            will default to an array of 0s if not provided.
        tensor_indices: optional list of indices of output tensors
            to consider for merging
            (in case some input layer node returns multiple tensors).
    '''
    all_keras_tensors = True
    for x in inputs:
        if not hasattr(x, '_keras_history'):
            all_keras_tensors = False
            break
    if all_keras_tensors:
        input_layers = []
        node_indices = []
        tensor_indices = []
        for x in inputs:
            input_layer, node_index, tensor_index = x._keras_history
            input_layers.append(input_layer)
            node_indices.append(node_index)
            tensor_indices.append(tensor_index)
        merge_layer = MaskMerge(input_layers, mode=mode,
                            concat_axis=concat_axis,
                            dot_axes=dot_axes,
                            output_shape=output_shape,
                            output_mask=output_mask,
                            node_indices=node_indices,
                            tensor_indices=tensor_indices,
                            name=name)
        return merge_layer.inbound_nodes[0].output_tensors[0]
    else:
        merge_layer = MaskMerge(mode=mode,
                            concat_axis=concat_axis,
                            dot_axes=dot_axes,
                            output_shape=output_shape,
                            output_mask=output_mask,
                            name=name)
        return merge_layer(inputs)
