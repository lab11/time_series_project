import numpy as np
import tensorflow as tf
import numbers
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import tensor_shape 

def dropout(x, keep_prob, is_training=False, noise_shape=None, seed=None, name=None):  
    with ops.name_scope(name, "dropout", [x]) as name:
        # Convert everything to tensors if needed (nop)
        x = ops.convert_to_tensor(x, name="x")
        is_training = ops.convert_to_tensor(is_training, name='is_training')
        keep_prob = ops.convert_to_tensor(keep_prob,
                                        dtype=x.dtype,
                                        name="keep_prob")

        # We don't do this, just use the shape of the input layer
        noise_shape = noise_shape if noise_shape is not None else x.shape

        # Save a copy
        random_tensor = keep_prob

        # These two lines implement the `Bernoulli` function from eq (10) by
        # taking the supplied probabilities (keep_prob), adding Uniform~(0,1],
        # and then flooring the result.
        random_tensor += random_ops.random_uniform(noise_shape,
                                                   seed=seed,
                                                   dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)


        ret = tf.cond(is_training, lambda: x * binary_tensor,  # is_training=True
                                    lambda: x * keep_prob)     # is_training=False

        ret.set_shape(x.get_shape())
        return ret, binary_tensor
