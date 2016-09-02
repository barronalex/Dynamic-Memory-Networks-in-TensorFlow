import numpy as np
import tensorflow as tf

def xavier_weight_init():
  def _xavier_initializer(shape, **kwargs):
    eps = np.sqrt(6) / np.sqrt(np.sum(shape))
    out = tf.random_uniform(shape, minval=-eps, maxval=eps)
    return out
  return _xavier_initializer

def test_initialization_basic():
  """
  Some simple tests for the initialization.
  """
  print "Running basic tests..."
  xavier_initializer = xavier_weight_init()
  shape = (1,)
  xavier_mat = xavier_initializer(shape)
  assert xavier_mat.get_shape() == shape

  shape = (1, 2, 3)
  xavier_mat = xavier_initializer(shape)
  assert xavier_mat.get_shape() == shape
  print "Basic (non-exhaustive) Xavier initialization tests pass\n"

def test_initialization():
  """ 
  Use this space to test your Xavier initialization code by running:
      python q1_initialization.py 
  This function will not be called by the autograder, nor will
  your tests be graded.
  """
  print "Running your tests..."
  ### YOUR CODE HERE
  raise NotImplementedError
  ### END YOUR CODE  

if __name__ == "__main__":
    test_initialization_basic()
