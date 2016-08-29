class DMN(object):
  """Abstracts a Tensorflow graph for DMN models.

  Adds ability to do embedding.
  """
  def load_data(self):
    """Loads data from disk and stores it in memory."""

    raise NotImplementedError("Each Model must re-implement this method.")

  def add_placeholders(self):
    """Adds placeholder variables to tensorflow computational graph.

    Tensorflow uses placeholder variables to represent locations in a
    computational graph where data is inserted.  These placeholders are used as
    inputs by the rest of the model building code and will be fed data during
    training.

    See for more information:

    https://www.tensorflow.org/versions/r0.7/api_docs/python/io_ops.html#placeholders
    """
    raise NotImplementedError("Each Model must re-implement this method.")

  def add_model(self, input_data):
    """Implements core of model that transforms input_data into predictions.

    The core transformation for this model which transforms a batch of input
    data into a batch of predictions.

    Args:
      input_data: A tensor of shape (batch_size, n_features).
    Returns:
      out: A tensor of shape (batch_size, n_classes)
    """
    raise NotImplementedError("Each Model must re-implement this method.")

  def add_loss_op(self, pred):
    """Adds ops for loss to the computational graph.

    Args:
      pred: A tensor of shape (batch_size, n_classes)
    Returns:
      loss: A 0-d tensor (scalar) output
    """
    raise NotImplementedError("Each Model must re-implement this method.")

  def run_epoch(self, sess, input_data, input_labels):
    """Runs an epoch of training.

    Trains the model for one-epoch.
  
    Args:
      sess: tf.Session() object
      input_data: np.ndarray of shape (n_samples, n_features)
      input_labels: np.ndarray of shape (n_samples, n_classes)
    Returns:
      average_loss: scalar. Average minibatch loss of model on epoch.
    """
    raise NotImplementedError("Each Model must re-implement this method.")

  def get_question_representation(self, inputs):
    raise NotImplementedError("Each Model must re-implement this method.")
  
  def get_attention(self, q_vec, prev_memory):
    raise NotImplementedError("Each Model must re-implement this method.")

  def generate_episode(self, memory, q_vec, cands):
    raise NotImplementedError("Each Model must re-implement this method.")


