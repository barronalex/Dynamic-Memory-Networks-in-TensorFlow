import getpass


import sys
import time

import numpy as np
import pandas as pd
from copy import deepcopy

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell

import utils

from model import DMN
import cPickle
import tables

floatX = np.float32

train_mode = True

# from https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/
def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].
    The input Tensor `t` should be a gradient.
    The output will be `t` + gaussian noise.
    0.001 was said to be a good fixed value for memory networks [2].
    """
    with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)


class Config(object):
  """Holds model hyperparams and data information.

  The config class is used to store various hyperparameters and dataset
  information parameters. Model objects are passed a Config() object at
  instantiation.
  """
  beta = 1
  alpha = 1

  restore = True

  batch_size = 50
  embed_size = 100
  hidden_size = 100
  max_epochs = 100
  early_stopping = 10
  dropout = 0.9
  drop_grus = True
  lr = 0.01
  l2 = 0.001
  anneal_threshold = 1
  anneal_by = 1.5
  num_hops = 5
  max_grad_val = 10
  num_gru_layers = 3
  num_train = 900

  babi_id = "9"
  babi_test_id = ""


  # can be word or sentence
  input_mask_mode = "sentence"

class DMN_QA_Model(DMN):

  

  def get_lens(self, inputs):
    lens = np.zeros((len(inputs)), dtype=int)
    for i, t in enumerate(inputs):
        lens[i] = t.shape[0]
    return lens

  def pad_inputs(self, inputs, lens, max_len, mask=False):
    if mask:
      padded = [np.pad(inp, (0, max_len - lens[i]), 'constant', constant_values=0) for i, inp in enumerate(inputs)]
      return np.stack(padded, axis=0)

    padded = [np.pad(np.squeeze(inp, axis=1), (0, max_len - lens[i]), 'constant', constant_values=0) for i, inp in enumerate(inputs)]
    return np.stack(padded, axis=0)
   

  def load_data(self, debug=False):


    """Loads starter word-vectors and train/dev/test data."""

    self.vocab = {}
    self.ivocab = {}

    self.babi_train_raw, self.babi_test_raw = utils.get_babi_raw(self.config.babi_id, self.config.babi_test_id)
    self.word2vec = utils.load_glove(self.config.embed_size)


    print '==> get train inputs'
    train_input, train_q, train_answer, train_input_mask, train_rel_labels = self._process_input(self.babi_train_raw)

    #convert word2vec to matrix representation
    self.word_embedding = utils.create_embedding(self.word2vec, self.ivocab, self.config.embed_size)

    self.train_input_lens = self.get_lens(train_input)
    self.train_q_lens = self.get_lens(train_q)
    self.train_mask_lens = self.get_lens(train_input_mask)

    max_train_input_len = np.max(self.train_input_lens)
    max_train_q_len = np.max(self.train_q_lens)
    max_train_mask_len = np.max(self.train_mask_lens)

    print '==> get test inputs'
    test_input, test_q, test_answer, test_input_mask, test_rel_labels = self._process_input(self.babi_test_raw)

    self.test_input_lens = self.get_lens(test_input)
    self.test_q_lens = self.get_lens(test_q)
    self.test_mask_lens = self.get_lens(test_input_mask)

    max_test_input_len = np.max(self.test_input_lens)
    max_test_q_len = np.max(self.test_q_lens)
    max_test_mask_len = np.max(self.test_mask_lens)

    self.max_q_len = np.max([max_train_q_len, max_test_q_len])
    self.max_input_len = np.max([max_train_input_len, max_test_input_len])
    self.max_mask_len = np.max([max_train_mask_len, max_test_mask_len])

    # first pad out arrays to max
    self.train_input = self.pad_inputs(train_input, self.train_input_lens, self.max_input_len)
    self.train_q = self.pad_inputs(train_q, self.train_q_lens, self.max_q_len)
    self.train_mask = self.pad_inputs(train_input_mask, self.train_mask_lens, self.max_mask_len, mask=True)

    self.test_input = self.pad_inputs(test_input, self.test_input_lens, self.max_input_len)
    self.test_q = self.pad_inputs(test_q, self.test_q_lens, self.max_q_len)
    self.test_mask = self.pad_inputs(test_input_mask, self.test_mask_lens, self.max_mask_len, mask=True)

    self.train_answers = np.stack(train_answer)
    self.test_answers = np.stack(test_answer)

    self.train_rel_labels = np.zeros((len(train_rel_labels), len(train_rel_labels[0])))
    self.test_rel_labels = np.zeros((len(test_rel_labels), len(test_rel_labels[0])))

    for i, tt in enumerate(train_rel_labels):
        self.train_rel_labels[i] = np.array(tt, dtype=int)

    for i, tt in enumerate(test_rel_labels):
        self.test_rel_labels[i] = np.array(tt, dtype=int)


    
    self.train = self.train_q[:self.config.num_train], self.train_input[:self.config.num_train], self.train_q_lens[:self.config.num_train], self.train_input_lens[:self.config.num_train], self.train_mask[:self.config.num_train], self.train_answers[:self.config.num_train], self.train_rel_labels[:self.config.num_train] 
    self.valid = self.train_q[self.config.num_train:], self.train_input[self.config.num_train:], self.train_q_lens[self.config.num_train:], self.train_input_lens[self.config.num_train:], self.train_mask[self.config.num_train:], self.train_answers[self.config.num_train:], self.train_rel_labels[self.config.num_train:] 
    self.test = self.test_q, self.test_input, self.test_q_lens, self.test_input_lens, self.test_mask, self.test_answers, self.test_rel_labels 

    self.visualize()
    sys.exit(0)
    


    self.vocab_size = len(self.vocab)

  def print_story(self, story, length):
      for i in range(length):
          print self.ivocab[story[i]],
      print '--------------------------'


  def visualize(self):
      for i in range(10):
          # first print the story
          self.print_story(self.train_input[i], self.train_input_lens[i])


  # from https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/
  def _process_input(self, data_raw):
    questions = []
    inputs = []
    answers = []
    input_masks = []
    relevant_labels = []
    for x in data_raw:
        inp = x["C"].lower().split(' ') 
        inp = [w for w in inp if len(w) > 0]
        q = x["Q"].lower().split(' ')
        q = [w for w in q if len(w) > 0]

        
        inp_vector = [utils.process_word(word = w, 
                                    word2vec = self.word2vec, 
                                    vocab = self.vocab, 
                                    ivocab = self.ivocab, 
                                    word_vector_size = self.config.embed_size, 
                                    to_return = "index") for w in inp]
                                    
        q_vector = [utils.process_word(word = w, 
                                    word2vec = self.word2vec, 
                                    vocab = self.vocab, 
                                    ivocab = self.ivocab, 
                                    word_vector_size = self.config.embed_size, 
                                    to_return = "index") for w in q]
        
        inputs.append(np.vstack(inp_vector).astype(floatX))
        questions.append(np.vstack(q_vector).astype(floatX))
        answers.append(utils.process_word(word = x["A"], 
                                        word2vec = self.word2vec, 
                                        vocab = self.vocab, 
                                        ivocab = self.ivocab, 
                                        word_vector_size = self.config.embed_size, 
                                        to_return = "index"))
        # NOTE: here we assume the answer is one word! 
        if self.config.input_mask_mode == 'word':
            input_masks.append(np.array([index for index, w in enumerate(inp)], dtype=np.int32)) 
        elif self.config.input_mask_mode == 'sentence': 
            input_masks.append(np.array([index for index, w in enumerate(inp) if w == '.'], dtype=np.int32)) 
        else:
            raise Exception("invalid input_mask_mode")

        relevant_labels.append(x["S"])
    
    return inputs, questions, answers, input_masks, relevant_labels 

  def add_placeholders(self):
    """Generate placeholder variables to represent the input tensors

    These placeholders are used as inputs by the rest of the model building
    code and will be fed data during training.  Note that when None is in a
    placeholder's shape, it's flexible

    Adds following nodes to the computational graph.
    (When None is in a placeholder's shape, it's flexible)

    input_placeholder: Input placeholder tensor of shape
                       (None, num_steps), type tf.int32
    labels_placeholder: Labels placeholder tensor of shape
                        (None, num_steps), type tf.int32
    dropout_placeholder: Dropout value placeholder (scalar),
                         type tf.float32

    Add these placeholders to self as the instance variables
  
      self.input_placeholder
      self.labels_placeholder
      self.dropout_placeholder

    (Don't change the variable names)
    """
    self.question_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.max_q_len))
    self.input_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.max_input_len))

    self.question_len_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size,))
    self.input_len_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size,))

    self.input_mask_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.max_mask_len))

    self.answer_placeholder = tf.placeholder(tf.int64, shape=(self.config.batch_size,))

    self.rel_label_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.train_rel_labels.shape[1]))

    self.dropout_placeholder = tf.placeholder(tf.float32)

    gru_cell = rnn_cell.GRUCell(self.config.hidden_size)
    if self.config.drop_grus:
        self.drop_gru = rnn_cell.DropoutWrapper(gru_cell, input_keep_prob=self.dropout_placeholder, output_keep_prob=self.dropout_placeholder)
    else:
        self.drop_gru = gru_cell
    multi_gru = rnn_cell.MultiRNNCell([gru_cell] * self.config.num_gru_layers)
    self.dropm_gru = rnn_cell.DropoutWrapper(multi_gru, input_keep_prob=self.dropout_placeholder, output_keep_prob=self.dropout_placeholder)

    with tf.variable_scope("memory/attention"):
        b_1 = tf.get_variable("b_1", (self.config.embed_size,))
        W_1 = tf.get_variable("W_1", (self.config.embed_size*7, self.config.embed_size))

        W_2 = tf.get_variable("W_2", (self.config.embed_size, 1))
        b_2 = tf.get_variable("b_2", 1)

  def add_embedding(self):

    embeddings = tf.Variable(self.word_embedding.astype(np.float32), name="Embedding")

    questions = tf.nn.embedding_lookup(embeddings, self.question_placeholder)
    inputs = tf.nn.embedding_lookup(embeddings, self.input_placeholder)

    # do dropout + regularization
    reg = self.config.l2*tf.nn.l2_loss(embeddings)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, reg)

    questions = tf.split(1, self.max_q_len, questions)
    inputs = tf.split(1, self.max_input_len, inputs)

    questions = [tf.squeeze(q, squeeze_dims=[1]) for q in questions]
    inputs = [tf.squeeze(i, squeeze_dims=[1]) for i in inputs]

    questions = [tf.nn.dropout(q, self.dropout_placeholder) for q in questions]
    inputs = [tf.nn.dropout(inn, self.dropout_placeholder) for inn in inputs]

    return questions, inputs
  
  def add_projection(self, rnn_output):
    """Adds a projection layer.

    The projection layer transforms the hidden representation to a distribution
    over the vocabulary.

    Hint: Here are the dimensions of the variables you will need to
          create 
          
          U:   (hidden_size, len(vocab))
          b_2: (len(vocab),)

    Args:
      rnn_output: a matrix of dimension (batch_size, hidden_size)
                   a tensor of shape (batch_size, embed_size).
    Returns:
      output: a matrix of shape (batch_size, fact_embed_size)
    """
    with tf.variable_scope("projection"):
        # in this baseline implementation, we train 3 different output matricies for
        # the subject, relation and object components of a fact
        U = tf.get_variable("U", (self.config.embed_size, self.vocab_size))
        b_p = tf.get_variable("b_p", (self.vocab_size,))

        reg = self.config.l2*tf.nn.l2_loss(U)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, reg)



        output = tf.matmul(rnn_output, U) + b_p

        return output

  def add_loss_op(self, output):

    """Adds loss ops to the computational graph.


    Args:
      output: A tensor of shape (batch_size, fact_embed_size)
    Returns:
      loss: A 0-d tensor (scalar)
    """
    gate_loss = 0

    for i, att in enumerate(self.attentions):
        #if i == self.rel_label_placeholder.get_shape()[1]: break
        labels = tf.gather(tf.transpose(self.rel_label_placeholder), 0)
        gate_loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(att, labels))

    loss = self.config.alpha*gate_loss + self.config.beta*tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(output, self.answer_placeholder))
    loss += tf.reduce_sum(tf.pack(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))
    return loss

  def get_predictions(self, output):
      preds = tf.nn.softmax(output)
      pred = tf.argmax(preds, 1)
      return pred
      
        

  def add_training_op(self, loss):
    """Sets up the training Ops.

    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train. See 

    https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

    for more information.

    Hint: Use tf.train.AdamOptimizer for this model.
          Calling optimizer.minimize() will return a train_op object.

    Args:
      loss: Loss tensor, from cross_entropy_loss.
    Returns:
      train_op: The Op for training.
    """
    opt = tf.train.AdamOptimizer(learning_rate=self.config.lr)
    gvs = opt.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_norm(grad, self.config.max_grad_val), var) for grad, var in gvs]
    noised_gvs = [(add_gradient_noise(grad), var) for grad, var in gvs]
    train_op = opt.apply_gradients(gvs)
    return train_op
  
  def __init__(self, config):

    self.config = config
    self.variables_to_save = {}
    self.load_data(debug=False)
    self.add_placeholders()
    self.rnn_outputs = self.add_model()
    self.output = self.add_projection(self.rnn_outputs)
    self.pred = self.get_predictions(self.output)
  
    self.calculate_loss = self.add_loss_op(self.output)
    self.train_step = self.add_training_op(self.calculate_loss)

  def get_question_representation(self, inputs):
    outputs, q_vec = rnn.rnn(self.drop_gru, inputs, dtype=np.float32, sequence_length=self.question_len_placeholder)
    return q_vec

  def get_input_representation(self, inputs):

    outputs, _ = rnn.rnn(self.drop_gru, inputs, dtype=np.float32, sequence_length=self.input_len_placeholder)
    # pick out gru outputs at the points specfied by input mask
    outputs = tf.pack(outputs)

    outputs = tf.split(1, self.config.batch_size, outputs)

    outputs = [tf.gather(out, tf.gather(self.input_mask_placeholder, i)) for i, out in enumerate(outputs)]


    fact_vecs = tf.concat(1, outputs)

    return fact_vecs



  def get_attention(self, q_vec, prev_memory, fact_vec):
    with tf.variable_scope("attention", reuse=True):

        b_1 = tf.get_variable("b_1")
        W_1 = tf.get_variable("W_1")

        W_2 = tf.get_variable("W_2")
        b_2 = tf.get_variable("b_2")

        #W_a = tf.get_variable("W_a")
        #b_a = tf.get_variable("b_a")

        features = [fact_vec, prev_memory, q_vec, fact_vec*prev_memory, fact_vec*q_vec, tf.abs(fact_vec - q_vec), tf.abs(fact_vec - prev_memory)]

        feature_vec = tf.concat(1, features)

        reg = self.config.l2*(tf.nn.l2_loss(W_1) + tf.nn.l2_loss(W_2))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, reg)


        attention = tf.matmul(tf.tanh(tf.matmul(feature_vec, W_1) + b_1), W_2) + b_2
        # normalize attention?
        
    return attention

  # generates an episode using the inner GRU using the current memory state and 
  def generate_episode(self, memory, q_vec, fact_vecs):


      fact_vecs_split = tf.split(0, self.max_mask_len, fact_vecs)
      fact_vecs_split = [tf.squeeze(f, squeeze_dims=[0]) for f in fact_vecs_split]

      attentions = [tf.squeeze(self.get_attention(q_vec, memory, fv), squeeze_dims=[1]) for fv in fact_vecs_split]
      
      attentions = tf.transpose(tf.pack(attentions))
      self.attentions.append(attentions)

      softs = tf.nn.softmax(attentions)
      softs = tf.transpose(softs)

      weighted_facts = tf.expand_dims(softs, 2)*fact_vecs

      episode = tf.reduce_sum(weighted_facts, reduction_indices=[0])
      
      return episode


  def add_model(self):
    """Creates the RNN LM model.

    In the space provided below, you need to implement the equations for the
    RNN LM model. Note that you may NOT use built in rnn_cell functions from
    tensorflow.

    Args:
      inputs: List of length num_steps, each of whose elements should be
              a tensor of shape (batch_size, embed_size).
    Returns:
      outputs: List of length num_steps, each of whose elements should be
               a tensor of shape (batch_size, hidden_size)
    """
    # add embedding split inputs so they can go into rnns
    questions, inputs = self.add_embedding()
     
    with tf.variable_scope("question"):
        print '==> get question representation'
        q_vec = self.get_question_representation(questions)
     

    with tf.variable_scope("input"):
        print '==> get input representation'
        fact_vecs = self.get_input_representation(inputs)

    self.attentions = []

    print '==> build episodic memory'

    with tf.variable_scope("memory"):

        # generate n_hops episodes
        prev_memory = q_vec

        for i in range(self.config.num_hops):
            # get a new episode
            print '==> generating episode', i
            episode = self.generate_episode(prev_memory, q_vec, fact_vecs)

            # do a GRU step to get the new memory
            _, prev_memory = rnn.rnn(self.drop_gru, [prev_memory, episode], dtype=np.float32)
            tf.get_variable_scope().reuse_variables()

        output = prev_memory

    return output


  def run_epoch(self, session, data, train_op=None, verbose=2, train=False):
    config = self.config
    dp = config.dropout
    if not train_op:
      train_op = tf.no_op()
      dp = 1
    total_steps = len(data[0]) / config.batch_size
    total_loss = []
    accuracy = 0
    
    p = np.random.permutation(len(data[0]))
    qp, ip, ql, il, im, a, r = data
    qp, ip, ql, il, im, a, r = qp[p], ip[p], ql[p], il[p], im[p], a[p], r[p] 




    for step in range(total_steps):
      feed = {self.question_placeholder: qp[step*config.batch_size:(step+1)*config.batch_size],
              self.input_placeholder: ip[step*config.batch_size:(step+1)*config.batch_size],
              self.question_len_placeholder: ql[step*config.batch_size:(step+1)*config.batch_size],
              self.input_len_placeholder: il[step*config.batch_size:(step+1)*config.batch_size],
              self.input_mask_placeholder: im[step*config.batch_size:(step+1)*config.batch_size],
              self.answer_placeholder: a[step*config.batch_size:(step+1)*config.batch_size],
              self.rel_label_placeholder: r[step*config.batch_size:(step+1)*config.batch_size],
              self.dropout_placeholder: dp}
      loss, pred, _ = session.run(
          [self.calculate_loss, self.pred, train_op], feed_dict=feed)

      answers = a[step*config.batch_size:(step+1)*config.batch_size]
      accuracy += np.sum(pred == answers)/float(len(answers))


      total_loss.append(loss)
      if verbose and step % verbose == 0:
          sys.stdout.write('\r{} / {} : loss = {}'.format(
              step, total_steps, np.mean(total_loss)))
          sys.stdout.flush()


    if verbose:
      sys.stdout.write('\r')

    print ''
    if self.config.beta > 0:
        print "accuracy:", accuracy/float(total_steps)
    
    return np.mean(total_loss)

def test_RNNLM():
  config = Config()

  # We create the training model and generative model
  with tf.variable_scope('RNNLM') as scope:

    model = DMN_QA_Model(config)
  init = tf.initialize_all_variables()
  saver = tf.train.Saver()

  

  for v in tf.trainable_variables():
      print v.get_shape()

  with tf.Session() as session:
    session.run(init)

    best_val_loss = float('inf')
    # get previous best val loss from file

    best_val_epoch = 0

    prev_epoch_loss = float('inf')
    
    if model.config.restore:
        print 'restoring weights'
        saver.restore(session, 'weights/mem' + str(model.config.babi_id) + 'beta=' + str(model.config.beta) + '.weights')

    if train_mode:
        for epoch in xrange(config.max_epochs):
          print 'Epoch {}'.format(epoch)
          start = time.time()
          ###
          train_loss = model.run_epoch(
              session, model.train,
              train_op=model.train_step, train=True)
          valid_loss = model.run_epoch(session, model.valid)
          print 'Training loss: {}'.format(train_loss)
          print 'Validation loss: {}'.format(valid_loss)

          if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_val_epoch = epoch
            saver.save(session, 'weights/mem' + str(model.config.babi_id) + 'beta=' + str(model.config.beta) + '.weights')

          # anneal
          if train_loss>prev_epoch_loss*model.config.anneal_threshold:
              model.config.lr/=model.config.anneal_by
              print 'annealed lr to %f'%model.config.lr

          prev_epoch_loss = train_loss
          

          if epoch - best_val_epoch > config.early_stopping:
            break
          print 'Total time: {}'.format(time.time() - start)
    else:
          test_loss = model.run_epoch(session, model.test)



if __name__ == "__main__":
    test_RNNLM()
