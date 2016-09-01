import sys
import time

import numpy as np
import pandas as pd
import cPickle
import tables
from copy import deepcopy

import tensorflow as tf

import babi_input
from model import DMN
from xavier_initializer import xavier_weight_init

floatX = np.float32
ROOT3 = 1.7320508

word2vec_init = False


class Config(object):
    """Holds model hyperparams and data information."""

    # set to zero with strong supervision to only train gates
    beta = 1

    batch_size = 100
    embed_size = 80
    hidden_size = 80

    word2vec_init = False
    embedding_init = 1.7320508 # root 3

    max_epochs = 256
    early_stopping = 20

    dropout = 0.9
    lr = 0.001
    l2 = 0.001

    drop_grus = True
    num_gru_layers = 1

    anneal_threshold = 1000
    anneal_by = 1.5

    num_hops = 3
    num_attention_features = 4
    max_grad_val = 10
    num_train = 9000

    floatX = np.float32

    babi_id = "1"
    babi_test_id = ""

    train_mode = True

def _add_gradient_noise(t, stddev=1e-3, name=None):
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

# from https://github.com/domluna/memn2n
def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 in http://arxiv.org/pdf/1503.08895v5.pdf
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)

    # TODO fix positional encoding so that it varies according to sentence size

class DMN_PLUS(DMN):

    def load_data(self, debug=False):
        """Loads starter word-vectors and train/dev/test data."""
        if self.config.train_mode:
            self.train, self.valid, self.word_embedding, self.max_q_len, self.max_input_len, self.max_sen_len, self.num_supporting_facts, self.vocab_size = babi_input.load_babi(self.config, split_sentences=True)
        else:
            self.test, self.word_embedding, self.max_q_len, self.max_input_len, self.max_sen_len, self.num_supporting_facts, self.vocab_size = babi_input.load_babi(self.config, split_sentences=True)
        self.encoding = position_encoding(self.max_sen_len, self.config.embed_size)

    def add_placeholders(self):

        self.question_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.max_q_len))
        self.input_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.max_input_len, self.max_sen_len))

        self.question_len_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size,))
        self.input_len_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size,))

        self.answer_placeholder = tf.placeholder(tf.int64, shape=(self.config.batch_size,))

        self.rel_label_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.num_supporting_facts))

        self.dropout_placeholder = tf.placeholder(tf.float32)

        gru_cell = tf.nn.rnn_cell.GRUCell(self.config.hidden_size)

        # apply droput to grus if flag set
        if self.config.drop_grus:
            self.drop_gru = tf.nn.rnn_cell.DropoutWrapper(gru_cell, input_keep_prob=self.dropout_placeholder, output_keep_prob=self.dropout_placeholder)
        else:
            self.drop_gru = gru_cell

        multi_gru = tf.nn.rnn_cell.MultiRNNCell([gru_cell] * self.config.num_gru_layers)
        self.dropm_gru = tf.nn.rnn_cell.DropoutWrapper(multi_gru, input_keep_prob=self.dropout_placeholder, output_keep_prob=self.dropout_placeholder)

        with tf.variable_scope("memory/attention", initializer=xavier_weight_init()):
            b_1 = tf.get_variable("b_1", (self.config.embed_size,))
            W_1 = tf.get_variable("W_1", (self.config.embed_size*self.config.num_attention_features, self.config.embed_size))

            W_2 = tf.get_variable("W_2", (self.config.embed_size, 1))
            b_2 = tf.get_variable("b_2", 1)

        with tf.variable_scope("memory/attention_gru", initializer=xavier_weight_init()):
            Wr = tf.get_variable("Wr", (self.config.embed_size, self.config.hidden_size))
            Ur = tf.get_variable("Ur", (self.config.hidden_size, self.config.hidden_size))
            br = tf.get_variable("br", (1, self.config.hidden_size))

            W = tf.get_variable("W", (self.config.embed_size, self.config.hidden_size))
            U = tf.get_variable("U", (self.config.hidden_size, self.config.hidden_size))
            bh = tf.get_variable("bh", (1, self.config.hidden_size))

    def get_predictions(self, output):
        preds = tf.nn.softmax(output)
        pred = tf.argmax(preds, 1)
        return pred
      
    def add_loss_op(self, output):

        """Adds loss ops to the computational graph.


        Args:
          output: A tensor of shape (batch_size, fact_embed_size)
        Returns:
          loss: A 0-d tensor (scalar)
        """

        gate_loss = 0
        if self.config.strong_supervision:
            for i, att in enumerate(self.attentions):
                #if i == self.rel_label_placeholder.get_shape()[1]: break
                labels = tf.gather(tf.transpose(self.rel_label_placeholder), 0)
                gate_loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(att, labels))

        loss = self.config.beta*tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(output, self.answer_placeholder)) + gate_loss

        loss += tf.reduce_sum(tf.pack(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))

        tf.scalar_summary('loss', loss)

        return loss
        
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
        noised_gvs = [(_add_gradient_noise(grad), var) for grad, var in gvs]
        train_op = opt.apply_gradients(gvs)
        return train_op
  

    def get_question_representation(self, embeddings):

        questions = tf.nn.embedding_lookup(embeddings, self.question_placeholder)

        questions = tf.nn.dropout(questions, self.dropout_placeholder)

        questions = tf.split(1, self.max_q_len, questions)
        questions = [tf.squeeze(q, squeeze_dims=[1]) for q in questions]

        _, q_vec = tf.nn.rnn(self.drop_gru, questions, dtype=np.float32, sequence_length=self.question_len_placeholder)
        return q_vec

    def get_input_representation(self, embeddings):

        inputs = tf.nn.embedding_lookup(embeddings, self.input_placeholder)

        print inputs.get_shape()

        # use encoding to get sentence representation
        inputs = tf.reduce_sum(inputs * self.encoding, 2)

        print inputs.get_shape()
        
        inputs = tf.nn.dropout(inputs, self.dropout_placeholder)

        inputs = tf.split(1, self.max_input_len, inputs)
        inputs = [tf.squeeze(i, squeeze_dims=[1]) for i in inputs]

        outputs, _, _ = tf.nn.bidirectional_rnn(self.drop_gru, self.drop_gru, inputs, dtype=np.float32, sequence_length=self.input_len_placeholder)

        # f<-> = f-> + f<-
        fact_vecs = [tf.reduce_sum(tf.pack(tf.split(1, 2, out)), 0) for out in outputs]

        print fact_vecs[0].get_shape()
        print len(fact_vecs)

        return fact_vecs

    def get_attention(self, q_vec, prev_memory, fact_vec):
        with tf.variable_scope("attention", reuse=True, initializer=xavier_weight_init()):

            b_1 = tf.get_variable("b_1")
            W_1 = tf.get_variable("W_1")

            W_2 = tf.get_variable("W_2")
            b_2 = tf.get_variable("b_2")

            features = [fact_vec*q_vec, fact_vec*prev_memory, tf.abs(fact_vec - q_vec), tf.abs(fact_vec - prev_memory)]

            feature_vec = tf.concat(1, features)

            reg = self.config.l2*(tf.nn.l2_loss(W_1) + tf.nn.l2_loss(W_2))
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, reg)


            attention = tf.matmul(tf.tanh(tf.matmul(feature_vec, W_1) + b_1), W_2) + b_2
            # normalize attention?
            
        return attention

    def _attention_GRU_step(self, rnn_input, h, g):

        with tf.variable_scope("attention_gru", reuse=True, initializer=xavier_weight_init()):

            Wr = tf.get_variable("Wr")
            Ur = tf.get_variable("Ur")
            br = tf.get_variable("br")

            W = tf.get_variable("W")
            U = tf.get_variable("U")
            bh = tf.get_variable("bh")

            r = tf.sigmoid(tf.matmul(rnn_input, Wr) + tf.matmul(h, Ur) + br)
            h_hat = tf.tanh(tf.matmul(rnn_input, W) + r*tf.matmul(h, U) + bh)
            rnn_output = g*h_hat + (1-g)*h

            return rnn_output

    # generates an episode using the inner GRU using the current memory state and 
    def generate_episode(self, memory, q_vec, fact_vecs):


        attentions = [tf.squeeze(self.get_attention(q_vec, memory, fv), squeeze_dims=[1]) for fv in fact_vecs]

        attentions = tf.transpose(tf.pack(attentions))
        self.attentions.append(attentions)

        softs = tf.nn.softmax(attentions)
        softs = tf.split(1, self.max_input_len, softs)
        
        gru_outputs = []

        # set initial state to zero
        h = tf.zeros((self.config.batch_size, self.config.hidden_size))

        # use attention gru
        for i, fv in enumerate(fact_vecs):
            h = self._attention_GRU_step(fv, h, softs[i])
            gru_outputs.append(h)

        # episode is final gru state
        episode = h

        # TODO extract gru outputs at proper index according to input_lens

        return episode

    def add_answer_module(self, rnn_output):
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

    def inference(self):
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

        # set up embedding
        embeddings = tf.Variable(self.word_embedding.astype(np.float32), name="Embedding")
         
        # input fusion module
        with tf.variable_scope("question", initializer=xavier_weight_init()):
            print '==> get question representation'
            q_vec = self.get_question_representation(embeddings)
         

        with tf.variable_scope("input", initializer=xavier_weight_init()):
            print '==> get input representation'
            fact_vecs = self.get_input_representation(embeddings)

        self.attentions = []

        with tf.variable_scope("memory", initializer=xavier_weight_init()):
            print '==> build episodic memory'

            # generate n_hops episodes
            prev_memory = q_vec

            for i in range(self.config.num_hops):
                # get a new episode
                print '==> generating episode', i
                episode = self.generate_episode(prev_memory, q_vec, fact_vecs)

                # do a GRU step to get the new memory
                _, prev_memory = tf.nn.rnn(self.drop_gru, [prev_memory, episode], dtype=np.float32)
                tf.get_variable_scope().reuse_variables()

            output = prev_memory

        output = self.add_answer_module(output)

        return output


    def run_epoch(self, session, data, num_epoch=0, train_writer=None, train_op=None, verbose=2, train=False):
        config = self.config
        dp = config.dropout
        if not train_op:
            train_op = tf.no_op()
            dp = 1
        total_steps = len(data[0]) / config.batch_size
        total_loss = []
        accuracy = 0
        
        # shuffle data
        p = np.random.permutation(len(data[0]))
        qp, ip, ql, il, im, a, r = data
        qp, ip, ql, il, im, a, r = qp[p], ip[p], ql[p], il[p], im[p], a[p], r[p] 

        for step in range(total_steps):
            index = range(step*config.batch_size,(step+1)*config.batch_size)
            feed = {self.question_placeholder: qp[index],
                  self.input_placeholder: ip[index],
                  self.question_len_placeholder: ql[index],
                  self.input_len_placeholder: il[index],
                  self.answer_placeholder: a[index],
                  self.rel_label_placeholder: r[index],
                  self.dropout_placeholder: dp}
            loss, pred, summary, _ = session.run(
              [self.calculate_loss, self.pred, self.merged, train_op], feed_dict=feed)

            if train_writer is not None:
                train_writer.add_summary(summary, num_epoch*total_steps + step)

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


    def __init__(self, config):

        self.config = config
        self.variables_to_save = {}
        self.load_data(debug=False)
        self.add_placeholders()
        self.output = self.inference()
        self.pred = self.get_predictions(self.output)
        self.calculate_loss = self.add_loss_op(self.output)
        self.train_step = self.add_training_op(self.calculate_loss)
        self.merged = tf.merge_all_summaries()

