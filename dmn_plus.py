import sys
import time

import numpy as np
import pandas as pd
import cPickle
import tables
from copy import deepcopy

import tensorflow as tf

import babi_input

class Config(object):
    """Holds model hyperparams and data information."""

    batch_size = 100
    embed_size = 80
    hidden_size = 80

    max_epochs = 256
    early_stopping = 20

    dropout = 0.9
    lr = 0.001
    l2 = 0.001

    cap_grads = False
    max_grad_val = 10
    noisy_grads = False

    word2vec_init = False
    embedding_init = 1.7320508 # root 3

    # set to zero with strong supervision to only train gates
    strong_supervision = False
    beta = 1

    drop_grus = False

    anneal_threshold = 1000
    anneal_by = 1.5

    num_hops = 3
    num_attention_features = 4

    max_allowed_inputs = 130
    num_train = 9000

    floatX = np.float32

    babi_id = "1"
    babi_test_id = ""

    train_mode = True

def _add_gradient_noise(t, stddev=1e-3, name=None):
    """Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].
    The input Tensor `t` should be a gradient.
    The output will be `t` + gaussian noise.
    0.001 was said to be a good fixed value for memory networks [2]."""
    with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)

# from https://github.com/domluna/memn2n
def _position_encoding(sentence_size, embedding_size):
    """Position encoding described in section 4.1 in "End to End Memory Networks" (http://arxiv.org/pdf/1503.08895v5.pdf)"""
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)

    # TODO fix positional encoding so that it varies according to sentence lengths

def _xavier_weight_init():
    """Xavier initializer for all variables except embeddings as desribed in [1]"""
    def _xavier_initializer(shape, **kwargs):
        eps = np.sqrt(6) / np.sqrt(np.sum(shape))
        out = tf.random_uniform(shape, minval=-eps, maxval=eps)
        return out
    return _xavier_initializer

# from https://danijar.com/variable-sequence-lengths-in-tensorflow/
# used only for custom attention GRU as TF handles this with the sequence length param for normal RNNs
def _last_relevant(output, length):
    """Finds the output at the end of each input"""
    batch_size = int(output.get_shape()[0])
    max_length = int(output.get_shape()[1])
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant
    

class DMN_PLUS(object):

    def load_data(self, debug=False):
        """Loads train/valid/test data and sentence encoding"""
        if self.config.train_mode:
            self.train, self.valid, self.word_embedding, self.max_q_len, self.max_input_len, self.max_sen_len, self.num_supporting_facts, self.vocab_size = babi_input.load_babi(self.config, split_sentences=True)
        else:
            self.test, self.word_embedding, self.max_q_len, self.max_input_len, self.max_sen_len, self.num_supporting_facts, self.vocab_size = babi_input.load_babi(self.config, split_sentences=True)
        self.encoding = _position_encoding(self.max_sen_len, self.config.embed_size)

    def add_placeholders(self):
        """add data placeholder to graph"""
        self.question_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.max_q_len))
        self.input_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.max_input_len, self.max_sen_len))

        self.question_len_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size,))
        self.input_len_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size,))

        self.answer_placeholder = tf.placeholder(tf.int64, shape=(self.config.batch_size,))

        self.rel_label_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.num_supporting_facts))

        self.dropout_placeholder = tf.placeholder(tf.float32)

    def add_reused_variables(self):
        """Adds trainable variables which are later reused""" 
        gru_cell = tf.nn.rnn_cell.GRUCell(self.config.hidden_size)

        # apply droput to grus if flag set
        if self.config.drop_grus:
            self.gru_cell = tf.nn.rnn_cell.DropoutWrapper(gru_cell, input_keep_prob=self.dropout_placeholder, output_keep_prob=self.dropout_placeholder)
        else:
            self.gru_cell = gru_cell

        with tf.variable_scope("memory/attention", initializer=_xavier_weight_init()):
            b_1 = tf.get_variable("bias_1", (self.config.embed_size,))
            W_1 = tf.get_variable("W_1", (self.config.embed_size*self.config.num_attention_features, self.config.embed_size))

            W_2 = tf.get_variable("W_2", (self.config.embed_size, 1))
            b_2 = tf.get_variable("bias_2", 1)

        with tf.variable_scope("memory/attention_gru", initializer=_xavier_weight_init()):
            Wr = tf.get_variable("Wr", (self.config.embed_size, self.config.hidden_size))
            Ur = tf.get_variable("Ur", (self.config.hidden_size, self.config.hidden_size))
            br = tf.get_variable("bias_r", (1, self.config.hidden_size))

            W = tf.get_variable("W", (self.config.embed_size, self.config.hidden_size))
            U = tf.get_variable("U", (self.config.hidden_size, self.config.hidden_size))
            bh = tf.get_variable("bias_h", (1, self.config.hidden_size))

    def get_predictions(self, output):
        """Get answer predictions from output"""
        preds = tf.nn.softmax(output)
        pred = tf.argmax(preds, 1)
        return pred
      
    def add_loss_op(self, output):
        """Calculate loss"""
        # optional strong supervision of attention with supporting facts
        gate_loss = 0
        if self.config.strong_supervision:
            for i, att in enumerate(self.attentions):
                labels = tf.gather(tf.transpose(self.rel_label_placeholder), 0)
                gate_loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(att, labels))

        loss = self.config.beta*tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(output, self.answer_placeholder)) + gate_loss

        # add l2 regularization for all variables except biases
        for v in tf.trainable_variables():
            if not 'bias' in v.name.lower():
                loss += self.config.l2*tf.nn.l2_loss(v)

        tf.scalar_summary('loss', loss)

        return loss
        
    def add_training_op(self, loss):
        """Calculate and apply gradients"""
        opt = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        gvs = opt.compute_gradients(loss)

        # optionally cap and noise gradients to regularize
        if self.config.cap_grads:
            gvs = [(tf.clip_by_norm(grad, self.config.max_grad_val), var) for grad, var in gvs]
        if self.config.noisy_grads:
            gvs = [(_add_gradient_noise(grad), var) for grad, var in gvs]

        train_op = opt.apply_gradients(gvs)
        return train_op
  

    def get_question_representation(self, embeddings):
        """Get question vectors via embedding and GRU"""
        questions = tf.nn.embedding_lookup(embeddings, self.question_placeholder)

        questions = tf.split(1, self.max_q_len, questions)
        questions = [tf.squeeze(q, squeeze_dims=[1]) for q in questions]

        _, q_vec = tf.nn.rnn(self.gru_cell, questions, dtype=np.float32, sequence_length=self.question_len_placeholder)
        
        return q_vec

    def get_input_representation(self, embeddings):
        """Get fact (sentence) vectors via embedding, positional encoding and bi-directional GRU"""
        # get word vectors from embedding
        inputs = tf.nn.embedding_lookup(embeddings, self.input_placeholder)

        # use encoding to get sentence representation
        inputs = tf.reduce_sum(inputs * self.encoding, 2)

        inputs = tf.split(1, self.max_input_len, inputs)
        inputs = [tf.squeeze(i, squeeze_dims=[1]) for i in inputs]

        outputs, _, _ = tf.nn.bidirectional_rnn(self.gru_cell, self.gru_cell, inputs, dtype=np.float32, sequence_length=self.input_len_placeholder)

        # f<-> = f-> + f<-
        fact_vecs = [tf.reduce_sum(tf.pack(tf.split(1, 2, out)), 0) for out in outputs]

        # apply dropout
        fact_vecs = [tf.nn.dropout(fv, self.dropout_placeholder) for fv in fact_vecs]

        return fact_vecs

    def get_attention(self, q_vec, prev_memory, fact_vec):
        """Use question vector and previous memory to create scalar attention for current fact"""
        with tf.variable_scope("attention", reuse=True, initializer=_xavier_weight_init()):

            W_1 = tf.get_variable("W_1")
            b_1 = tf.get_variable("bias_1")

            W_2 = tf.get_variable("W_2")
            b_2 = tf.get_variable("bias_2")

            features = [fact_vec*q_vec, fact_vec*prev_memory, tf.abs(fact_vec - q_vec), tf.abs(fact_vec - prev_memory)]

            feature_vec = tf.concat(1, features)

            attention = tf.matmul(tf.tanh(tf.matmul(feature_vec, W_1) + b_1), W_2) + b_2
            # normalize attention?
            
        return attention

    def _attention_GRU_step(self, rnn_input, h, g):
        """Implement attention GRU as described by https://arxiv.org/abs/1603.01417"""
        with tf.variable_scope("attention_gru", reuse=True, initializer=_xavier_weight_init()):

            Wr = tf.get_variable("Wr")
            Ur = tf.get_variable("Ur")
            br = tf.get_variable("bias_r")

            W = tf.get_variable("W")
            U = tf.get_variable("U")
            bh = tf.get_variable("bias_h")

            r = tf.sigmoid(tf.matmul(rnn_input, Wr) + tf.matmul(h, Ur) + br)
            h_hat = tf.tanh(tf.matmul(rnn_input, W) + r*tf.matmul(h, U) + bh)
            rnn_output = g*h_hat + (1-g)*h

            return rnn_output

    def generate_episode(self, memory, q_vec, fact_vecs):
        """Generate episode by applying attention to current fact vectors through a modified GRU"""

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

        # extract gru outputs at proper index according to input_lens
        gru_outputs = tf.pack(gru_outputs)
        gru_outputs = tf.transpose(gru_outputs, perm=[1,0,2])
        episode = _last_relevant(gru_outputs, self.input_len_placeholder)

        return episode

    def add_answer_module(self, rnn_output, q_vec):
        """Linear softmax answer module"""
        with tf.variable_scope("answer"):
            U = tf.get_variable("U", (2*self.config.embed_size, self.vocab_size))
            b_p = tf.get_variable("bias_p", (self.vocab_size,))

            output = tf.matmul(tf.concat(1, [rnn_output, q_vec]), U) + b_p

            output = tf.nn.dropout(output, self.dropout_placeholder)

            return output

    def inference(self):
        """Performs inference on the DMN model"""

        # set up embedding
        embeddings = tf.Variable(self.word_embedding.astype(np.float32), name="Embedding")
         
        # input fusion module
        with tf.variable_scope("question", initializer=_xavier_weight_init()):
            print '==> get question representation'
            q_vec = self.get_question_representation(embeddings)
         

        with tf.variable_scope("input", initializer=_xavier_weight_init()):
            print '==> get input representation'
            fact_vecs = self.get_input_representation(embeddings)

        # keep track of attentions for possible strong supervision
        self.attentions = []

        # memory module
        with tf.variable_scope("memory", initializer=_xavier_weight_init()):
            print '==> build episodic memory'

            # generate n_hops episodes
            prev_memory = q_vec

            for i in range(self.config.num_hops):
                # get a new episode
                print '==> generating episode', i
                episode = self.generate_episode(prev_memory, q_vec, fact_vecs)

                # untied weights for memory update
                Wt = tf.get_variable("W_t"+ str(i), (2*self.config.hidden_size+self.config.embed_size, self.config.hidden_size))
                bt = tf.get_variable("bias_t"+ str(i), (self.config.hidden_size,))

                # update memory with Relu
                prev_memory = tf.nn.relu(tf.matmul(tf.concat(1, [prev_memory, episode, q_vec]), Wt) + bt)

            output = prev_memory

        # pass memory module output through linear answer module
        output = self.add_answer_module(output, q_vec)

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

        return np.mean(total_loss), accuracy/float(total_steps)


    def __init__(self, config):

        self.config = config
        self.variables_to_save = {}
        self.load_data(debug=False)
        self.add_placeholders()
        self.add_reused_variables()
        self.output = self.inference()
        self.pred = self.get_predictions(self.output)
        self.calculate_loss = self.add_loss_op(self.output)
        self.train_step = self.add_training_op(self.calculate_loss)
        self.merged = tf.merge_all_summaries()

