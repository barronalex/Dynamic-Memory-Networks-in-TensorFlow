import tensorflow as tf
import numpy as np

from dmn import DMN

import time

train_mode = True

class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    beta = 1
    alpha = 0

    restore = False

    # fix batch size
    batch_size = 100
    embed_size = 80
    hidden_size = 80
    max_epochs = 256
    early_stopping = 20
    dropout = 0.9
    drop_grus = True
    lr = 0.001
    l2 = 0.001
    anneal_threshold = 1
    anneal_by = 1.5
    num_hops = 3
    num_attention_features = 7
    max_grad_val = 10
    num_gru_layers = 1
    num_train = 9000

    babi_id = "2"
    babi_test_id = ""

config = Config()

# We create the training model and generative model
with tf.variable_scope('RNNLM') as scope:
    model = DMN(config)

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
