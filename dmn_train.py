import tensorflow as tf
import numpy as np

from dmn import DMN
from dmn import Config

import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--babi_task_id", help="specify babi task 1-20 (default=1)")
parser.add_argument("-r", "--restore", help="restore previously trained weights (default=false)")
parser.add_argument("-s", "--strong-supervision", help="use labelled supporting facts (default=false)")

args = parser.parse_args()

config = Config()

if args.babi_task_id is not None:
    config.babi_id = args.babi_task_id

print 'Training DMN on babi task', config.babi_id

config.strong_supervision = args.strong_supervision if args.strong_supervision is not None else False

# create model
with tf.variable_scope('RNNLM') as scope:
    model = DMN(config)

print '==> initializing variables'
init = tf.initialize_all_variables()
saver = tf.train.Saver()


with tf.Session() as session:

    train_writer = tf.train.SummaryWriter('summaries/train', session.graph)

    session.run(init)

    best_val_loss = float('inf')
    best_val_epoch = 0
    prev_epoch_loss = float('inf')

    if args.restore:
        print '==> restoring weights'
        saver.restore(session, 'weights/mem' + str(model.config.babi_id) + 'beta=' + str(model.config.beta) + '.weights')

    print '==> starting training'
    for epoch in xrange(config.max_epochs):
        print 'Epoch {}'.format(epoch)
        start = time.time()
        ###
        train_loss = model.run_epoch(
          session, model.train, epoch, train_writer,
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
