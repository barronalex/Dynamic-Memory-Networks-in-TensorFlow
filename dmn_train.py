import tensorflow as tf
import numpy as np

import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--babi_task_id", help="specify babi task 1-20 (default=1)")
parser.add_argument("-r", "--restore", help="restore previously trained weights (default=false)")
parser.add_argument("-s", "--strong_supervision", help="use labelled supporting facts (default=false)")
parser.add_argument("-t", "--dmn_type", help="specify type of dmn (default=original)")
parser.add_argument("-l", "--l2_loss", type=float, help="specify l2 loss constant")

args = parser.parse_args()

dmn_type = args.dmn_type if args.dmn_type is not None else "plus"

if dmn_type == "original":
    from dmn_original import Config
    config = Config()
elif dmn_type == "plus":
    from dmn_plus import Config
    config = Config()
else:
    raise NotImplementedError(dmn_type + ' DMN type is not currently implemented')

if args.babi_task_id is not None:
    config.babi_id = args.babi_task_id

config.babi_id = args.babi_task_id if args.babi_task_id is not None else 1
config.l2 = args.l2_loss if args.l2_loss is not None else 0.001
config.strong_supervision = args.strong_supervision if args.strong_supervision is not None else False

print 'Training DMN ' + dmn_type + ' on babi task', config.babi_id

# create model
with tf.variable_scope('DMN') as scope:
    if dmn_type == "original":
        from dmn_original import DMN
        model = DMN(config)
    elif dmn_type == "plus":
        from dmn_plus import DMN_PLUS
        model = DMN_PLUS(config)


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
