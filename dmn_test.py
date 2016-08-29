import tensorflow as tf
import numpy as np

from dmn import DMN
from dmn import Config

import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--babi_task_id", help="specify babi task 1-20 (default=1)")
args = parser.parse_args()

config = Config()

if args.babi_task_id is not None:
    config.babi_id = args.babi_task_id

config.strong_supervision = False

print 'Testing DMN on babi task', config.babi_id

# create model
with tf.variable_scope('RNNLM') as scope:
    model = DMN(config)

print '==> initializing variables'
init = tf.initialize_all_variables()
saver = tf.train.Saver()

with tf.Session() as session:
    session.run(init)

    print '==> restoring weights'
    saver.restore(session, 'weights/mem' + str(model.config.babi_id) + 'beta=' + str(model.config.beta) + '.weights')

    print '==> running DMN'
    test_loss = model.run_epoch(session, model.test)
