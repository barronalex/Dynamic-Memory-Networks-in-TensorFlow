import tensorflow as tf
import numpy as np

from dmn import DMN
from dmn import Config

from dmn_plus import DMN_PLUS
from dmn_plus import Config as Config_plus

import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--babi_task_id", help="specify babi task 1-20 (default=1)")
parser.add_argument("-t", "--dmn_type", help="specify type of dmn (default=original)")
args = parser.parse_args()

dmn_type = args.dmn_type if args.dmn_type is not None else "plus"

if dmn_type == "original":
    config = Config()
elif dmn_type == "plus":
    config = Config_plus()
else:
    raise NotImplementedError(dmn_type + ' DMN type is not currently implemented')

config.train_mode = False

if args.babi_task_id is not None:
    config.babi_id = args.babi_task_id

config.strong_supervision = False

print 'Testing DMN on babi task', config.babi_id

# create model
with tf.variable_scope('DMN') as scope:
    if dmn_type == "original":
        model = DMN(config)
    elif dmn_type == "plus":
        model = DMN_PLUS(config)

print '==> initializing variables'
init = tf.initialize_all_variables()
saver = tf.train.Saver()

with tf.Session() as session:
    session.run(init)

    print '==> restoring weights'
    saver.restore(session, 'weights/mem' + str(model.config.babi_id) + 'beta=' + str(model.config.beta) + '.weights')

    print '==> running DMN'
    test_loss = model.run_epoch(session, model.test)
