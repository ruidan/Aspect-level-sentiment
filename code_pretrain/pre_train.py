import argparse
import logging
import numpy as np
from time import time

logging.basicConfig(
                    # filename='out.log',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


###############################################################################################################################
## Parse arguments
#

parser = argparse.ArgumentParser()
parser.add_argument("--domain", dest="domain", type=str, metavar='<str>', required=True, help="The name of the domain (electronics_large|yelp_large)")
parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=10000, help="Vocab size. '0' means no limit (default=10000)")
parser.add_argument("--n-class", dest="n_class", type=int, metavar='<int>', default=3, help="The number of ouput classes")
parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=300, help="Embeddings dimension (default=300)")
parser.add_argument("-r", "--rnndim", dest="rnn_dim", type=int, metavar='<int>', default=300, help="RNN dimension. '0' means no RNN layer (default=300)")
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=10, help="Number of epochs (default=10)")
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=50, help="Batch size (default=50)")
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1234, help="Random seed (default=1234)")
parser.add_argument("-p", dest="percetage", type=float, metavar='<float>', default=1.0, help="The percetage of data used for training")

args = parser.parse_args()
if args.seed > 0:
    np.random.seed(args.seed)

###############################################################################################################################
## Prepare data
#

logger.info('  Preparing data')

from read import prepare_data
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical

vocab, data_list, label_list, overall_maxlen = prepare_data(args.domain, args.vocab_size)

rand = np.arange(len(data_list))
np.random.shuffle(rand)

data_list = data_list[rand]
label_list = to_categorical(label_list)[rand]
data_size = len(data_list)

dev_x = data_list[0:1000]
dev_y = label_list[0:1000]
train_x = data_list[1000:int(data_size*args.percetage)]
train_y = label_list[1000:int(data_size*args.percetage)]

maxlen = np.max([len(d) for d in dev_x])
dev_x = sequence.pad_sequences(dev_x, maxlen)

import operator
vocab_list = [x for (x, _) in sorted(vocab.items(), key=operator.itemgetter(1))]


def batch_generator(data1, data2, batch_size):
    len_ = len(data1)
    while True:
        indices = np.random.choice(len_, batch_size)
        x = data1[indices]
        y = data2[indices]

        maxlen = np.max([len(d) for d in x])
        x = sequence.pad_sequences(x, maxlen)
        yield x, y


###############################################################################################################################
## Building model

from pre_model import create_model
import keras.optimizers as opt

logger.info('  Building model')

model = create_model(args, vocab, args.n_class)
optimizer = opt.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06, clipnorm=10, clipvalue=0)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])


###############################################################################################################################
## Training
#
import pickle  
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def save_we(vocab, weights):
    word_emb = {}
    for i, j in zip(vocab, weights):
        word_emb[i] = j
    if args.domain == 'electronics_large':
        save_obj(word_emb, '../pretrained_weights/word_emb_lt'+str(args.percetage))
    else:
        save_obj(word_emb, '../pretrained_weights/word_emb_res'+str(args.percetage))


from tqdm import tqdm
logger.info('--------------------------------------------------------------------------------------------------------------------------')

train_gen = batch_generator(train_x, train_y, batch_size=args.batch_size)
batches_per_epoch = len(train_x) / args.batch_size

best_acc = 0
best_loss = 100
for ii in xrange(args.epochs):
    t0 = time()
    loss, metric = 0., 0.

    for b in tqdm(xrange(batches_per_epoch)):
        batch_x,  batch_y = train_gen.next()
        loss_, metric_ = model.train_on_batch([batch_x], batch_y)
        loss += loss_ / batches_per_epoch
        metric += metric_ / batches_per_epoch

    tr_time = time() - t0

    dev_loss, dev_metric = model.evaluate([dev_x], dev_y, batch_size=args.batch_size)

    logger.info('Epoch %d, train: %is' % (ii, tr_time))
    logger.info('[Train] loss: %.4f, metric: %.4f' % (loss, metric))
    logger.info('[Dev] loss: %.4f, metric: %.4f' % (dev_loss, dev_metric))

    if dev_metric > best_acc:
        best_acc = dev_metric
        word_emb = model.get_layer('word_emb').get_weights()[0]
        lstm_weights = model.get_layer('lstm').get_weights()
        dense_weights = model.get_layer('dense').get_weights()

        save_we(vocab_list, word_emb)

        if args.domain == 'electronics_large':
            save_obj(lstm_weights, '../pretrained_weights/lstm_weights_lt'+str(args.percetage))
            save_obj(dense_weights, '../pretrained_weights/dense_weights_lt'+str(args.percetage))
        else:
            save_obj(lstm_weights, '../pretrained_weights/lstm_weights_res'+str(args.percetage))
            save_obj(dense_weights, '../pretrained_weights/dense_weights_res'+str(args.percetage))

        print '------- Saved Weights -------'



