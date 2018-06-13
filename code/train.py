import argparse
import logging
import numpy as np
from time import time
import utils as U
import reader as dataset

logging.basicConfig(
                    # filename='transfer.log',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


###############################################################################################################################
## Parse arguments
#

parser = argparse.ArgumentParser()
parser.add_argument("-u", "--rec-unit", dest="recurrent_unit", type=str, metavar='<str>', default='lstm', help="Recurrent unit type (lstm|gru|simple) (default=lstm)")
parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='rmsprop', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=rmsprop)")
parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=300, help="Embeddings dimension (default=300)")
parser.add_argument("-r", "--rnndim", dest="rnn_dim", type=int, metavar='<int>', default=300, help="RNN dimension. (default=300)")
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=32, help="Batch size (default=32)")
parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=10000, help="Vocab size. '0' means no limit (default=10000)")
parser.add_argument("--dropout", dest="dropout_prob", type=float, metavar='<float>', default=0.5, help="The dropout probability of output layer. (default=0.5)")
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=15, help="Number of epochs (default=15)")
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1234, help="Random seed (default=1234)")
parser.add_argument("--domain", dest="domain", type=str, metavar='<str>', default='res', help="domain of the corpus (res|lt|res_15|res_16)")
parser.add_argument("--dropout-W", dest="dropout_W", type=float, metavar='<float>', default=0.5, help="The dropout of input to RNN")
parser.add_argument("--dropout-U", dest="dropout_U", type=float, metavar='<float>', default=0.1, help="The dropout of recurrent of RNN")
parser.add_argument("--alpha", dest="alpha", type=float, metavar='<float>', default=0.1, help="The weight of the doc-level training objective (\lambda in the paper)")
parser.add_argument("--is-pretrain", dest="is_pretrain", type=int, metavar='<int>', default=1, help="Whether to used pretrained weights")
parser.add_argument("-p", dest="percetage", type=float, metavar='<float>', default=1.0, help="The percetage of document data used for training")

args = parser.parse_args()

U.print_args(args)

assert args.algorithm in {'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adam', 'adamax'}
assert args.recurrent_unit in {'lstm', 'gru', 'simple'}
assert args.domain in {'res', 'lt', 'res_15', 'res_16'}

if args.seed > 0:
    np.random.seed(args.seed)


###############################################################################################################################
## Prepare data
#

from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical


train_x, train_y, train_aspect, test_x, test_y, test_aspect, \
    vocab, overal_maxlen, overal_maxlen_aspect, \
    pretrain_data, pretrain_label, pretrain_maxlen = dataset.prepare_data(args.domain, args.vocab_size)

# Pad aspect sentences sequences for mini-batch processing
train_x = sequence.pad_sequences(train_x, maxlen=overal_maxlen)
test_x = sequence.pad_sequences(test_x, maxlen=overal_maxlen)
train_aspect = sequence.pad_sequences(train_aspect, maxlen=overal_maxlen_aspect)
test_aspect = sequence.pad_sequences(test_aspect, maxlen=overal_maxlen_aspect)

# convert y to categorical labels
train_y = to_categorical(train_y, 3)
test_y = to_categorical(test_y, 3)
pretrain_label = to_categorical(pretrain_label, 3)

def shuffle(array_list):
    len_ = len(array_list[0])
    for x in array_list:
        assert len(x) == len_
    p = np.random.permutation(len_)
    return [x[p] for x in array_list]

train_x, train_y, train_aspect = shuffle([train_x, train_y, train_aspect])
pretrain_data, pretrain_label = shuffle([pretrain_data, pretrain_label])
doc_size = len(pretrain_data)

pretrain_data = pretrain_data[0: int(doc_size*args.percetage)]
pretrain_label = pretrain_label[0: int(doc_size*args.percetage)]
print 'Document size for training: ', len(pretrain_label)

validation_ratio = 0.2
validation_size = int(len(train_x) * validation_ratio)
print 'Validation size: ', validation_size
dev_x = train_x[:validation_size]
dev_y = train_y[:validation_size]
dev_aspect = train_aspect[:validation_size]
train_x = train_x[validation_size:]
train_y = train_y[validation_size:]
train_aspect = train_aspect[validation_size:]


def batch_generator(data_list_1, data_list_2, batch_size):
    len_1 = len(data_list_1[0])
    len_2 = len(data_list_2[0])

    data_list_1 = shuffle(data_list_1)
    batch_count = 0
    n_batch = len_1 / batch_size

    while True:
        if batch_count == n_batch:
            data_list_1 = shuffle(data_list_1)
            batch_count = 0

        x = data_list_1[0][batch_count*batch_size: (batch_count+1)*batch_size]
        y = data_list_1[1][batch_count*batch_size: (batch_count+1)*batch_size]
        aspect = data_list_1[2][batch_count*batch_size: (batch_count+1)*batch_size]
        batch_count += 1

        indices_2 = np.random.choice(len_2, batch_size)
        pretrain_x = data_list_2[0][indices_2]
        pretrain_y = data_list_2[1][indices_2]
        maxlen = np.max([len(d) for d in pretrain_x])
        pretrain_x = sequence.pad_sequences(pretrain_x, maxlen)

        yield x, y, aspect, pretrain_x, pretrain_y



###############################################################################################################################
## Optimizaer algorithm
#

from optimizers import get_optimizer
optimizer = get_optimizer(args)


###############################################################################################################################
## Building model
#

from model import create_model
from sklearn.metrics import precision_recall_fscore_support

def macro_f1(y_true, y_pred):
    preds = np.argmax(y_pred, axis=-1)
    true = np.argmax(y_true, axis=-1)
    p_macro, r_macro, f_macro, support_macro \
      = precision_recall_fscore_support(true, preds, average='macro')
    f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
    return f_macro


model = create_model(args, vocab, 3, overal_maxlen, overal_maxlen_aspect)
model.compile(optimizer=optimizer,
              loss={'aspect_model': 'categorical_crossentropy', 'pretrain_model': 'categorical_crossentropy'},
              loss_weights = {'aspect_model': 1, 'pretrain_model': args.alpha},
              metrics = {'aspect_model': 'categorical_accuracy', 'pretrain_model': 'categorical_accuracy'})



###############################################################################################################################
## Training
#

from tqdm import tqdm

logger.info('--------------------------------------------------------------------------------------------------------------------------')

train_gen = batch_generator([train_x, train_y, train_aspect], [pretrain_data, pretrain_label], batch_size=args.batch_size)
batches_per_epoch = len(train_x) / args.batch_size

for ii in xrange(args.epochs):
    t0 = time()
    overall_loss, aspect_loss, doc_loss, aspect_metric, doc_metric = 0., 0., 0., 0., 0

    for b in tqdm(xrange(batches_per_epoch)):
        batch_x,  batch_y, batch_aspect, batch_pretrain_x, batch_pretrain_y = train_gen.next()

        overall_loss_, aspect_loss_, doc_loss_, aspect_metric_, doc_metric_ = \
                model.train_on_batch([batch_x, batch_aspect, batch_pretrain_x], [batch_y, batch_pretrain_y])

        overall_loss += overall_loss_ / batches_per_epoch
        aspect_loss += aspect_loss_ / batches_per_epoch
        doc_loss += doc_loss_ / batches_per_epoch
        aspect_metric += aspect_metric_ / batches_per_epoch
        doc_metric += doc_metric_ / batches_per_epoch

    tr_time = time() - t0

    logger.info('Epoch %d, train: %is' % (ii, tr_time))
    logger.info('[Train] loss: %.4f, aspect loss: %.4f, doc loss: %.4f, aspect metric: %.4f, doc metric: %.4f' % (overall_loss, aspect_loss, doc_loss, aspect_metric, doc_metric))

    a, dev_loss, b, dev_metric, c = model.evaluate([dev_x, dev_aspect, dev_x], [dev_y, dev_y])
    y_pred = model.predict([dev_x, dev_aspect, dev_x])[0]
    dev_fs = macro_f1(dev_y, y_pred)
    logger.info('[Validation] loss: %.4f, acc: %.4f, macro_f1: %.4f' % (dev_loss, dev_metric, dev_fs))

    a, test_loss, b, test_metric, c = model.evaluate([test_x, test_aspect, test_x], [test_y, test_y])
    y_pred = model.predict([test_x, test_aspect, test_x])[0]
    test_fs = macro_f1(test_y, y_pred)
    logger.info('[Testing] loss: %.4f, acc: %.4f, macro_f1: %.4f' % (test_loss, test_metric, test_fs))


    


