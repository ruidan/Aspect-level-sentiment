import logging
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, Input
from keras.models import Model

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def create_model(args, vocab, num_outputs):
   
    ###############################################################################################################################
    ## Create Model
    #

    dropout = 0.5    
    recurrent_dropout = 0.1    
    vocab_size = len(vocab)

    ##### Inputs #####
    sentence_input = Input(shape=(None,), dtype='int32', name='sentence_input')

    word_emb = Embedding(vocab_size, args.emb_dim, mask_zero=True, name='word_emb')
    output = word_emb(sentence_input)

    print 'use a rnn layer'
    output = LSTM(args.rnn_dim, return_sequences=False, dropout=dropout, recurrent_dropout=recurrent_dropout, name='lstm')(output)

    print 'use 0.5 dropout layer'
    output = Dropout(0.5)(output)

    densed = Dense(num_outputs, name='dense')(output)
    probs = Activation('softmax')(densed)
    model = Model(inputs=[sentence_input], outputs=probs)

    
    ##### Initialization #####
    from w2vEmbReader import W2VEmbReader as EmbReader
    logger.info('Initializing lookup table')
    emb_path = '../glove/%s.txt'%(args.domain)
    emb_reader = EmbReader(emb_path, emb_dim=args.emb_dim)
    model.get_layer('word_emb').set_weights(emb_reader.get_emb_matrix_given_vocab(vocab, model.get_layer('word_emb').get_weights()))
    logger.info('  Done')
    
    return model

