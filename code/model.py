import logging
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, Input
from my_layers import Attention, Average, WeightedSum
from keras.models import Model


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def create_model(args, vocab, num_outputs, overal_maxlen, maxlen_aspect):
    
    ###############################################################################################################################
    ## Recurrence unit type
    #

    if args.recurrent_unit == 'lstm':
        from keras.layers.recurrent import LSTM as RNN
    elif args.recurrent_unit == 'gru':
        from keras.layers.recurrent import GRU as RNN
    elif args.recurrent_unit == 'simple':
        from keras.layers.recurrent import SimpleRNN as RNN

    ###############################################################################################################################
    ## Create Model
    #

    dropout = args.dropout_W       
    recurrent_dropout = args.dropout_U  
    vocab_size = len(vocab)

    logger.info('Building a LSTM attention model to predict term/aspect sentiment')
    print '\n\n'

    ##### Inputs #####
    sentence_input = Input(shape=(overal_maxlen,), dtype='int32', name='sentence_input')
    aspect_input = Input(shape=(maxlen_aspect,), dtype='int32', name='aspect_input')
    pretrain_input = Input(shape=(None,), dtype='int32', name='pretrain_input')

    ##### construct word embedding layer #####
    word_emb = Embedding(vocab_size, args.emb_dim, mask_zero=True, name='word_emb')

    ### represent aspect as averaged word embedding ###
    print 'use average term embs as aspect embedding'
    aspect_term_embs = word_emb(aspect_input)
    aspect_embs = Average(mask_zero=True, name='aspect_emb')(aspect_term_embs)

    ### sentence representation ###
    sentence_output = word_emb(sentence_input)
    pretrain_output = word_emb(pretrain_input)


    print 'use a rnn layer'
    rnn = RNN(args.rnn_dim, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout, name='lstm')
    sentence_output = rnn(sentence_output)
    pretrain_output = rnn(pretrain_output)

    print 'use content attention to get term weights'
    att_weights = Attention(name='att_weights')([sentence_output, aspect_embs])
    sentence_output = WeightedSum()([sentence_output, att_weights])

    pretrain_output = Average(mask_zero=True)(pretrain_output)
  
    if args.dropout_prob > 0:
        print 'use dropout layer'
        sentence_output = Dropout(args.dropout_prob)(sentence_output)
        pretrain_output = Dropout(args.dropout_prob)(pretrain_output)


    sentence_output = Dense(num_outputs, name='dense_1')(sentence_output)
    pretrain_output = Dense(num_outputs, name='dense_2')(pretrain_output)

    aspect_probs = Activation('softmax', name='aspect_model')(sentence_output)
    doc_probs = Activation('softmax', name='pretrain_model')(pretrain_output)

    model = Model(inputs=[sentence_input, aspect_input, pretrain_input], outputs=[aspect_probs, doc_probs])


    logger.info('  Done')

    ###############################################################################################################################
    ## Initialize embeddings if requested
    #

    if args.is_pretrain:

        import pickle

        print 'Set embedding, lstm, and dense weights from pre-trained models'
        if args.domain == 'lt':
            f_1 = open('../pretrained_weights/lstm_weights_lt%.1f.pkl'%(args.percetage), 'rb')
            f_2 = open('../pretrained_weights/dense_weights_lt%.1f.pkl'%(args.percetage), 'rb')
        else:
            f_1 = open('../pretrained_weights/lstm_weights_res%.1f.pkl'%(args.percetage), 'rb')
            f_2 = open('../pretrained_weights/dense_weights_res%.1f.pkl'%(args.percetage), 'rb')

        lstm_weights = pickle.load(f_1)
        dense_weights = pickle.load(f_2)
      
        model.get_layer('lstm').set_weights(lstm_weights)
        model.get_layer('dense_1').set_weights(dense_weights)
        model.get_layer('dense_2').set_weights(dense_weights)


    from w2vEmbReader import W2VEmbReader as EmbReader
    logger.info('Initializing lookup table')
    emb_path = '../glove/%s.txt'%(args.domain)
    emb_reader = EmbReader(args, emb_path)
    model.get_layer('word_emb').set_weights(emb_reader.get_emb_matrix_given_vocab(vocab, model.get_layer('word_emb').get_weights()))
    logger.info('  Done')

    return model







