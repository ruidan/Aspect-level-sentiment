import codecs
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

class W2VEmbReader:

    def __init__(self, emb_path, emb_dim=None):
        logger.info('Loading embeddings from: ' + emb_path)
        self.embeddings = {}

        emb_file = codecs.open(emb_path, 'r', encoding='utf8')
        self.vocab_size = 0
        self.emb_dim = -1
        for line in emb_file:
            tokens = line.split()
            if len(tokens) == 0:
                continue
            if self.emb_dim == -1:
                self.emb_dim = len(tokens) - 1
                assert self.emb_dim == emb_dim
            
            word = tokens[0]
            vec = tokens[1:]
            self.embeddings[word] = vec
            self.vocab_size += 1
        emb_file.close()

        logger.info('  #vectors: %i, #dimensions: %i' % (self.vocab_size, self.emb_dim))


    def get_emb_matrix_given_vocab(self, vocab, emb_matrix):
        counter = 0.
        for word, index in vocab.iteritems():
            try:
                emb_matrix[0][index] = self.embeddings[word]
                counter += 1
            except KeyError:
                pass

        logger.info('%i/%i word vectors initialized (hit rate: %.2f%%)' % (counter, len(vocab), 100*counter/len(vocab)))
        
        return emb_matrix

        