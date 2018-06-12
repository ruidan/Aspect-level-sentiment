import codecs
import re
from itertools import izip
import operator
import numpy as np  


num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')

def is_number(token):
    return bool(num_regex.match(token))


def create_vocab(domain, maxlen=0, vocab_size=0):
    assert domain in ['res', 'lt', 'res_15', 'res_16']

    file_list = ['../data_aspect/%s/train/sentence.txt'%(domain),
                 '../data_aspect/%s/test/sentence.txt'%(domain)]

    if domain in ['lt']:
        file_list.append('../data_doc/electronics_large/text.txt')
    else:
        file_list.append('../data_doc/yelp_large/text.txt')

    print 'Creating vocab ...'

    total_words, unique_words = 0, 0
    word_freqs = {}

    for f in file_list:
        top = 0
        fin = codecs.open(f, 'r', 'utf-8')
        for line in fin:
            words = line.split()
            if maxlen > 0 and len(words) > maxlen:
                continue
            for w in words:
                if not is_number(w):
                    try:
                        word_freqs[w] += 1
                    except KeyError:
                        unique_words += 1
                        word_freqs[w] = 1
                    total_words += 1

    print ('  %i total words, %i unique words' % (total_words, unique_words))
    sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)

    vocab = {'<pad>':0, '<unk>':1, '<num>':2}
    index = len(vocab)
    for word, _ in sorted_word_freqs:
        vocab[word] = index
        index += 1
        if vocab_size > 0 and index > vocab_size + 2:
            break
    if vocab_size > 0:
        print (' keep the top %i words' % vocab_size)

    #Write vocab to a txt file
    # vocab_file = codecs.open(domain+'_vocab', mode='w', encoding='utf8')
    # sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1))
    # for word, index in sorted_vocab:
    #     vocab_file.write(word+'\t'+str(index)+'\n')
    # vocab_file.close()
    
    return vocab


def read_dataset_aspect(domain, phase, vocab, maxlen):
    assert domain in ['res', 'lt', 'res_15', 'res_16']
    assert phase in ['train', 'test']
    
    print 'Preparing dataset ...'

    data_x, data_y, aspect = [], [], []
    polarity_category = {'positive': 0, 'negative': 1, 'neutral': 2}

    file_names = [ '../data_aspect/%s/%s/sentence.txt'%(domain, phase),
                   '../data_aspect/%s/%s/polarity.txt'%(domain, phase),
                   '../data_aspect/%s/%s/term.txt'%(domain, phase)]

    num_hit, unk_hit, total = 0., 0., 0.
    maxlen_x = 0
    maxlen_aspect = 0

    files = [open(i, 'r') for i in file_names]
    for rows in izip(*files):
        content = rows[0].strip().split()
        polarity = rows[1].strip()
        aspect_content = rows[2].strip().split()

        if maxlen > 0 and len(content) > maxlen:
            continue

        content_indices = []
        if len(content) == 0:
            content_indices.append(vocab['<unk>'])
            unk_hit += 1
        for word in content:
            if is_number(word):
                content_indices.append(vocab['<num>'])
                num_hit += 1
            elif word in vocab:
                content_indices.append(vocab[word])
            else:
                content_indices.append(vocab['<unk>'])
                unk_hit += 1
            total += 1

        data_x.append(content_indices)
        data_y.append(polarity_category[polarity])

        aspect_indices = []
        if len(aspect_content) == 0:
            aspect_indices.append(vocab['<unk>'])
            unk_hit += 1
        for word in aspect_content:
            if is_number(word):
                aspect_indices.append(vocab['<num>'])
            elif word in vocab:
                aspect_indices.append(vocab[word])
            else:
                aspect_indices.append(vocab['<unk>'])
        aspect.append(aspect_indices)

        if maxlen_x < len(content_indices):
            maxlen_x = len(content_indices)
        if maxlen_aspect < len(aspect_indices):
            maxlen_aspect = len(aspect_indices)


    
    print '  <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100*num_hit/total, 100*unk_hit/total)
    return data_x, data_y, aspect, maxlen_x, maxlen_aspect


def get_data_aspect(vocab, domain, maxlen=0):
    assert domain in ['res', 'lt', 'res_15', 'res_16']

    train_x, train_y, train_aspect, train_maxlen, train_maxlen_aspect = read_dataset_aspect(domain, 'train', vocab, maxlen)
    test_x, test_y, test_aspect, test_maxlen, test_maxlen_aspect = read_dataset_aspect(domain, 'test', vocab, maxlen)
    overal_maxlen = max(train_maxlen, test_maxlen)
    overal_maxlen_aspect = max(train_maxlen_aspect, test_maxlen_aspect)

    print ' Overal_maxlen: ', overal_maxlen
    print ' Overal_maxlen_aspect: ', overal_maxlen_aspect
    return train_x, train_y, train_aspect, test_x, test_y, test_aspect, overal_maxlen, overal_maxlen_aspect


def create_data(vocab, text_path, label_path, skip_top, skip_len, replace_non_vocab):
    data = []
    label = [] # {pos: 0, neg: 1, neu: 2}
    f = codecs.open(text_path, 'r', 'utf-8')
    f_l = codecs.open(label_path, 'r', 'utf-8')
    num_hit, unk_hit, skip_top_hit, total = 0., 0., 0., 0.
    pos_count, neg_count, neu_count = 0, 0, 0
    max_len = 0

    for line, score in zip(f, f_l):
        word_indices = []
        words = line.split()
        if skip_len > 0 and len(words) > skip_len:
            continue

        score = float(score.strip())
        if score < 3:
            neg_count += 1
            label.append(1)
        elif score > 3:
            pos_count += 1
            label.append(0)
        else:
            neu_count += 1
            label.append(2)
            
        for word in words:
            if bool(num_regex.match(word)):
                word_indices.append(vocab['<num>'])
                num_hit += 1
            elif word in vocab:
                word_ind = vocab[word]
                if skip_top > 0 and word_ind < skip_top + 3:
                    skip_top_hit += 1
                else:
                    word_indices.append(word_ind)
            else:
                if replace_non_vocab:
                    word_indices.append(vocab['<unk>'])
                unk_hit += 1
            total += 1

        if len(word_indices) > max_len:
            max_len = len(word_indices)

        data.append(word_indices)

    f.close()
    f_l.close()

    print('  <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' %(100*num_hit/total, 100*unk_hit/total))

    return np.array(data), np.array(label), max_len



def prepare_data_doc(vocab, domain, skip_top=0, skip_len=0, replace_non_vocab=1):
   
    if domain in ['lt']:
        text_path = '../data_doc/electronics_large/text.txt'
        score_path = '../data_doc/electronics_large/label.txt'
    else:
        text_path= '../data_doc/yelp_large/text.txt'
        score_path = '../data_doc/yelp_large/label.txt'

    data, label, max_len = create_data(vocab, text_path, score_path, skip_top, skip_len, replace_non_vocab)

    return data, label, max_len


def prepare_data(domain, vocab_size, maxlen=0):
    vocab = create_vocab(domain, maxlen, vocab_size)

    train_x, train_y, train_aspect, test_x, test_y, test_aspect, overal_maxlen, overal_maxlen_aspect = get_data_aspect(vocab, domain)

    pretrain_data, pretrain_label, pretrain_maxlen = prepare_data_doc(vocab, domain)

    return train_x, train_y, train_aspect, test_x, test_y, test_aspect, vocab, overal_maxlen, overal_maxlen_aspect, pretrain_data, pretrain_label, pretrain_maxlen


