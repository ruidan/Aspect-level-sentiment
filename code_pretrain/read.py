import codecs
import operator
import numpy as np
import re

num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')

def create_vocab(domain, maxlen=0, vocab_size=0):
    
    print 'Creating vocab ...'

    f = '../data_doc/%s/text.txt'%(domain)

    total_words, unique_words = 0, 0
    word_freqs = {}

    fin = codecs.open(f, 'r', 'utf-8')
    for line in fin:
        words = line.split()
        if maxlen > 0 and len(words) > maxlen:
            continue

        for w in words:
            if not bool(num_regex.match(w)):
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


def create_data(vocab, text_path, label_path, domain, skip_top, skip_len, replace_non_vocab):
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

    print('  <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100*num_hit/total, 100*unk_hit/total))

    print domain
    print 'pos count: ', pos_count
    print 'neg count: ', neg_count
    print 'neu count: ', neu_count

    return np.array(data), np.array(label), max_len



def prepare_data(domain, vocab_size, skip_top=0, skip_len=0, replace_non_vocab=1):

    assert domain in ['electronics_large', 'yelp_large']

    vocab = create_vocab(domain, skip_len, vocab_size)

    text_path = '../data_doc/%s/text.txt'%(domain)
    score_path = '../data_doc/%s/label.txt'%(domain)

    data, label, max_len = create_data(vocab, text_path, score_path, domain, skip_top, skip_len, replace_non_vocab)

    return vocab, data, label, max_len

