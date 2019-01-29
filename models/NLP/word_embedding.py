import io
import numpy as np


def load_glove():
    print('loading glove embeddings.. takes 2 mins generally')
    glove_filename = '/home/jehill/python/NLP/datasets/GloVE/glove.6B.300d.txt'
    glove_vocab = []
    glove_embed = []
    embedding_dict = {}
    word2id = {}  # word to id mappings

    file = open(glove_filename, 'r', encoding='UTF-8')

    for line in file.readlines():
        row = line.strip().split(' ')
        vocab_word = row[0]
        glove_vocab.append(vocab_word)
        word2id[vocab_word] = len(word2id)
        embed_vector = [float(i) for i in row[1:]]  # convert to list of float
        embedding_dict[vocab_word] = embed_vector
        glove_embed.append(embed_vector)

    print('Completed loading glove embeddings..')
    file.close()
    id2word = {v: k for k, v in word2id.items()}

    return glove_embed, embedding_dict, word2id, id2word, glove_vocab


def load_muse(emb_path, nmax=50000):
    vectors = []
    word2id = {}  # word to id mappings
    vocab = []  # vocab
    embedding_dict = {}  #
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)  # we start from 1st word vectors
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            embedding_dict[word] = vect
            vectors.append(vect)
            word2id[word] = len(word2id)
            vocab.append(word)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)

    print("Completed loading the following embedding .... ")
    print(emb_path)
    [number_words, embedding_dim] = np.shape(embeddings)

    print("Number of words:" + str(number_words))
    print("Embedding dimension:" + str(embedding_dim))

    return embeddings, embedding_dict, id2word, word2id, vocab


if __name__ == '__main__':
    # german_embedding_path = '/home/jehill/python/NLP/datasets/GloVE/MUSE/wiki.multi.de.vec'
    # english_embedding_path = '/home/jehill/python/NLP/datasets/GloVE/MUSE/wiki.multi.en.vec'
    # word_embeddings, embedding_dict, tgt_id2word, tgt_word2id, target_vocab = load_muse(english_embedding_path, nmax=200000)



    """    
    # look up our word vectors and store them as numpy arrays
    king_vector = np.array(embedding_dict['king'])
    man_vector = np.array(embedding_dict['man'])
    woman_vector = np.array(embedding_dict['woman'])

    # add/subtract our vectors

    new_vector = king_vector - man_vector + woman_vector

    print(king_vector.shape)

    # here we use a scipy function to create a "tree" of word vectors
    # that we can run queries against

    tree = spatial.KDTree(glove_embed)

    # run query with our new_vector to find the closest word vectors

    nearest_dist, nearest_idx = tree.query(new_vector, 10)
    nearest_words = [glove_vocab[i] for i in nearest_idx]
    print(nearest_words)

    ['king', 'queen', 'monarch', 'mother', 'princess', 'daughter', 'elizabeth', 'throne', 'kingdom', 'wife']
    """
