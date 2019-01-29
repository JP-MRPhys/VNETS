import io

import numpy as np

data = np.random.uniform(-1.0, 1.0, (1, 300))[0]
print(np.shape(data))
print(type(data))


def splt(x):
    return x.split('   ')


def load_vec(emb_path, nmax=50000):
    vectors = []
    word2id = {}
    vocab = []
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)  # we start from 1st word vectors
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            vocab.append(word)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)

    print("Completed loading the following embedding .... ")
    print(emb_path)

    return embeddings, id2word, word2id, vocab


def load_glove(nmax=50000000):
    print("Load glove vectors .... ")

    emb_path = '/home/jehill/python/NLP/datasets/GloVE/glove.6B.300d.txt'
    vectors = []
    word2id = {}
    vocab = []
    index = 0
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        # next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)  # we start from 1st word vectors
            if index < 1:
                # print(word)
                index = index + 1

            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            vocab.append(word)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)

    print("Completed loading the following glove embedding.... ")
    print(np.shape(embeddings))

    print("Vocab size is :" + str(len(vocab)))

    return embeddings, id2word, word2id, vocab


target_embedding_path = '/home/jehill/python/NLP/datasets/GloVE/MUSE/wiki.multi.de.vec'
source_embedding_path = '/home/jehill/python/NLP/datasets/GloVE/MUSE/wiki.multi.en.vec'

# glove_embeddings, glove_id2word, glove_word2id, glove_vocab = load_glove()
# src_embeddings, src_id2word, src_word2id, source_vocab = load_vec(target_embedding_path, nmax=2000000)
# tgt_embeddings, tgt_id2word, tgt_word2id, target_vocab = load_vec(source_embedding_path, nmax=2000000)

# load the preprocessed and tokenised data from google (BPE preprocessing)
# details see.. https://google.github.io/seq2seq/nmt/


# dir='/home/jehill/python/NLP/nmt-master/nmt/wmt16_en_de_preprocessed/'
# source_vocab_file=os.path.join(dir, 'vocab.bpe.32000.en')
# target_vocab_file=os.path.join(dir, 'vocab.bpe.32000.de')

# source_file=os.path.join(dir, 'train.tok.clean.en')
# source_data=pd.read_fwf(source_file, header=None)
# source_data.columns=source_data.columns.map(str)


# target_file=os.path.join(dir, 'train.tok.clean.de')
# target_data=pd.read_fwf(target_file,header=None)
# target_data.columns=target_data.columns.map(str)

# target_vocab=pd.read_fwf(target_vocab_file, header=None)
# target_vocab.columns=target_vocab.columns.map(str)
# target_vocab=target_vocab["0"].tolist()

# source_vocab=pd.read_fwf(source_vocab_file, header=None)
# source_vocab.columns=source_vocab.columns.map(str)
# source_vocab=source_vocab["0"].tolist()


token_len = lambda x: len(x)


def get_source_word_id(data, vocab):
    word_id = []

    for word in data.split():

        # for word in sentence:  # replace this with tokens...
        # print(word)
        if word in vocab:
            word_id.append(src_word2id[word.lower()])
        else:
            # add un known token, as word not in vocab
            # print(word.lower() + ": is not in the source vocab \n")
            word_id.append(src_word2id['unknown'])

    # print(word_id)
    return word_id


def get_target_word_id(data, vocab):
    word_id = []

    for word in data.split('  '):
        # sentence_id = []
        # for word in sentence:  # replace this with tokens...
        # print(word)
        if word in vocab:

            word_id.append(tgt_word2id[word.lower()])
        else:
            continue
            # unknow token, as word not in vocab
            # print(word.lower() + ": is not in the target vocab \n")

    # print(word_id)
    return word_id


def print_shape(narray):
    [x] = np.shape(narray)
    print("X: " + str(x))


def corpus_to_vocab(filename):
    words = set(open(filename).read().split())
    print("Vocab size :" + str(len(words)))

    return words


def pad(x):
    tokens = []

    tokens.append('start ')

    for token in x:
        tokens.append(token)

    tokens.append('  stop')

    tokens = ''.join(tokens)

    return tokens


def get_vocab(data):
    vocab = []

    symbols = {0: 'PAD', 1: 'UNK'}

    for row in data:
        for token in row:
            vocab.append(token.lower())
            print(token)

    vocab = list(set(vocab))

    return vocab


# load glove or any word embedding
def word_embedding_matrix(glove_filename, vocab, dim):
    # first and second vector are pad and unk words
    # glove_filename is the file containing the word embedding, can be word2vec or your favourite model.

    with open(glove_filename, 'r') as f:
        word_vocab = []
        embedding_matrix = []
        word_vocab.extend(['PAD', 'UNK'])
        d = np.random.uniform(-1.0, 1.0, (1, 300))[0]
        print_shape(data)
        embedding_matrix.append(data)
        embedding_matrix.append(np.random.uniform(-1.0, 1.0, (1, dim))[0])

        index = 0
        for line in f.readlines():
            row = line.strip().split()
            print(row)
            word = row[0]
            embed_vector = [float(i) for i in row[1:]]
            embed_vector_array = np.array(embed_vector)
            index = index + 1

            if word in vocab and index > 1:
                print("Adding word embedding" + word)
                print(embed_vector)
                word_vocab.append(word)
                embedding_matrix.append(embed_vector_array)
        print(np.shape(embedding_matrix))

        print("Cleaned vocab:" + str(len(word_vocab)))

    return {'word_vocab': word_vocab, 'Embedding_matrix': np.reshape(embedding_matrix, [-1, dim]).astype(np.float32)}


# train_data=pd.DataFrame()
# train_data['source_tokens']=source_data["0"]
# train_data['target_tokens']=target_data["0"].apply(splt).apply(pad) #split charater to words, apply start end tokens.
# train_data['source_token_ids']=train_data["source_tokens"].apply(get_target_word_id, vocab=source_vocab)


# train_data['source_token_len']=train_data["source_tokens"].apply(token_len)
# train_data['target_token_len']=train_data["target_tokens"].apply(token_len)

# max_length_source=train_data['source_token_len'].max()
# max_length_target=train_data['target_token_len'].max()


if __name__ == '__main__':

    # data_sample=train_data.head(10)

    # print(data_sample)

    def get_batch_data(batch_data, source_vocab, target_vocab):

        source_ids = []
        target_ids = []

        for idx, row in batch_data.iterrows():
            source_ids.append(get_source_word_id(row.source_tokens, source_vocab))
            target_ids.append(get_target_word_id(row.target_tokens, target_vocab))

        print("Source lengths:" + str(len(source_ids)))
        print("Target lengths:" + str(len(target_ids)))

        return source_ids, target_ids


    # tf.reset_default_graph()

    # encoder_input=tf.placeholder(tf.int32, shape=[None,None]) #batch_size, length...

    # decoder_input=tf.placeholder(tf.int32, shape=[None,None])

    # encoder_embedding = tf.get_variable(name="encoder_embedding", shape=np.shape(src_embeddings),   initializer=tf.constant_initializer(src_embeddings), trainable=False)
    # decoder_embedding = tf.get_variable(name="decoder_embedding", shape=np.shape(tgt_embeddings), initializer=tf.constant_initializer(tgt_embeddings), trainable=False)

    # encoder_embedding_lookup=tf.nn.embedding_lookup(encoder_embedding,encoder_input)
    # decoder_embedding_lookup=tf.nn.embedding_lookup(decoder_embedding,decoder_input)

    # with tf.Session() as sess:

    #    sess.run(tf.global_variables_initializer())

    count = 0
    index = 0
    batch_size = 100

    # for index in range(0, len(train_data), batch_size):

    # source, target=get_batch_data(train_data[index:index+batch_size], source_vocab, target_vocab)

    # print(sess.run(encoder_embedding_lookup, feed_dict={encoder_input: [inpt_sentence]}))
    # print(sess.run(decoder_embedding_lookup, feed_dict={decoder_input: [translate_sentence]}))
