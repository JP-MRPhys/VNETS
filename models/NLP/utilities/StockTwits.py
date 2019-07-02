import numpy as np
import pandas as pd
import spacy
from keras.preprocessing import sequence

nlp = spacy.load('en')


# nlp = spacy.load('en_vectors_web_lg')


def get_twets_data():
    data_location = '/home/jehill/python/NLP/datasets/twits/StockTwits_SPY_Sentiment_2017.gz'

    data = pd.read_csv(data_location,
                       encoding="utf-8",
                       compression="gzip",
                       index_col=0)

    return data


def sentence_embedding(sentence, embedding_dim):
    tokens = nlp(sentence)
    data = np.zeros((len(sentence), embedding_dim))
    k = 0
    for token in tokens:
        data[k, :] = token.vector
        k = k + 1

    return data


def get_training_batch_twets(data, batch_size, embedding_dim, num_classes, maxlen):
    num_classes = num_classes
    x = np.zeros([batch_size, maxlen, embedding_dim])
    y = np.zeros([batch_size, num_classes])

    index = 0

    bullish_count = 0
    bearish_count = 0

    for idx, row in data.iterrows():
        x[index, :, :] = sequence.pad_sequences([sentence_embedding(row['message'], embedding_dim)], maxlen=maxlen)
        if (row['sentiment'] == 'bullish'):
            y[index, :] = np.array([0, 1])
            bullish_count = bullish_count + 1
        else:
            y[index, :] = np.array([1, 0])
            bearish_count = bearish_count + 1

        index = index + 1

    print("Total messages in the batch" + str(len(data)))
    print("Bearish messages in the batch" + str(bearish_count))
    print("Bullish messages in the batch" + str(bullish_count))

    return x, y


if __name__ == '__main__':
    a = nlp('this')
    look_up = nlp.vocab.vectors.most_similar(a.vector)

    print(np.shape(a.vector))
    print(look_up)

    """
    train_data = get_twets_data();
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    emd_dim = 300
    batch_size = 1000

    for index in range(0, len(train_data), batch_size):
        print(index)
        BATCH_X, BATCH_Y = get_training_batch_twets(train_data[index:index + batch_size], batch_size=batch_size,
                                                    embedding_dim=emd_dim, num_classes=2, maxlen=250)
        print(np.shape(BATCH_X))
        print(np.shape(BATCH_Y))
    """
