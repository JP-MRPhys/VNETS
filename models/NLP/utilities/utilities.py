import nltk
import tensorflow as tf


# cherry pick the best translations based on blue scores

def cherry_pick(records, n, upper_bound=1.0):
    bleus = []

    for en, ch_gr, ch_pd in records:
        bleu = nltk.translate.bleu_score.sentence_bleu(
            [ch_gr], ch_pd)  # caculate BLEU by nltk
        bleus.append(bleu)

    lst = [i for i in range(len(records)) if bleus[i] <= upper_bound]
    lst = sorted(lst, key=lambda i: bleus[i], reverse=True)  # sort by BLEU score

    return [records[lst[i]] for i in range(n)]


def dense(self,
          inputs,
          units,
          activation=tf.tanh,
          use_bias=True,
          name=None):
    """ Fully-connected layer. """
    if activation is not None:
        activity_regularizer = self.fc_activity_regularizer
    else:
        activity_regularizer = None
    return tf.layers.dense(
        inputs=inputs,
        units=units,
        activation=activation,
        use_bias=use_bias,
        trainable=self.is_train,
        kernel_initializer=self.fc_kernel_initializer,
        kernel_regularizer=self.fc_kernel_regularizer,
        activity_regularizer=activity_regularizer,
        name=name)
