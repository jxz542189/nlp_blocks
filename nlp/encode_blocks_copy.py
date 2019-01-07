import tensorflow as tf
from nlp.nn import initializer, regularizer, spatial_dropout, get_lstm_init_state, dropout_res_layernorm


def LSTM_encode(seqs, causality=False, scope='lstm_encode_block', reuse=None, **kwargs):
    with tf.variable_scope(scope, reuse=reuse):
        if causality:
            kwargs['direction'] = 'unidirectional'
        if 'num_units' not in kwargs or kwargs['num_units'] is None:
            kwargs['num_units'] = seqs.get_shape().as_list()[-1]
        batch_size = tf.shape(seqs)[0]