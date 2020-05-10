import tensorflow as tf
from constants import *


def gram(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (x.shape[0], -1))
    return tf.matmul(features, tf.transpose(features))


def sse(base, combo):
    return tf.reduce_mean(tf.square(base - combo))


def style_loss(base, combo):
    base_gram = gram(base)
    combo_gram = gram(combo)
    return tf.reduce_mean(tf.square(base_gram - combo_gram))


def variation(x):
    a = x[:, 1:, :ncols-1, :] - x[:, :nrows-1, :ncols-1, :]
    b = x[:, :nrows-1, 1:, :] - x[:, :nrows-1, :ncols-1, :]
    return tf.pow(tf.reduce_mean(tf.square(a)) + tf.reduce_mean(tf.square(b)), 1.25)


def loss(base, style, combo, style_layer_names, content_layer_name, feature_map):
    inp = tf.concat([base, combo, style], axis=0)
    features = feature_map(inp)

    bf, sf, cf = 0, 1, 2

    s = tf.reduce_sum([
        style_loss(features[layer][cf], features[layer][sf])
        for layer in style_layer_names
    ])

    c = sse(features[content_layer_name][bf], features[content_layer_name][cf])

    v = variation(combo)

    return variation_weight*v + style_weight*s + content_weight*c
