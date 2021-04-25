import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_batch_1(x,y,batch_size):
    a = list(range(len(x)))
    np.random.shuffle(a)
    x = x[a]
    y = y[a]

    batch_x = [x[batch_size * i : (i+1)*batch_size,:].tolist() for i in range(len(x)//batch_size)]
    batch_y = [y[batch_size * i : (i+1)*batch_size].tolist() for i in range(len(x)//batch_size)]
    return batch_x, batch_y

def create_batch_2(x,batch_size):
    a = list(range(len(x)))
    np.random.shuffle(a)
    x = x[a]
    batch_x = [x[batch_size * i : (i+1)*batch_size,:] for i in range(len(x)//batch_size)]
    return batch_x

""" Calculates the gradient penalty"""
def gradient_penalty(discriminator, normal, anomaly):
	alpha = tf.random.uniform([normal.shape[0], 1, 1, 1], 0.0, 1.0)
	diff = alpha * x + (1 - alpha) * anomaly
	
	with tf.GradientTape() as gp_tape:
		gp_tape.watch(diff)
            # 1. Get the discriminator output for this interpolated data.
		pred = discriminator(diff, training=True)
		# 2. Calculate the gradients w.r.t to this interpolated data.
		grads = gp_tape.gradient(pred, diff)
		norm = tf.sqrt(tf.reduce_sum(grads ** 2, axis=[1, 2]))
		gp = tf.reduce_mean((norm - 1.0) ** 2)
		return gp
