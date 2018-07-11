'''This script demonstrates how to build a variational autoencoder with Keras.

 #Reference

 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import initializers
from keras import metrics
from keras.datasets import mnist

batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 50
epsilon_std = 1.0

Q1c = True
Q1e = True
Q1f = True

x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
if Q1f:
    z_log_var = Dense(latent_dim, kernel_initializer=initializers.Zeros(), use_bias=False, trainable=False)(h)

enc = Dense(latent_dim, activation='relu')(x)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# instantiate VAE model
vae = Model(x, x_decoded_mean)
encoder = Model(x, z_mean)

# Compute VAE loss
xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()


# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))

# the test set in the latent space
encoded_images = encoder.predict(x_test, batch_size=batch_size)

# visualize the test set in the latent space
if Q1c:
    plt.figure(figsize=(6, 6))
    plt.scatter(encoded_images[:, 0], encoded_images[:, 1], c = y_test)
    plt.colorbar()
    plt.show()

# generating new digits from latent space points
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(inputs=decoder_input, outputs= _x_decoded_mean)

if Q1e:
    ind_two = np.where(y_test == 2)[0][0]
    x_t, y_t = encoded_images[ind_two][0], encoded_images[ind_two][1]
    ind_seven = np.where(y_test == 7)[0][0]
    x_s, y_s = encoded_images[ind_seven][0], encoded_images[ind_seven][1]
    x_axis = np.linspace(x_t, x_s, 10)
    cfs = np.polyfit([x_t, x_s], [y_t, y_s], 1)
    y_axis = [x*cfs[0]+cfs[1] for x in x_axis]

    for i in range(10):
        z_sample = np.array([[x_axis[i], y_axis[i]]]) * epsilon_std
        x_decoded = generator.predict(z_sample)
        ax = plt.subplot(1, 10, (i+1))
        plt.imshow(np.reshape(x_decoded, (28, 28)))
    plt.show()

