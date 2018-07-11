"""This script demonstrates how to build a variational autoencoder with Keras.

 #Reference

 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
"""
from __future__ import print_function
import matplotlib.pyplot as plt
from keras.layers import *
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
Q1f = False

input_shape = (28, 28, 1)
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
# shape info needed to build decoder model
shape = K.int_shape(x)
# generate latent vector Q(z|X)
x = Flatten()(x)
x = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)
if Q1f:
    z_log_var = Dense(latent_dim, kernel_initializer=initializers.Zeros(), use_bias=False, trainable=False)(x)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon


z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# Layers setup
decoder_m1 = Dense(intermediate_dim, activation='relu')
decoder_m2 = Dense(shape[1] * shape[2] * shape[3], activation='relu')
decoder_m3 = Reshape((shape[1], shape[2], shape[3]))
decoder_m4 = Conv2D(8, (3, 3), activation='relu', padding='same')
decoder_m5 = UpSampling2D((2, 2))
decoder_m6 = Conv2D(16, (3, 3), activation='relu', padding='same')
decoder_m7 = UpSampling2D((2, 2))
decoder_output = Conv2D(1, (3, 3), activation='sigmoid', padding='same')

decoded_m1 = decoder_m1(z)
decoded_m2 = decoder_m2(decoded_m1)
decoded_m3 = decoder_m3(decoded_m2)
decoded_m4 = decoder_m4(decoded_m3)
decoded_m5 = decoder_m5(decoded_m4)
decoded_m6 = decoder_m6(decoded_m5)
decoded_m7 = decoder_m7(decoded_m6)
decoded_output = decoder_output(decoded_m7)

# instantiate VAE model
vae = Model(inputs, decoded_output, name='vae')
encoder = Model(inputs, [z_mean, z_log_var, z])

# Compute VAE loss
xent_loss = original_dim * metrics.binary_crossentropy(K.flatten(inputs), K.flatten(decoded_output))
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()


# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), 28, 28, 1))
x_test = x_test.reshape((len(x_test), 28, 28, 1))

vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))

# the test set in the latent space
encoded_images, _, _ = encoder.predict(x_test, batch_size=batch_size)

# visualize the test set in the latent space
if Q1c:
    plt.figure(figsize=(6, 6))
    plt.scatter(encoded_images[:, 0], encoded_images[:, 1], c=y_test)
    plt.colorbar()
    plt.show()

# generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
decoded_m1 = decoder_m1(decoder_input)
decoded_m2 = decoder_m2(decoded_m1)
decoded_m3 = decoder_m3(decoded_m2)
decoded_m4 = decoder_m4(decoded_m3)
decoded_m5 = decoder_m5(decoded_m4)
decoded_m6 = decoder_m6(decoded_m5)
decoded_m7 = decoder_m7(decoded_m6)
decoded_outputs = decoder_output(decoded_m7)
generator = Model(decoder_input, decoded_outputs)

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
