#----------------------------library--------------------------------------------------------------------------------------------
import tensorflow_addons as tfa
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import cv2
from glob import glob
from matplotlib import pyplot
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.layers import *

import random

#----------------------------universal-variable-------------------------------------------------------------------------------------------
# SEED = 42
# #os.environ['PYTHONHASHSEED'] = str(SEED)
# random.seed(SEED)
# np.random.seed(SEED)
# tf.random.set_seed(SEED)
IMG_H = 64
IMG_W = 64
IMG_C = 3  ## Change this to 1 for grayscale.
#random_noise_size = 128
# batch_size = 32
# epochs = 100
discriminator_extra_steps = 3
batch_size = 128
latent_dim = 128
num_epochs = 100
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
IMAGE_COLOR = 3
w_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
#images_path = glob(f'/Users/yaxin/Documents/GitHub/Final-Project-Group3/Data/anime_face/*')


#----------------------------load-data-------------------------------------------------------------------------------------------
def load_image(image_path):

    img = tf.io.read_file(image_path)
    img = tf.io.decode_png(img)
    img = tf.image.resize_with_crop_or_pad(img, IMG_H, IMG_W)
    img = tf.cast(img, tf.float32)
    img = (img - 127.5) / 127.5 #????
    return img


def tf_dataset(images_path, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(images_path)
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
#----------------------------generator-------------------------------------------------------------------------------------------

# Builds the generator.
# Improvement 1: Added more Convolutional Layers, 4 in total
# Improvement 2: Replaced sigmoid with Tanh.
# Improvement 3: Added Dropout and batchnormlization for each layer
def build_generator(latent_dim):
    # # global IMAGE_HEIGHT, IMAGE_WIDTH
    generator = keras.models.Sequential()
    #
    # f = [2 ** i for i in range(5)][::-1]
    # filters = 32
    # output_strides = 16
    # h_output = 64 // output_strides
    # w_output = 64 // output_strides
    #
    # noise = tf.keras.layers.Input(shape=(latent_dim,), name="gen_noise_input")

    # generator.add(layers.Dense(f[0]*filters * h_output * w_output, use_bias=False)(noise))
    # generator.add(keras.layers.BatchNormalization())
    # generator.add(keras.layers.ReLU())
    # generator.add(keras.layers.Reshape(h_output, w_output,  16 * filters))
    # generator.add(keras.layers.Dense(units=IMG_H*IMG_W*IMG_C), use_bias=False, input_shape=[latent_dim]))
    # generator.add(keras.layers.BatchNormalization())
    # #generator.add(keras.layers.LeakyReLU())
    # generator.add(keras.layers.ReLU())

    generator.add(keras.layers.Dense(units=7 * 7 * 256, use_bias=False, input_shape=(latent_dim,)))
    generator.add(keras.layers.BatchNormalization())
    generator.add(keras.layers.ReLU())

    generator.add(keras.layers.Reshape((7, 7, 256)))


    generator.add(keras.layers.Conv2DTranspose(filters=32, kernel_size=5, strides=1, use_bias=False, padding='same'))
    generator.add(keras.layers.BatchNormalization())
    #generator.add(keras.layers.LeakyReLU())
    generator.add(keras.layers.ReLU())
    generator.add(keras.layers.Dropout(rate=0.3))

    generator.add(keras.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=1, use_bias=False, padding='same'))
    generator.add(keras.layers.BatchNormalization())
    # generator.add(keras.layers.LeakyReLU())
    generator.add(keras.layers.ReLU())
    generator.add(keras.layers.Dropout(rate=0.3))

    generator.add(keras.layers.Conv2DTranspose(filters=128, kernel_size=5, strides=2, use_bias=False, padding='same'))
    generator.add(keras.layers.BatchNormalization())
    #generator.add(keras.layers.LeakyReLU())
    generator.add(keras.layers.ReLU())
    generator.add(keras.layers.Dropout(rate=0.3))

    generator.add(keras.layers.Conv2DTranspose(filters=1, kernel_size=5, strides=2, padding='same',activation='tanh'))
    #generator.add(keras.tanh())

    return generator

#----------------------------discriminator-------------------------------------------------------------------------------------------

# Builds the discriminator.
# Improvement 6. Added Gaussian Noise in the inputs of discriminator.
def build_discriminator():
    discriminator = keras.models.Sequential()

    discriminator.add(keras.layers.GaussianNoise(stddev=0.2))
    discriminator.add(keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, padding='same', use_bias=False,
                                          input_shape=(64, 64, 3)))
    discriminator.add(keras.layers.BatchNormalization())
    discriminator.add(keras.layers.LeakyReLU())
    discriminator.add(keras.layers.Dropout(rate=0.3))

    discriminator.add(keras.layers.Conv2D(filters=128, kernel_size=5, strides=2, padding='same', use_bias=False))
    discriminator.add(keras.layers.BatchNormalization())
    discriminator.add(keras.layers.LeakyReLU())
    discriminator.add(keras.layers.Dropout(rate=0.3))

    discriminator.add(keras.layers.Conv2D(filters=384, kernel_size=5, strides=2, padding='same', use_bias=False))
    discriminator.add(keras.layers.BatchNormalization())
    discriminator.add(keras.layers.LeakyReLU())
    discriminator.add(keras.layers.Dropout(rate=0.3))

    discriminator.add(keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, padding='same', use_bias=False))
    discriminator.add(keras.layers.BatchNormalization())
    discriminator.add(keras.layers.LeakyReLU())
    discriminator.add(keras.layers.Dropout(rate=0.3))

    discriminator.add(keras.layers.Flatten())
    discriminator.add(keras.layers.Dense(units=1, activation='sigmoid'))
    return discriminator

#----------------------------GAN model-------------------------------------------------------------------------------------------

# Builds the GAN model
# Improvement 5. Added label smoothing to loss function.
class GAN(tf.keras.models.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        for _ in range(2):
            ## Train the discriminator
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            generated_images = self.generator(random_latent_vectors)
            generated_labels = tf.zeros((batch_size, 1))

            with tf.GradientTape() as ftape:
                predictions = self.discriminator(generated_images)
                d1_loss = self.loss_fn(generated_labels, predictions)
            grads = ftape.gradient(d1_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

            ## Train the discriminator
            labels = tf.ones((batch_size, 1))

            with tf.GradientTape() as rtape:
                predictions = self.discriminator(real_images)
                d2_loss = self.loss_fn(labels, predictions)
            grads = rtape.gradient(d2_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        ## Train the generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        misleading_labels = tf.ones((batch_size, 1))

        with tf.GradientTape() as gtape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = gtape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"d1_loss": d1_loss, "d2_loss": d2_loss, "g_loss": g_loss}

# def build_gan_model(generator, discriminator):
#     discriminator.compile(
#         optimizer=tfa.optimizers.Yogi(learning_rate=0.001),
#         loss=keras.losses.BinaryCrossentropy(label_smoothing=0.25))
#     discriminator.trainable = False
#
#     gan = keras.models.Sequential([generator, discriminator])
#     gan.compile(
#         optimizer=tfa.optimizers.Yogi(learning_rate=0.001),
#         loss=keras.losses.BinaryCrossentropy(label_smoothing=0.25))
#     return gan

#----------------------------save model-------------------------------------------------------------------------------------------

# Saving the weights.
def save_model(gan, generator, discriminator):
    discriminator.trainable = False
    tf.keras.models.save_model(gan, 'model/gan')
    discriminator.trainable = True
    tf.keras.models.save_model(generator, 'model/generator')
    tf.keras.models.save_model(discriminator, 'model/discriminator')


#---------------------- Loading GAN's weights-----------------------------------------
def load_model():
    discriminator = tf.keras.models.load_model('model/discriminator')
    generator = tf.keras.models.load_model('model/generator')
    gan = tf.keras.models.load_model('model/gan')
    gan.summary()
    discriminator.summary()
    generator.summary()
    return generator, discriminator, gan

def save_plot(examples, epoch, n):
    examples = (examples + 1) / 2.0
    for i in range(n * n):
        pyplot.subplot(n, n, i+1)
        pyplot.axis("off")
        pyplot.imshow(examples[i])  ## pyplot.imshow(np.squeeze(examples[i], axis=-1))
    filename = f"/Users/yaxin/Documents/GitHub/Final-Project-Group3/Data/sample/opt_generated_plot_epoch-{epoch+1}.png"
    pyplot.savefig(filename)
    pyplot.close()

#-----------------------Building the model--------------------------------------------------------
#random_noise_size = 128
# generator = build_generator(latent_dim)
# discriminator = build_discriminator()
# GAN = build_gan_model(generator, discriminator)
if __name__ == "__main__":
    ## Hyperparameters
    batch_size = 128
    latent_dim = 128
    num_epochs = 100
    images_path = glob("/Data/anime_face/*")

    d_model = build_discriminator()
    g_model = build_generator(latent_dim)

    # d_model.load_weights("saved_model/d_model.h5")
    # g_model.load_weights("saved_model/g_model.h5")

    # d_model.summary()
    # g_model.summary()

    gan = GAN(d_model, g_model, latent_dim)

    bce_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.25)
    d_optimizer = tfa.optimizers.Yogi(learning_rate=0.001)
    g_optimizer = tfa.optimizers.Yogi(learning_rate=0.001)
    gan.compile(d_optimizer, g_optimizer, bce_loss_fn)

    images_dataset = tf_dataset(images_path, batch_size)

    for epoch in range(num_epochs):
        gan.fit(images_dataset, epochs=1)
        # g_model.save("saved_model/g_model.h5")
        # d_model.save("saved_model/d_model.h5")

        n_samples = 25
        noise = np.random.normal(size=(n_samples, latent_dim))
        examples = g_model.predict(noise)
        save_plot(examples, epoch, int(np.sqrt(n_samples)))
#-----------------------train process--------------------------------------------------------

# Training the GAN model.

# # Loading the MNIST Digits dataset.
# (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
#
# # Normalizing the data in range [-1, 1].
# x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype(np.float32)
# x_train = (x_train - 127.5) / 127.5
# x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype(np.float32)
# x_test = (x_test - 127.5) / 127.5
#
# dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(buffer_size=x_train.shape[0])
# inputs = dataset.batch(batch_size=batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
# batches_per_epoch = x_train.shape[0] // batch_size
#
# real_labels = tf.constant([[1.0]] * batch_size)
# mixed_labels = tf.constant([[0.0]] * batch_size + [[1.0]] * batch_size)
#
# for epoch in range(epochs):
#     print('\nTraining on epoch', epoch + 1)
#
#     for i, x_batch in enumerate(inputs):
#         # Training the discriminator first.
#         # Improvement 4. Training the discriminator more steps.
#         discriminator_loss = 0
#         for step in range(discriminator_extra_steps):
#             discriminator.trainable = True
#             random_noise = tf.random.normal(shape=[batch_size, random_noise_size])
#             fake_images = generator(random_noise)
#             mixed_images = tf.concat([fake_images, tf.dtypes.cast(x_batch, tf.float32)], axis=0)
#             discriminator_loss = discriminator.train_on_batch(mixed_images, mixed_labels)
#
#         # Training the generator after.
#         discriminator.trainable = False
#         random_noise = tf.random.normal(shape=[batch_size, random_noise_size])
#         generator_loss = GAN.train_on_batch(random_noise, real_labels)
#
#         print('\rCurrent batch: {}/{} , Discriminator loss = {} , Generator loss = {}'.format(
#             i + 1,
#             batches_per_epoch,
#             discriminator_loss,
#             generator_loss), end='')

# Saving the model.
#save_model(GAN, generator, discriminator)

# # Generating digits.
# digits_to_generate = 25
# random_noise = tf.random.normal(shape=[digits_to_generate, random_noise_size])
# generated_digits = generator(random_noise)
#
# rows = 5
# cols = 5
# fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 10))
# for i, digit in enumerate(generated_digits):
#     ax = axes[i // rows, i % cols]
#     ax.imshow(digit * 127.5 + 127.5, cmap='gray')
# plt.tight_layout()
# plt.show()