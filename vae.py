import os
import datetime
import time as time
import scipy.io.wavfile
import glob
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import librosa

PATH_TRAIN = "/scratch/vkhazaie/deepfake/LJSpeech-1.1/"
log_dir = "/scratch/vkhazaie/deepfake/logs/"

BUFFER_SIZE = 1024
BATCH_SIZE = 128
IMG_WIDTH = 128
IMG_HEIGHT = 128
EPOCHS = 50
CHANNEL = 3
LATENT_DIM = 100

train_files = glob.glob(PATH_TRAIN + "/wavs/*.wav")

def convert_to_spectrogram(audio):
    audio = tf.squeeze(audio)
    spectrogram = tf.signal.stft(audio, 256, 128, pad_end=True)
    spectrogram = tf.expand_dims(spectrogram, -1)
    spectrogram = tf.stack([
        spectrogram,
        spectrogram,
        spectrogram
    ], axis=-1)
    spectrogram = tf.squeeze(spectrogram)
    spectrogram = tf.cast(spectrogram, tf.float32)
    spectrogram = tf.image.resize(spectrogram, (128, 128))

    return spectrogram


def train_generator():
    for x in range(0, len(train_files)):
        path = train_files[x]
        path_label = path.split("/")[-1].split("_")[0]
        sr, audio = scipy.io.wavfile.read(path)
        audio = audio.astype(np.float32)
        audio = audio / 32767
        audio = audio * 1.414
        audio = librosa.core.resample(audio, 16000, 16384)
        audio = audio[0:16384]
        data = np.zeros(16384)
        data[0:audio.shape[0]] = audio
        spectrogram = convert_to_spectrogram(data)

        max = abs(spectrogram.numpy().max())
        if abs(spectrogram.numpy().min()) > max:
            max = abs(spectrogram.numpy().min())

        spectrogram = spectrogram / max
        spectrogram = tf.clip_by_value(spectrogram, -1, 1)

        yield spectrogram


class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


initializer = tf.random_normal_initializer(0., 0.02)

# Encoder
encoder_inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNEL))

x = tf.keras.layers.Conv2D(64, 4, strides=2,
                           padding='same',
                           kernel_initializer=initializer,
                           use_bias=False)(encoder_inputs)
x = tf.keras.layers.LeakyReLU()(x)

x = tf.keras.layers.Conv2D(128, 4, strides=2,
                           padding='same',
                           kernel_initializer=initializer,
                           use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU()(x)

x = tf.keras.layers.Conv2D(256, 4, strides=2,
                           padding='same',
                           kernel_initializer=initializer,
                           use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU()(x)

x = tf.keras.layers.Conv2D(512, 4, strides=2,
                           padding='same',
                           kernel_initializer=initializer,
                           use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU()(x)

x = tf.keras.layers.Conv2D(512, 4, strides=2,
                           padding='same',
                           kernel_initializer=initializer,
                           use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU()(x)

x = tf.keras.layers.Conv2D(512, 4, strides=2,
                           padding='same',
                           kernel_initializer=initializer,
                           use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU()(x)

x = tf.keras.layers.Conv2D(512, 4, strides=2,
                           padding='same',
                           kernel_initializer=initializer,
                           use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU()(x)

x = tf.keras.layers.Flatten()(x)

z_mean = tf.keras.layers.Dense(LATENT_DIM, name="z_mean")(x)
z_log_var = tf.keras.layers.Dense(LATENT_DIM, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

encoder.summary()


# Decoder

latent_inputs = tf.keras.Input(shape=(LATENT_DIM,))
x = tf.keras.layers.Dense(1 * 1 * 512, activation="relu")(latent_inputs)
x = tf.keras.layers.Reshape((1, 1, 512))(x)

x = tf.keras.layers.Conv2DTranspose(512, 4, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.ReLU()(x)


x = tf.keras.layers.Conv2DTranspose(512, 4, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.ReLU()(x)


x = tf.keras.layers.Conv2DTranspose(512, 4, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.ReLU()(x)


x = tf.keras.layers.Conv2DTranspose(256, 4, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.ReLU()(x)


x = tf.keras.layers.Conv2DTranspose(128, 4, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)


x = tf.keras.layers.Conv2DTranspose(64, 4, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)


x = tf.keras.layers.Conv2DTranspose(3, 4, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('tanh')(x)

decoder = tf.keras.Model(latent_inputs, x, name="decoder")
decoder.summary()

vae_input = tf.keras.Input((128, 128, 3))
reconstructed = decoder(encoder(vae_input))
vae = tf.keras.Model(vae_input, reconstructed)
vae.summary()

@tf.function
def train_step(spectrogram, epoch):
    with tf.GradientTape() as tape:
        z_mean, z_log_var, z = encoder(spectrogram)
        reconstruction = decoder(z)
        reconstruction_loss = mae(spectrogram, reconstruction)
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        total_loss = reconstruction_loss + kl_loss

    grads = tape.gradient(total_loss, vae.trainable_weights)
    optimizer.apply_gradients(zip(grads, vae.trainable_weights))

    with summary_writer.as_default():
        tf.summary.scalar('loss', total_loss, step=epoch)
        tf.summary.scalar('reconstruction_loss', reconstruction_loss, step=epoch)
        tf.summary.scalar('kl_loss', kl_loss, step=epoch)



def fit(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        print("Epoch: ", epoch + 1)

        # Train
        for n, (spectrogram) in dataset.enumerate():
            train_step(spectrogram, epoch)

        # saving the model each epoch
        vae.save('/scratch/vkhazaie/deepfake/models/vae' + str(epoch) + '.h5')
        encoder.save('/scratch/vkhazaie/deepfake/models/encoder' + str(epoch) + '.h5')
        decoder.save('/scratch/vkhazaie/deepfake/models/decoder' + str(epoch) + '.h5')
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))


os.makedirs('/scratch/vkhazaie/deepfake/models', exist_ok=True)
dataset = tf.data.Dataset.from_generator(train_generator, output_types=(tf.float32)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
mae = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam(2e-3, beta_1=0.5)

summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

fit(dataset, EPOCHS)