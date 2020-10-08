import time
import datetime
import cv2
import os
import time as time
import scipy.io.wavfile
import glob
import numpy as np
import tensorflow as tf
import librosa



PATH_TRAIN = "/scratch/vkhazaie/deepfake/LJSpeech-1.1/"
log_dir = "/scratch/vkhazaie/deepfake/logs_alocc/"

BUFFER_SIZE = 128
BATCH_SIZE = 128
IMG_WIDTH = 128
IMG_HEIGHT = 128
EPOCHS = 50
CHANNELS = 3

"""Input Pipeline"""
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


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def Generator():
    inputs = tf.keras.layers.Input(shape=[IMG_HEIGHT,IMG_WIDTH,CHANNELS])

    down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(128, 4), # (bs, 64, 64, 128)
    downsample(256, 4), # (bs, 32, 32, 256)
    downsample(512, 4), # (bs, 16, 16, 512)
    downsample(512, 4), # (bs, 8, 8, 512)
    downsample(512, 4), # (bs, 4, 4, 512)
    downsample(512, 4), # (bs, 2, 2, 512)
    ]

    up_stack = [
    upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(512, 4), # (bs, 16, 16, 1024)
    upsample(256, 4), # (bs, 32, 32, 512)
    upsample(128, 4), # (bs, 64, 64, 256)
    upsample(64, 4), # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + l1_loss

    return total_gen_loss, gan_loss, l1_loss


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[IMG_HEIGHT,IMG_WIDTH,CHANNELS], name='input_image')

    down1 = downsample(64, 4, False)(inp) # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

    x = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.Model(inputs=inp, outputs=x)


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss



@tf.function
def train_step(input_image, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator(input_image, training=True)
        disc_generated_output = discriminator(gen_output, training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, input_image)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)


def fit(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        print("Epoch: ", epoch + 1)

        # Train
        for n, (spectrogram) in dataset.enumerate():
            train_step(spectrogram, epoch)

        # saving the model each epoch
        generator.save('/scratch/vkhazaie/deepfake/models_alocc/encoder' + str(epoch) + '.h5')
        discriminator.save('/scratch/vkhazaie/deepfake/models_alocc/discriminator' + str(epoch) + '.h5')
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))


"""Building the model"""
generator = Generator()
discriminator = Discriminator()

"""Define Loss and Optimizer"""
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


os.makedirs('/scratch/vkhazaie/deepfake/models_alocc', exist_ok=True)
dataset = tf.data.Dataset.from_generator(train_generator, output_types=(tf.float32)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

fit(dataset, EPOCHS)
