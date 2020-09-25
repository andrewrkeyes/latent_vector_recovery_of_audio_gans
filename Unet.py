import os
import tensorflow as tf
import time
import datetime

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

PATH_TRAIN = "LJspeech_spectrograms"
log_dir = "logs_specs/"

BUFFER_SIZE = 128
BATCH_SIZE = 64
IMG_WIDTH = 128
IMG_HEIGHT = 128
CHANNELS = 3
EPOCHS = 250


def load(image_file):
    input_image = tf.io.read_file(image_file)
    input_image = tf.image.decode_jpeg(input_image)
    input_image = tf.cast(input_image, tf.float32)

    return input_image


def resize(input_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.BICUBIC)

    return input_image


# normalizing the images to [0, 1]
def normalize(input_image):
    return input_image / 255.


def load_image_train(image_file):
    input_image = load(image_file)
    face_normalized = normalize(resize(input_image, IMG_HEIGHT, IMG_WIDTH))

    return face_normalized


def load_image_test(image_file):
    input_image = load(image_file)
    face_normalized = normalize(resize(input_image, IMG_HEIGHT, IMG_WIDTH))

    return face_normalized


def downsample(filters, size, strides=2, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, strides=2, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=strides,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def UNet():
    inputs = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, CHANNELS])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 64, 64, 64)
        downsample(128, 4),  # (bs, 32, 32, 128)
        downsample(256, 4),  # (bs, 16, 16, 256)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4),  # (bs, 16, 16, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='relu')  # (bs, 128, 128, 3)

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



def loss(hr_face, sr_face):
    # MAE between faces in pixel space
    pixel_loss = mae(hr_face, sr_face)

    return pixel_loss


@tf.function
def train_step(hr_face, epoch):
    with tf.GradientTape() as grad_tape:
        upsampled = model(hr_face, training=True)

        total_loss = loss(hr_face, upsampled)

        gradients = grad_tape.gradient(total_loss, model.trainable_variables)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar('total_loss', total_loss, step=epoch)


def fit(train_ds, epochs):
    for epoch in range(epochs):
        start = time.time()

        print("Epoch: ", epoch + 1)

        # Train
        for n, face_normalized in train_ds.enumerate():
            print(n.numpy() + 1)

            train_step(face_normalized, epoch)

        # saving (checkpoint) the model each epoch
        checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))


"""Input Pipeline"""
train_dataset = tf.data.Dataset.list_files(PATH_TRAIN + '/*')
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)


model = UNet()
model.summary()

"""Define Loss and Optimizer"""
mae = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.SGD(2e-4, momentum=0.9)

checkpoint_dir = './checkpoints_specs'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 model=model,
                                 )

summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

"""Train the Model"""
fit(train_dataset, EPOCHS)
