import tensorflow as tf
import time
import datetime
import numpy as np
import os


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


# Load the model
tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()
saver = tf.compat.v1.train.import_meta_graph('infer.meta')
graph = tf.compat.v1.get_default_graph()
sess = tf.compat.v1.InteractiveSession()
saver.restore(sess, 'model.ckpt')



log_dir = "logs/"
BATCH_SIZE = 64
STEPS = 20000


def generate_spectrograms(ngenerate=64, set_random=True, latent=None):
    z = graph.get_tensor_by_name('z:0')
    G_z = graph.get_tensor_by_name('G_z:0')[:, :, 0]
    G_z_spec = graph.get_tensor_by_name('G_z_spec:0')

    if set_random:
        # Sample latent vectors
        _z = (np.random.rand(ngenerate, 100) * 2.) - 1.

        # Generate
        _G_z, _G_z_spec = sess.run([G_z, G_z_spec], {z: _z})

    else:
        # Generate
        _G_z, _G_z_spec = sess.run([G_z, G_z_spec], {z: latent})


    spectrograms = np.expand_dims(np.array(_G_z_spec), axis=3).astype("float32") / 255.

    return spectrograms



def UNet():
    # Encoder
    inputs = tf.keras.layers.Input(shape=[256, 256, 1])

    enc_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same',
                                   kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(inputs)
    enc_1 = tf.keras.layers.LeakyReLU()(enc_1)

    enc_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, padding='same',
                                   kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(enc_1)
    enc_2 = tf.keras.layers.BatchNormalization()(enc_2)
    enc_2 = tf.keras.layers.LeakyReLU()(enc_2)

    enc_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=4, strides=2, padding='same',
                                   kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(enc_2)
    enc_3 = tf.keras.layers.BatchNormalization()(enc_3)
    enc_3 = tf.keras.layers.LeakyReLU()(enc_3)

    enc_4 = tf.keras.layers.Conv2D(filters=512, kernel_size=4, strides=2, padding='same',
                                   kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(enc_3)
    enc_4 = tf.keras.layers.BatchNormalization()(enc_4)
    enc_4 = tf.keras.layers.LeakyReLU()(enc_4)

    enc_5 = tf.keras.layers.Conv2D(filters=512, kernel_size=4, strides=2, padding='same',
                                   kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(enc_4)
    enc_5 = tf.keras.layers.BatchNormalization()(enc_5)
    enc_5 = tf.keras.layers.LeakyReLU()(enc_5)

    enc_6 = tf.keras.layers.Conv2D(filters=512, kernel_size=4, strides=2, padding='same',
                                   kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(enc_5)
    enc_6 = tf.keras.layers.BatchNormalization()(enc_6)
    enc_6 = tf.keras.layers.LeakyReLU()(enc_6)

    enc_7 = tf.keras.layers.Conv2D(filters=512, kernel_size=4, strides=2, padding='same',
                                   kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(enc_6)
    enc_7 = tf.keras.layers.BatchNormalization()(enc_7)
    enc_7 = tf.keras.layers.LeakyReLU()(enc_7)

    enc_8 = tf.keras.layers.Conv2D(filters=512, kernel_size=4, strides=2, padding='same',
                                   kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(enc_7)
    enc_8 = tf.keras.layers.BatchNormalization()(enc_8)
    enc_8 = tf.keras.layers.LeakyReLU()(enc_8)

    # latent space
    latent = tf.keras.layers.Reshape((512,))(enc_8)
    latent = tf.keras.layers.Dense(100, activation='tanh')(latent)
    dec_1 = tf.keras.layers.Dense(512)(latent)
    dec_1 = tf.keras.layers.Reshape((1, 1, 512))(dec_1)
    dec_1 = tf.keras.layers.BatchNormalization()(dec_1)
    dec_1 = tf.keras.layers.LeakyReLU()(dec_1)

    # Decoder
    dec_1 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(
        dec_1)
    dec_1 = tf.keras.layers.BatchNormalization()(dec_1)
    dec_1 = tf.keras.layers.Dropout(0.5)(dec_1)
    dec_1 = tf.keras.layers.ReLU()(dec_1)
    dec_1 = tf.keras.layers.Concatenate()([dec_1, enc_7])

    dec_2 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(
        dec_1)
    dec_2 = tf.keras.layers.BatchNormalization()(dec_2)
    dec_2 = tf.keras.layers.Dropout(0.5)(dec_2)
    dec_2 = tf.keras.layers.ReLU()(dec_2)
    dec_2 = tf.keras.layers.Concatenate()([dec_2, enc_6])

    dec_3 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(
        dec_2)
    dec_3 = tf.keras.layers.BatchNormalization()(dec_3)
    dec_3 = tf.keras.layers.Dropout(0.5)(dec_3)
    dec_3 = tf.keras.layers.ReLU()(dec_3)
    dec_3 = tf.keras.layers.Concatenate()([dec_3, enc_5])

    dec_4 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(
        dec_3)
    dec_4 = tf.keras.layers.BatchNormalization()(dec_4)
    dec_4 = tf.keras.layers.Dropout(0.5)(dec_4)
    dec_4 = tf.keras.layers.ReLU()(dec_4)
    dec_4 = tf.keras.layers.Concatenate()([dec_4, enc_4])

    dec_5 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(
        dec_4)
    dec_5 = tf.keras.layers.BatchNormalization()(dec_5)
    dec_5 = tf.keras.layers.ReLU()(dec_5)
    dec_5 = tf.keras.layers.Concatenate()([dec_5, enc_3])

    dec_6 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(
        dec_5)
    dec_6 = tf.keras.layers.BatchNormalization()(dec_6)
    dec_6 = tf.keras.layers.ReLU()(dec_6)
    dec_6 = tf.keras.layers.Concatenate()([dec_6, enc_2])

    dec_7 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(
        dec_6)
    dec_7 = tf.keras.layers.BatchNormalization()(dec_7)
    dec_7 = tf.keras.layers.ReLU()(dec_7)
    dec_7 = tf.keras.layers.Concatenate()([dec_7, enc_1])

    last = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=4, strides=2, padding='same',
                                           kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                           activation='relu')(dec_7)

    return tf.keras.Model(inputs=inputs, outputs=[last, latent])


def loss(x_spec, x_spec_autoencoded, x_spec_prime):
    reconstruction_loss = mae(x_spec, x_spec_autoencoded)
    inverse_loss = mae(x_spec, x_spec_prime)

    return reconstruction_loss + inverse_loss



@tf.function
def fit(steps):
    for step in range(steps):
        start = time.time()

        print("Step: [", step + 1, '/', steps, ']')

        # Train
        with tf.GradientTape() as grad_tape:
            spec = generate_spectrograms(ngenerate=BATCH_SIZE, set_random=True)
            x_spec_autoencoded, latent = model(spec, training=True)
            x_spec_prime = generate_spectrograms(ngenerate=BATCH_SIZE, set_random=False, latent=latent)

            total_loss = loss(spec, x_spec_autoencoded, x_spec_prime)

            gradients = grad_tape.gradient(total_loss, model.trainable_variables)

            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            with summary_writer.as_default():
                tf.summary.scalar('total_loss', total_loss, step=step)

        print('Time taken for this step is {} sec\n'.format(time.time() - start))

        # saving (checkpoint) the model each epoch
        checkpoint.save(file_prefix=checkpoint_prefix)



"""Building the model"""
model = UNet()
model.summary()


"""Define Loss and Optimizer"""
mae = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.SGD(1e-3, momentum=0.9)


checkpoint_dir = './checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 model=model,
                                 )
summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


"""Train the Model"""
fit(STEPS)

