import numpy as np
import PIL.Image
import time as time
import tensorflow as tf
from tensorflow.python.client import device_lib
import tensorflow_hub as hub
from tqdm import tqdm

from models import resnet
from tensorboardX import SummaryWriter

# CHANGE THIS to use a different dataset
dataset = 'speech' # one of 'digits', 'speech', 'birds', 'drums', 'piano'

# Confirm GPU is running

writer = SummaryWriter("./logs")

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
if len(get_available_gpus()) == 0:
  for i in range(4):
    print('WARNING: Not running on a GPU! See above for faster generation')


with tf.Graph().as_default():
  #tf.compat.v1.reset_default_graph()
  saver = tf.compat.v1.train.import_meta_graph('checkpoint/infer.meta')
  graph = tf.compat.v1.get_default_graph()
  sess = tf.compat.v1.InteractiveSession()
  saver.restore(sess, 'checkpoint/model.ckpt')

#inverse_mapping_model = tf.keras.applications.ResNet50(classes=100, weights=None, input_shape=(256, 256, 3))
inverse_mapping_model = resnet.resnet_18()

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')


@tf.function
def train_step(audio, labels):
  '''
  transformed_audio = tf.signal.stft(audio, frame_length=1024, frame_step=128)
  '''

  transformed_audio = tf.expand_dims(audio, -1)

  transformed_audio = tf.stack([
    transformed_audio,
    transformed_audio,
    transformed_audio
  ], axis=-1)

  transformed_audio = tf.squeeze(transformed_audio)
  transformed_audio = tf.cast(transformed_audio, tf.float32)

  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = inverse_mapping_model(transformed_audio, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, inverse_mapping_model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, inverse_mapping_model.trainable_variables))

  train_loss(loss)

@tf.function
def test_step(audio, labels):

  #transformed_audio = tf.signal.stft(audio, frame_length=1024, frame_step=128)
  transformed_audio = tf.expand_dims(audio, -1)

  transformed_audio = tf.stack([
    transformed_audio,
    transformed_audio,
    transformed_audio
  ], axis=-1)
  

  transformed_audio = tf.squeeze(transformed_audio)
  transformed_audio = tf.cast(transformed_audio, tf.float32)

  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = inverse_mapping_model(transformed_audio, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)

batch_size = 64
test_z = (np.random.rand(batch_size, 100) * 2.) - 1.

for epoch in range(0, 100):
  train_loss.reset_states()
  

  for x in tqdm(range(0, 100)):
    n_iter = epoch*100 + x
    with tf.Graph().as_default():
      _z = (np.random.rand(batch_size, 100) * 2.) - 1.
      z = graph.get_tensor_by_name('z:0')
      G_z = graph.get_tensor_by_name('G_z:0')[:, :, 0]
      G_z_spec = graph.get_tensor_by_name('G_z_spec:0')
      _G_z, _G_z_spec = sess.run([G_z, G_z_spec], {z: _z})
    train_step(_G_z_spec, _z)
    writer.add_scalar("train_loss", train_loss.result().numpy(), n_iter)
    train_loss.reset_states()

  
  test_loss.reset_states()
  for x in tqdm(range(0, 1)):
    with tf.Graph().as_default():
      z = graph.get_tensor_by_name('z:0')
      G_z = graph.get_tensor_by_name('G_z:0')[:, :, 0]
      G_z_spec = graph.get_tensor_by_name('G_z_spec:0')
      _G_z, _G_z_spec = sess.run([G_z, G_z_spec], {z: test_z})
    test_step(_G_z_spec, test_z)
    writer.add_scalar("test_loss", test_loss.result().numpy(), n_iter)
    test_loss.reset_states()
  
  '''
  template = 'Epoch {}, Loss: {}, Test Loss: {}'
  print(template.format(epoch + 1,
                        train_loss.result(),
                        test_loss.result()
  ))
  '''
  inverse_mapping_model.save_weights("inverse_mapping_model_checkpoint/inverse_mapping_weights.ckpt")




'''
# Sample latent vectors
_z = (np.random.rand(ngenerate, 100) * 2.) - 1.

# Generate
z = graph.get_tensor_by_name('z:0')
G_z = graph.get_tensor_by_name('G_z:0')[:, :, 0]
G_z_spec = graph.get_tensor_by_name('G_z_spec:0')

start = time.time()
_G_z, _G_z_spec = sess.run([G_z, G_z_spec], {z: _z})
print('Finished! (Took {} seconds)'.format(time.time() - start))

'''

