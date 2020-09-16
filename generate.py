import numpy as np
import PIL.Image
import time as time
import tensorflow as tf
from tensorflow.python.client import device_lib
import tensorflow_hub as hub
from tqdm import tqdm
import scipy.io.wavfile

from models import resnet
from tensorboardX import SummaryWriter

# CHANGE THIS to use a different dataset
dataset = 'digits' # one of 'digits', 'speech', 'birds', 'drums', 'piano'

# Confirm GPU is running

#writer = SummaryWriter("./logs")

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
#inverse_mapping_model.load_weights("inverse_mapping_model_checkpoint/inverse_mapping_weights_1d_mse.ckpt")

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')


@tf.function
def train_step(audio, labels):

  audio = tf.signal.stft(audio, 256, 128, pad_end=True)
  
  audio = tf.expand_dims(audio, -1)

  audio = tf.stack([
    audio,
    audio,
    audio
  ], axis=-1)

  audio = tf.squeeze(audio)
  audio = tf.cast(audio, tf.float32)

  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = inverse_mapping_model(audio, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, inverse_mapping_model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, inverse_mapping_model.trainable_variables))

  train_loss(loss)

@tf.function
def test_step(audio, labels):
  
  audio = tf.signal.stft(audio, 256, 128, pad_end=True)
  audio = tf.expand_dims(audio, -1)

  
  audio = tf.stack([
    audio,
    audio,
    audio
  ], axis=-1)
  

  audio = tf.squeeze(audio)
  audio = tf.cast(audio, tf.float32)

  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = inverse_mapping_model(audio, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)

@tf.function
def sample_step(audio):
  
  audio = tf.signal.stft(audio, 256, 128, pad_end=True)
  audio = tf.expand_dims(audio, -1)


  audio = tf.stack([
    audio,
    audio,
    audio
  ], axis=-1)
  

  audio = tf.squeeze(audio)
  audio = tf.cast(audio, tf.float32)

  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = inverse_mapping_model(audio, training=False)
  return predictions


batch_size = 64
test_z = (np.random.rand(batch_size, 100) * 2.) - 1.
'''
try:
  starting_epoch = np.load("inverse_mapping_model_checkpoint/epoch.npy")[0]
except:
  starting_epoch = 0
'''
starting_epoch = 0
for epoch in range(starting_epoch, 100):
  train_loss.reset_states()
  for x in tqdm(range(0, 300)):
    n_iter = epoch*300 + x
    with tf.Graph().as_default():
      _z = (np.random.rand(batch_size, 100) * 2.) - 1.
      z = graph.get_tensor_by_name('z:0')
      G_z = graph.get_tensor_by_name('G_z:0')[:, :, 0]
      G_z_spec = graph.get_tensor_by_name('G_z_spec:0')
      _G_z, _G_z_spec = sess.run([G_z, G_z_spec], {z: _z})
    print(_G_z.shape)
    train_step(_G_z, _z)
    #writer.add_scalar("2d_simple_spec_train_mse_resnet_18", train_loss.result().numpy(), n_iter)
    train_loss.reset_states()

  
  test_loss.reset_states()
  for x in tqdm(range(0, 1)):
    with tf.Graph().as_default():
      z = graph.get_tensor_by_name('z:0')
      G_z = graph.get_tensor_by_name('G_z:0')[:, :, 0]
      G_z_spec = graph.get_tensor_by_name('G_z_spec:0')
      _G_z, _G_z_spec = sess.run([G_z, G_z_spec], {z: test_z})
    
    test_step(_G_z, test_z)
    #writer.add_scalar("2d_simple_spec_test_mse_resnet_18", test_loss.result().numpy(), n_iter)
    test_loss.reset_states()

  num_samples = 2
  sample_z = (np.random.rand(num_samples, 100) * 2.) - 1.
  with tf.Graph().as_default():
      z = graph.get_tensor_by_name('z:0')
      G_z = graph.get_tensor_by_name('G_z:0')[:, :, 0]
      G_z_spec = graph.get_tensor_by_name('G_z_spec:0')
      _G_z, _G_z_spec = sess.run([G_z, G_z_spec], {z: sample_z})
  
  original_generated_sound = _G_z

  predictions = sample_step(_G_z)
  with tf.Graph().as_default():
      z = graph.get_tensor_by_name('z:0')
      G_z = graph.get_tensor_by_name('G_z:0')[:, :, 0]
      G_z_spec = graph.get_tensor_by_name('G_z_spec:0')
      _G_z, _G_z_spec = sess.run([G_z, G_z_spec], {z: predictions.numpy()})
    
  inversed_mapped_generated_sound = _G_z

  for x in range(0, num_samples):
    scipy.io.wavfile.write("waves/original_"+str(epoch)+"sample_"+str(x)+".wav", 16384, original_generated_sound[x])
    scipy.io.wavfile.write("waves/inverse_mapping_"+str(epoch)+"sample_"+str(x)+".wav", 16384, inversed_mapped_generated_sound[x])
  

  #inverse_mapping_model.save_weights("inverse_mapping_model_checkpoint/inverse_mapping_weights_2d_mse.ckpt")
  np.save("inverse_mapping_model_checkpoint/epoch.npy", np.zeros(1)*epoch)




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

