import numpy as np
import PIL.Image
import time as time
import tensorflow as tf
from tensorflow.python.client import device_lib
import tensorflow_hub as hub
from tqdm import tqdm
import scipy.io.wavfile
import glob
import random
import librosa

from models import resnet
from tensorboardX import SummaryWriter

# CHANGE THIS to use a different dataset
dataset = 'digits' # one of 'digits', 'speech', 'birds', 'drums', 'piano'

# Confirm GPU is running

writer = SummaryWriter("./logs/generated_and_real_training")

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
if len(get_available_gpus()) == 0:
  for i in range(4):
    print('WARNING: Not running on a GPU! See above for faster generation')


'''
with tf.Graph().as_default():
  #tf.compat.v1.reset_default_graph()
  saver = tf.compat.v1.train.import_meta_graph('checkpoint/infer.meta')
  graph = tf.compat.v1.get_default_graph()
  input_graph_def = graph.as_graph_def()
  sess = tf.compat.v1.Session()
  saver.restore(sess, 'checkpoint/model.ckpt')
 
  output_node_names="G_z"
  output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
              sess, # The session
              input_graph_def, # input_graph_def is useful for retrieving the nodes 
              output_node_names.split(",")  
  )
  output_graph="checkpoint_frozen/saved_model.pb"
  with tf.compat.v1.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
 
  sess.close()
'''

def wrap_frozen_graph(graph_def, inputs, outputs):
  def _imports_graph_def():
    tf.compat.v1.import_graph_def(graph_def, name="")
  wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
  import_graph = wrapped_import.graph
  return wrapped_import.prune(
      tf.nest.map_structure(import_graph.as_graph_element, inputs),
      tf.nest.map_structure(import_graph.as_graph_element, outputs))

graph_def = tf.compat.v1.GraphDef()
path = "checkpoint_frozen/saved_model.pb"

loaded = graph_def.ParseFromString(open(path,'rb').read())
wavegan = wrap_frozen_graph(
    graph_def, inputs='z:0',
    outputs='G_z:0')




inverse_mapping_model = resnet.resnet_18()
#inverse_mapping_model.load_weights("inverse_mapping_model_checkpoint/inverse_mapping_weights_2d_mse.ckpt")
classifier = resnet.resnet_18(num_classes=10,activation=tf.keras.activations.softmax)
for l in classifier.layers:
    l.trainable = False

classifier_models = list()
classifier.load_weights("classifier_checkpoint/model.ckpt")
perceptual_layers = [0, 3, 4, 5, 6, 8]
for layer_index in range(len(perceptual_layers)):
  classifier_models.append(tf.keras.Sequential(classifier.layers[0:layer_index]))


train_files = glob.glob("datasets/sc09/train/*")
random.shuffle(train_files)
def train_generator():
  for x in range(0, len(train_files)):
    path = train_files[x]
    path_label = path.split("/")[-1].split("_")[0]
    sr, audio = scipy.io.wavfile.read(path)
    if "sc09" in path:
        audio = audio.astype(np.float32)
        audio = audio / 32767
        audio = audio * 1.414
        audio = librosa.core.resample(audio, 16000, 16384)
    audio = audio[0:16384]
    data = np.zeros(16384)
    data[0:audio.shape[0]] = audio

    yield 0, data


valid_files = glob.glob("sc09/valid/*")
def valid_generator():
  for x in range(0, len(train_files)):
    path = train_files[x]
    path_label = path.split("/")[-1].split("_")[0]

    sr, audio = scipy.io.wavfile.read(path)
    audio = audio.astype(np.float32)
    audio = audio / 32767
    audio = audio * 1.414
    audio = librosa.core.resample(audio, 16000, 16384)
    data = np.zeros(16384)
    data[0:audio.shape[0]] = audio

    yield 0, data

loss_object = tf.keras.losses.MeanSquaredError()
mse = tf.keras.losses.MeanSquaredError()
mae = tf.keras.losses.MeanAbsoluteError()

optimizer = tf.keras.optimizers.Adam()

train_z_loss = tf.keras.metrics.Mean(name='train_z_loss')
train_percep_loss = tf.keras.metrics.Mean(name='train_percep_loss')
train_percep_loss_real = tf.keras.metrics.Mean(name='train_percep_real')
train_loss = tf.keras.metrics.Mean(name='train_loss')

test_z_loss = tf.keras.metrics.Mean(name='test_z_loss')
test_percep_loss = tf.keras.metrics.Mean(name='test_percep_loss')
test_percep_loss_real = tf.keras.metrics.Mean(name='test_percep_loss_real')
test_loss = tf.keras.metrics.Mean(name='test_loss')


def compute_perceptual_loss(original_audio_spec, inverse_audio):
  inverse_audio = tf.signal.stft(inverse_audio, 256, 128, pad_end=True)
  
  inverse_audio = tf.expand_dims(inverse_audio, -1)

  inverse_audio = tf.stack([
    inverse_audio,
    inverse_audio,
    inverse_audio
  ], axis=-1)

  inverse_audio = tf.squeeze(inverse_audio)
  inverse_audio = tf.cast(inverse_audio, tf.float32)
  #print("Inverse Audio shape: "+str(inverse_audio.shape))
  #print("Original_audio_shape: "+str(original_audio_spec.shape))


  sr_embd = classifier(original_audio_spec)
  hr_embd = classifier(inverse_audio)
  embedding_loss = mae(sr_embd, hr_embd)

  total_loss = 0
  for x in range(len(perceptual_layers)):
      sr_features = classifier_models[x](original_audio_spec)
      hr_features = classifier_models[x](inverse_audio)
      loss = mse(sr_features, hr_features)
      total_loss += loss

  total_loss = total_loss / len(perceptual_layers)
  return total_loss, embedding_loss


@tf.function
def train_step(audio, labels, compute_z_loss=True):
  audio = tf.squeeze(audio)
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
    '''
    with tf.Graph().as_default():
      z = graph.get_tensor_by_name('z:0')
      G_z = graph.get_tensor_by_name('G_z:0')[:, :, 0]
      G_z_spec = graph.get_tensor_by_name('G_z_spec:0')
      _G_z, _G_z_spec = sess.run([G_z, G_z_spec], {z: predictions.numpy()})
    '''
    _G_z = wavegan(predictions)
    _G_z = tf.squeeze(_G_z)

    #print("Before passing: "+str(audio.shape))
    if compute_z_loss:
      z_loss = loss_object(labels, predictions)
    percep_loss, _ = compute_perceptual_loss(audio, _G_z)
    if compute_z_loss:
      loss = z_loss + percep_loss
    else:
      loss = percep_loss


  gradients = tape.gradient(loss, inverse_mapping_model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, inverse_mapping_model.trainable_variables))

  if compute_z_loss:
    train_z_loss(z_loss)
    train_percep_loss(percep_loss)
    train_loss(loss)
  else:
    train_percep_loss_real(percep_loss)
  
@tf.function
def test_step(audio, labels, compute_z_loss=True):

  audio = tf.squeeze(audio)
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
  predictions = inverse_mapping_model(audio, training=True)
  '''
  with tf.Graph().as_default():
    z = graph.get_tensor_by_name('z:0')
    G_z = graph.get_tensor_by_name('G_z:0')[:, :, 0]
    G_z_spec = graph.get_tensor_by_name('G_z_spec:0')
    _G_z, _G_z_spec = sess.run([G_z, G_z_spec], {z: predictions.numpy()})
  '''

  _G_z = wavegan(predictions)
  _G_z = tf.squeeze(_G_z)
  if compute_z_loss:
    z_loss = loss_object(labels, predictions)
  percep_loss, _ = compute_perceptual_loss(audio, _G_z)
  if compute_z_loss:
    loss = z_loss + percep_loss
  else:
    loss = percep_loss

  if compute_z_loss:
    test_z_loss(z_loss)
    test_percep_loss(percep_loss)
    test_loss(loss)
  else:
    test_percep_loss_real(percep_loss)

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
test_z = (tf.random.uniform([batch_size, 100])*2.0) - 1.0

'''
try:
  starting_epoch = np.load("inverse_mapping_model_checkpoint/epoch.npy")[0]
except:
  starting_epoch = 0
'''
n_iter = 0
starting_epoch = 0
for epoch in range(starting_epoch, 100):
  train_loss.reset_states()
  train_z_loss.reset_states()
  train_percep_loss.reset_states()
  train_percep_loss_real.reset_states()

  '''
  for x in tqdm(range(0, 2)):
    n_iter = epoch*300 + x
    with tf.Graph().as_default():
      _z = (np.random.rand(batch_size, 100) * 2.) - 1.
      z = graph.get_tensor_by_name('z:0')
      G_z = graph.get_tensor_by_name('G_z:0')[:, :, 0]
      G_z_spec = graph.get_tensor_by_name('G_z_spec:0')
      _G_z, _G_z_spec = sess.run([G_z, G_z_spec], {z: _z})
   #_z = (np.random.rand(batch_size, 100) * 2.) - 1.0
  '''
  
  dataset = tf.data.Dataset.from_generator(train_generator, output_types=(tf.int64, tf.float32)).shuffle(500).batch(batch_size)
  
  for label, audio in dataset:
    train_step(audio, None, compute_z_loss=False)
    writer.add_scalar("train_percep_loss_real", train_percep_loss_real.result().numpy(), n_iter)
    train_percep_loss_real.reset_states()


    _z = (tf.random.uniform([batch_size, 100])*2.0) - 1.0
    _G_z = wavegan(_z)
    

    train_step(_G_z, _z, compute_z_loss=True)
    writer.add_scalar("train_combined_loss", train_loss.result().numpy(), n_iter)
    writer.add_scalar("train_z_loss", train_z_loss.result().numpy(), n_iter)
    writer.add_scalar("train_percep_loss", train_percep_loss.result().numpy(), n_iter)

    train_loss.reset_states()
    train_z_loss.reset_states()
    train_percep_loss.reset_states()
    n_iter += 1
    

  
  test_loss.reset_states()
  for x in tqdm(range(0, 1)):
    '''
    with tf.Graph().as_default():
      z = graph.get_tensor_by_name('z:0')
      G_z = graph.get_tensor_by_name('G_z:0')[:, :, 0]
      G_z_spec = graph.get_tensor_by_name('G_z_spec:0')
      _G_z, _G_z_spec = sess.run([G_z, G_z_spec], {z: test_z})
    '''
    _G_z = wavegan(test_z)
    
    
    test_step(_G_z, test_z)
    writer.add_scalar("test_combined_loss", test_loss.result().numpy(), epoch)
    writer.add_scalar("test_z_loss", test_z_loss.result().numpy(), epoch)
    writer.add_scalar("test_percep_loss", test_percep_loss.result().numpy(), epoch)
    test_loss.reset_states()
    test_z_loss.reset_states()
    test_percep_loss.reset_states()

  dataset = tf.data.Dataset.from_generator(valid_generator, output_types=(tf.int64, tf.float32)).shuffle(500).batch(batch_size)
  y = 0
  for label, audio in dataset:
    test_step(audio, None, compute_z_loss=False)
  writer.add_scalar("test_percep_loss_real", test_percep_loss_real.result().numpy(), epoch)
  test_percep_loss_real.reset_states()

  num_samples = 2
  sample_z = (tf.random.uniform([num_samples, 100])*2.0) - 1.0

  '''
  with tf.Graph().as_default():
      z = graph.get_tensor_by_name('z:0')
      G_z = graph.get_tensor_by_name('G_z:0')[:, :, 0]
      G_z_spec = graph.get_tensor_by_name('G_z_spec:0')
      _G_z, _G_z_spec = sess.run([G_z, G_z_spec], {z: sample_z})
  '''
  _G_z = wavegan(sample_z)
  _G_z = tf.squeeze(_G_z)
  original_generated_sound = _G_z

  predictions = sample_step(_G_z)
  '''
  with tf.Graph().as_default():
      z = graph.get_tensor_by_name('z:0')
      G_z = graph.get_tensor_by_name('G_z:0')[:, :, 0]
      G_z_spec = graph.get_tensor_by_name('G_z_spec:0')
      _G_z, _G_z_spec = sess.run([G_z, G_z_spec], {z: predictions.numpy()})
  '''
  _G_z = wavegan(predictions)
  _G_z = tf.squeeze(_G_z)
    
  inversed_mapped_generated_sound = _G_z

  for x in range(0, num_samples):
    scipy.io.wavfile.write("waves/original_"+str(epoch)+"sample_"+str(x)+".wav", 16384, original_generated_sound.numpy()[x])
    scipy.io.wavfile.write("waves/inverse_mapping_"+str(epoch)+"sample_"+str(x)+".wav", 16384, inversed_mapped_generated_sound.numpy()[x])
  

  inverse_mapping_model.save_weights("inverse_mapping_model_checkpoint/generated_and_real_training.ckpt")
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

