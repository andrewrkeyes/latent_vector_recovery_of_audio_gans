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
  sess = tf.compat.v1.InteractiveSession()
  saver.restore(sess, 'checkpoint/model.ckpt')
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
inverse_mapping_model.load_weights("inverse_mapping_model_checkpoint/generated_and_real_training.ckpt")

@tf.function
def sample_step(audio):
  audio = tf.squeeze(audio)
  audio = tf.expand_dims(audio,0)
  audio = tf.signal.stft(audio, 256, 128, pad_end=True)
  
  audio = tf.expand_dims(audio, -1)


  audio = tf.stack([
    audio,
    audio,
    audio
  ], axis=-1)
  

  audio = tf.squeeze(audio)
  audio = tf.expand_dims(audio, 0)
  audio = tf.cast(audio, tf.float32)

  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = inverse_mapping_model(audio, training=False)

  return predictions


batch_size = 6

def load_real_audio(path):
    sr, data = scipy.io.wavfile.read(path)
    return data

real_audio_path = "real_audio_cut.wav"
batch = np.zeros((1, 16384))
batch[0] = load_real_audio(real_audio_path)
print(batch.shape)

predictions = sample_step(batch)
_G_z = wavegan(predictions)

inversed_mapped_generated_sound = _G_z

scipy.io.wavfile.write("real_audio_waves/wavenet_inverse_250_attempt.wav", 16384, inversed_mapped_generated_sound.numpy()[0])


#inverse_mapping_model.save_weights("inverse_mapping_model_checkpoint/inverse_mapping_weights_2d_mse.ckpt")



