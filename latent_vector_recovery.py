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
from scipy import optimize

from models import resnet


files = "0-9_real_quality/"


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
if len(get_available_gpus()) == 0:
  for i in range(4):
    print('WARNING: Not running on a GPU! See above for faster generation')

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

for l in inverse_mapping_model.layers:
    l.trainable = False

classifier = resnet.resnet_18(num_classes=10, activation=tf.keras.activations.softmax)
classifier.load_weights("classifier_checkpoint/model.ckpt")

for l in classifier.layers:
    l.trainable = False

classifier_middle_layer = tf.keras.Sequential(classifier.layers[0:5])



def train_generator():
    for x in range(0, 10):
        path = files+str(x)+".wav"
        path_label = path.split("/")[-1].split("_")[0]
        sr, audio = scipy.io.wavfile.read(path)
        #NOTE: If your audio is in PCM16 format or is not in the the correct sample rate, then uncomment this code
        '''
        audio = audio.astype(np.float32)
        audio = audio / 32767
        audio = audio * 1.414
        '''
        audio = librosa.core.resample(audio, sr, 16384)

        try:
            z_vector = np.load(path.split(".")[0]+".npy")
        except:
            z_vector = np.array([0]*100)
        yield z_vector, audio


@tf.function
def resnet_prediction_step(audio):

    audio = tf.signal.stft(audio, 256, 128, pad_end=True)
    audio = tf.expand_dims(audio, -1)


    audio = tf.stack([
     audio,
     audio,
     audio
    ], axis=-1)

    audio = tf.squeeze(audio)
    audio = tf.cast(audio, tf.float32)
    audio = tf.expand_dims(audio,0)

    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = inverse_mapping_model(audio, training=False)
    return predictions

def convert_to_spec(audio):
  if(len(tf.shape(audio))<4):
    audio = tf.expand_dims(audio, 0)
  audio = tf.signal.stft(audio, 256, 128, pad_end=True)
  audio = tf.expand_dims(audio, -1)


  audio = tf.stack([
    audio,
    audio,
    audio
  ], axis=-1)

  audio = tf.squeeze(audio)
  if(len(tf.shape(audio))<4):
      audio = tf.expand_dims(audio, 0)
  audio = tf.cast(audio, tf.float32)
  return audio


def middle_layer_percep_loss(reconstruction, audio):
  audio = tf.squeeze(audio)
  audio = tf.expand_dims(audio, 0)
  audio = convert_to_spec(audio)
  reconstruction = convert_to_spec(reconstruction)


  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  audio_predictions = classifier_middle_layer(audio)
  reconstructions_predictions = classifier_middle_layer(reconstruction)
  loss = mae(audio_predictions, reconstructions_predictions)
  return loss


def f_bfgs(z0, audio):
    z0_n = z0.astype(np.float32)
    z0_n = tf.convert_to_tensor(z0_n)

    audio_spec = tf.signal.stft(audio, 256, 128, pad_end=True)
    audio_spec = tf.cast(audio_spec, tf.float32)

    with tf.GradientTape() as grad_tape:
      z0_n = tf.reshape(z0_n, (1, 100))
      grad_tape.watch(z0_n)
      reconstructions_audio = wavegan(z0_n)

      reconstructions = tf.squeeze(reconstructions_audio)
      reconstructions = tf.expand_dims(reconstructions, 0)
      reconstructions = tf.signal.stft(reconstructions, 256, 128, pad_end=True)
      reconstructions = tf.cast(reconstructions, tf.float32)

      loss = 0
      percep_loss = middle_layer_percep_loss(audio, reconstructions_audio)

      loss += percep_loss

      loss += mae(audio_spec, reconstructions)
      loss += mae(audio, reconstructions_audio)

      gradients = grad_tape.gradient(loss, z0_n)
      loss = loss.numpy()
      gradients = gradients.numpy()

    return loss.astype(np.float64), gradients[0].astype(np.float64)

batch_size = 1

mae = tf.keras.losses.MeanAbsoluteError()
mse = tf.keras.losses.MeanSquaredError()
dataset = tf.data.Dataset.from_generator(train_generator, output_types=(tf.int64, tf.float32)).batch(batch_size)

for idx, (z_vector, audio) in enumerate(dataset):

    predicted_z = resnet_prediction_step(audio)

    audio = tf.squeeze(audio)
    audio = tf.expand_dims(audio, 0)
    _z = _z.numpy()
    #Uncomment this line for to use the hybrid method
    #prob = optimize.minimize(f_bfgs, _z, args=(audio,), tol=1e-8, jac=True, method='L-BFGS-B', options={'maxiter': 200})
    _z = prob.x.astype(np.float32)

    predictions = tf.convert_to_tensor(_z)
    predictions = tf.cast(predictions, tf.float32)
    predictions = tf.expand_dims(predictions, 0)
    _z = predictions
    print('n_iters = %3d, f = %.3f' % (prob.nit, prob.fun))


    reconstructions = wavegan(_z)
    scipy.io.wavfile.write("grad_only_real/"+str(idx)+".wav", 16384, reconstructions.numpy()[0])
    np.save("grad_only_real/"+str(idx), _z.numpy()[0])
