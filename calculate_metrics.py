import numpy as np
import PIL.Image
import time as time
import tensorflow as tf
from tensorflow.python.client import device_lib
import tensorflow_hub as hub
from tqdm import tqdm
import scipy.io.wavfile
import glob
import librosa
import random
import time
from scipy import optimize

from models import resnet
from tensorboardX import SummaryWriter
import tensorflow_datasets



synth = False
gradient = False
inverse_mapping = True

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

string_to_label = {
  "Zero": 0,
  "One": 1,
  "Two": 2,
  "Three": 3,
  "Four": 4,
  "Five": 5,
  "Six": 6,
  "Seven": 7,
  "Eight": 8,
  "Nine": 9
}



inverse_mapping_model = resnet.resnet_18()
inverse_mapping_model.load_weights("inverse_mapping_model_checkpoint/generated_and_real_training.ckpt")

classifier = resnet.resnet_18(num_classes=10, activation=tf.keras.activations.softmax)
classifier.load_weights("classifier_checkpoint/model.ckpt")

for l in classifier.layers:
    l.trainable = False

classifier_middle_layer = tf.keras.Sequential(classifier.layers[0:5])

for l in inverse_mapping_model.layers:
    l.trainable = False

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_ssim_loss = tf.keras.metrics.Mean(name='test_ssim_loss')
test_acc = tf.keras.metrics.Accuracy(name='test_acc')



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
  if(len(tf.shape(audio))<4):
      audio = tf.expand_dims(audio, 0)
  audio = tf.cast(audio, tf.float32)

  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = classifier(audio, training=False)

  test_acc(tf.argmax(predictions,1), tf.argmax(labels,1))
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

valid_files = glob.glob("datasets/sc09/valid/*")
def valid_generator():
  for x in range(0, len(valid_files)):
    path = valid_files[x]
    path_label = path.split("/")[-1].split("_")[0]
    label = np.zeros(10)
    label[string_to_label[path_label]] = 1
    sr, audio = scipy.io.wavfile.read(path)
    audio = audio.astype(np.float32)
    audio = audio / 32767
    audio = audio * 1.414
    audio = librosa.core.resample(audio, 16000, 16384)
    data = np.zeros(16384)
    data[0:min(audio.shape[0], 16384)] = audio[0:min(audio.shape[0], 16384)]

    yield np.zeros(61), data

mse = tf.keras.losses.MeanSquaredError()
mae = tf.keras.losses.MeanAbsoluteError()

def get_loss(raw_reconstruction, audio):
  if(len(tf.shape(audio))<4):
    audio = tf.expand_dims(audio, 0)
  audio = tf.reshape(audio, (1, -1))

  if(len(tf.shape(raw_reconstruction))<4):
    raw_reconstruction = tf.expand_dims(raw_reconstruction, 0)
  raw_reconstruction = tf.reshape(raw_reconstruction, (1, -1))

  return mse(raw_reconstruction, audio)

def get_ssim_loss(raw_reconstruction, audio):
  if(len(tf.shape(audio))<4):
    audio = tf.expand_dims(audio, 0)
  audio = tf.signal.stft(audio, 256, 128, pad_end=True)

  if(len(tf.shape(raw_reconstruction))<4):
    raw_reconstruction = tf.expand_dims(raw_reconstruction, 0)
  raw_reconstruction = tf.signal.stft(raw_reconstruction, 256, 128, pad_end=True)

  audio = tf.cast(audio, tf.float32)
  raw_reconstruction = tf.cast(raw_reconstruction, tf.float32)
  min_val = np.amin([audio.numpy(), raw_reconstruction.numpy()])
  max_val = np.amax([audio.numpy(), raw_reconstruction.numpy()])

  ssim = tf.image.ssim(tf.expand_dims(tf.squeeze(raw_reconstruction), -1), tf.expand_dims(tf.squeeze(audio), -1), max_val - min_val)
  return ssim



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
    audio_spec = tf.signal.stft(audio, 256, 128, pad_end=True)
    audio_spec = tf.cast(audio_spec, tf.float32)

    with tf.GradientTape() as grad_tape:
      z0_n = tf.convert_to_tensor(z0_n)
      z0_n = tf.reshape(z0_n, (1, 100))
      grad_tape.watch(z0_n)
      reconstructions_audio = wavegan(z0_n)

      reconstructions = tf.squeeze(reconstructions_audio)
      reconstructions = tf.expand_dims(reconstructions, 0)
      reconstructions = tf.signal.stft(reconstructions, 256, 128, pad_end=True)
      reconstructions = tf.cast(reconstructions, tf.float32)

      loss = 0
      percep_loss = middle_layer_percep_loss(audio, reconstructions_audio)
      loss += percep_loss*0.002

      loss += mae(reconstructions, audio_spec)
      loss += mae(reconstructions_audio, audio)

      gradients = grad_tape.gradient(loss, z0_n)
      loss = loss.numpy()
      gradients = gradients.numpy()

    return loss.astype(np.float64), gradients[0].astype(np.float64)




dataset = tf.data.Dataset.from_generator(valid_generator, output_types=(tf.int64, tf.float32)).batch(1)

test_loss.reset_states()
test_ssim_loss.reset_states()
test_acc.reset_states()
preds = []
tf.random.set_seed(0)
np.random.seed(0)
for idx, (label, audio) in enumerate(dataset):
    print(idx)
    if idx > 500:
      break
    batch_size = 1
    if synth:
      sample_z = (tf.random.uniform([batch_size, 100])*2.0) - 1.0
      #sample_z = np.load("sample_tensor_vals.npy")
      #sample_z = tf.convert_to_tensor(sample_z)
      audio = wavegan(sample_z)
      audio = tf.squeeze(audio)
      audio = tf.expand_dims(audio, 0)
    audio_spec = convert_to_spec(audio)
    predictions = inverse_mapping_model(audio_spec)
    if not inverse_mapping:
      predictions = (tf.random.stateless_uniform([batch_size, 100], [np.random.randint(0,10000),np.random.randint(0,10000)])*2.0) - 1.0

    if gradient:
        audio_spec = tf.signal.stft(audio, 256, 128, pad_end=True)
        audio_spec = tf.cast(audio_spec, tf.float32)

        _z = predictions
        start = time.time()


        #START NEW CODE

        if synth:
          print("Before Gradient Descent", ': ',  mae(sample_z, predictions.numpy()).numpy())
        _z = _z.numpy()
        prob = optimize.minimize(f_bfgs, _z, args=(audio,), tol=1e-8, jac=True, method='L-BFGS-B', options={'maxiter': 50000})
        _z = prob.x.astype(np.float64)

        predictions = tf.convert_to_tensor(_z)
        predictions = tf.cast(predictions, tf.float32)
        predictions = tf.expand_dims(predictions, 0)
        print('n_iters = %3d, f = %.3f' % (prob.nit, prob.fun))
        if synth:
          print("After Gradient Descent", ': ',  mae(sample_z, predictions.numpy()).numpy())


    reconstruction = wavegan(predictions)
    reconstruction = tf.squeeze(reconstruction)
    reconstruction = tf.expand_dims(reconstruction, 0)

    test_loss(get_loss(reconstruction, audio))
    test_ssim_loss(get_ssim_loss(reconstruction, audio))

    pred = test_step(reconstruction, label)

    preds.append(pred)



preds = np.concatenate(preds, 0)
scores = []
splits = 2
for i in range(splits):
    part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
    kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
    kl = np.mean(np.sum(kl, 1))
    scores.append(np.exp(kl))
print(np.mean(scores))
print(np.std(scores))
print("MSE: ", test_loss.result().numpy())
print("SSIM: ", test_ssim_loss.result().numpy())

print("\n\n\n\n\n")
print("Accuracy: " + str(test_acc.result().numpy()))
print("\n\n\n\n\n")

test_loss.reset_states()
test_acc.reset_states()



