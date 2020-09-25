



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

from models import resnet
from tensorboardX import SummaryWriter
import tensorflow_datasets

# CHANGE THIS to use a different dataset
dataset = 'digits' # one of 'digits', 'speech', 'birds', 'drums', 'piano'

# Confirm GPU is running

writer = SummaryWriter("./logs/deepfake")

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
if len(get_available_gpus()) == 0:
  for i in range(4):
    print('WARNING: Not running on a GPU! See above for faster generation')


classifier = resnet.resnet_18(num_classes=100, activation=None)
classifier.load_weights("inverse_mapping_model_checkpoint/combined_loss_model.ckpt")

for l in classifier.layers:
    l.trainable = False

classifier = tf.keras.Sequential([
    classifier,
    tf.keras.layers.Dense(2)
])


loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc = tf.keras.metrics.Accuracy(name='train_acc')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_acc = tf.keras.metrics.Accuracy(name='test_acc')



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
    predictions = tf.keras.activations.softmax(classifier(audio, training=True))
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, classifier.trainable_variables)
  optimizer.apply_gradients(zip(gradients, classifier.trainable_variables))

  train_loss(loss)
  train_acc(tf.argmax(predictions,1), tf.argmax(labels,1))

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
  predictions = tf.keras.activations.softmax(classifier(audio, training=False))
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_acc(tf.argmax(predictions,1), tf.argmax(labels,1))


batch_size = 64
train_files = glob.glob("datasets/*/train/*")
random.shuffle(train_files)
def train_generator():
  for x in range(0, len(train_files)):
    path = train_files[x]
    path_label = path.split("/")[-1].split("_")[0]
    label = np.zeros(2)
    if "sc09" in path:
        label[0] = 1
    else:
        label[1] = 1

    sr, audio = scipy.io.wavfile.read(path)
    if "sc09" in path:
        audio = audio.astype(np.float32)
        audio = audio / 32767
        audio = audio * 1.414
        audio = librosa.core.resample(audio, 16000, 16384)
    data = np.zeros(16384)
    data[0:audio.shape[0]] = audio

    yield label, data

valid_files = glob.glob("datasets/*/valid/*")
def valid_generator():
  for x in range(0, len(train_files)):
    path = train_files[x]
    path_label = path.split("/")[-1].split("_")[0]
    label = np.zeros(2)
    if "sc09" in path:
        label[0] = 1
    else:
        label[1] = 1

    sr, audio = scipy.io.wavfile.read(path)
    if "sc09" in path:
        audio = audio.astype(np.float32)
        audio = audio / 32767
        audio = audio * 1.414
        audio = librosa.core.resample(audio, 16000, 16384)
    data = np.zeros(16384)
    data[0:audio.shape[0]] = audio

    yield label, data


starting_epoch = 0
n_iter_train = 0
n_iter_test = 0
best_val = 0.0
for epoch in range(starting_epoch, 100):
  dataset = tf.data.Dataset.from_generator(train_generator, output_types=(tf.int64, tf.float32)).shuffle(500).batch(64)
  train_loss.reset_states()
  train_acc.reset_states()
  for label, audio in dataset:
    train_step(audio, label)

    writer.add_scalar("train_loss", train_loss.result().numpy(), n_iter_train)
    writer.add_scalar("train_acc", train_acc.result().numpy(), n_iter_train)
    train_loss.reset_states()
    train_acc.reset_states()
    n_iter_train += 1

  dataset = tf.data.Dataset.from_generator(valid_generator, output_types=(tf.int64, tf.float32)).batch(64)
  test_loss.reset_states()
  test_acc.reset_states()
  for label, audio in dataset:
    test_step(audio, label)

  writer.add_scalar("test_loss", test_loss.result().numpy(), epoch)
  writer.add_scalar("test_acc", test_acc.result().numpy(), epoch)
  if(test_acc.result().numpy() > best_val):
    best_val = test_acc.result().numpy()
    classifier.save_weights("deepfake_classifier_checkpoint/model.ckpt")
  test_loss.reset_states()
  test_acc.reset_states()
  n_iter_test += 1