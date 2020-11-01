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



batch_size = 64
for epoch in range(0, 2):
    print(epoch)
    sample_z = (tf.random.uniform([batch_size, 100])*2.0) - 1.0
    _G_z = wavegan(sample_z).numpy()
    for x in range(0, batch_size):
        scipy.io.wavfile.write("fake_dataset/epoch_"+str(epoch)+"_x_"+str(x)+".wav", 16384, _G_z[x])
        np.save("fake_dataset/z_vector_epoch_"+str(epoch)+"_x_"+str(x), sample_z.numpy()[x])


