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

batch_size = 64
for epoch in range(0, 39):
    with tf.Graph().as_default():
      _z = (np.random.rand(batch_size, 100) * 2.) - 1.
      z = graph.get_tensor_by_name('z:0')
      G_z = graph.get_tensor_by_name('G_z:0')[:, :, 0]
      G_z_spec = graph.get_tensor_by_name('G_z_spec:0')
      _G_z, _G_z_spec = sess.run([G_z, G_z_spec], {z: _z})

    for x in range(0, batch_size):
        scipy.io.wavfile.write("fake_audio/valid/"+str(epoch)+"_"+str(x)+".wav", 16384, _G_z[x])