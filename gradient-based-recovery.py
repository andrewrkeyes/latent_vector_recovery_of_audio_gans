import numpy as np
import tensorflow.compat.v1 as tf
import random
import os
import time
import math
import glob
from scipy.io import wavfile

from PIL import Image
import librosa


tf.disable_eager_execution()

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
print("Current dir: ", os.getcwd())

BATCH_SIZE = 50
np.random.seed(1)

"""
with tf.Graph().as_default():
  #tf.compat.v1.reset_default_graph()
  saver = tf.compat.v1.train.import_meta_graph('infer.meta')
  graph = tf.compat.v1.get_default_graph()
  input_graph_def = graph.as_graph_def()
  sess = tf.compat.v1.Session()
  saver.restore(sess, 'model.ckpt')
  
  output_node_names="G_z"
  output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
              sess, # The session
              input_graph_def, # input_graph_def is useful for retrieving the nodes 
              output_node_names.split(",")  
  )
  output_graph="saved_model.pb"
  with tf.compat.v1.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
  
  sess.close()
"""
def wrap_frozen_graph(graph_def, inputs, outputs):
  def _imports_graph_def():
    tf.import_graph_def(graph_def, name="")
  wrapped_import = tf.wrap_function(_imports_graph_def, [])
  import_graph = wrapped_import.graph
  return wrapped_import.prune(
      tf.nest.map_structure(import_graph.as_graph_element, inputs),
      tf.nest.map_structure(import_graph.as_graph_element, outputs))

graph_def = tf.GraphDef()
path = "saved_model.pb"

loaded = graph_def.ParseFromString(open(path,'rb').read())
wavegan = wrap_frozen_graph(
    graph_def, inputs='z:0',
    outputs='G_z:0')
    
      
#tf.reset_default_graph()
#saver = tf.train.import_meta_graph('infer.meta')
#graph = tf.get_default_graph()
#sess = tf.InteractiveSession()
#saver.restore(sess, 'model.ckpt')


def convert_to_spectogram(audio):
    audio = tf.squeeze(audio)
    audio = tf.signal.stft(audio, 256, 128, pad_end=True)
    audio = tf.expand_dims(audio, -1)

    audio = tf.stack([
        audio,
        audio,
        audio
    ], axis=-1)
    
    audio = tf.squeeze(audio)
    #audio = tf.cast(audio, tf.float32)
    #audio = tf.reshape(audio, [1,128,129,3])
    return audio


def generate_fake_audio():

    # Create 50 random latent vectors z
    _z = (np.random.rand(BATCH_SIZE, 100) * 2.) - 1
    # Synthesize G(z)
    z = graph.get_tensor_by_name('z:0')
    G_z = graph.get_tensor_by_name('G_z:0')
    _G_z = sess.run(G_z, {z: _z})

    for x in range(0, BATCH_SIZE):
        spectogram = convert_to_spectogram(_G_z[x])
        spectogram = sess.run(spectogram)
        spectogram = (spectogram * 255.).astype(np.uint8)
        print("Spectogram shape: ", spectogram.shape)
        # Image.fromarray(spectogram).save("fake_audio/wavegan_digit/"+"audio_"+str(x)+".png")
        #wavfile.write("fake_audio/wavegan_digit/"+"audio_"+str(x)+".wav", 16384, _G_z[x])


train_path = glob.glob("/home/nbayat5/wavegan_inverse_mapping/fake_audio/wavegan_digit/*.wav")


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, train_path, batch_size=1):
        self.batch_size = batch_size
        self.size = len(train_path)

    def __len__(self):
        return math.floor(self.size / self.batch_size)

    def __getitem__(self, idx):
        path = train_path[idx]
        path_label = path.split("/")[-1].split(".")[0]
        sr, audio = wavfile.read(path)
        audio = audio.astype(np.float32)
        audio = audio / 32767
        audio = audio * 1.414
        audio = librosa.core.resample(audio, 16000, 16384)
        audio = audio[0:16384]
        data = np.zeros(16384)
        data[0:audio.shape[0]] = audio
        return path_label, data.astype(np.float32)




def recover_latent_vector():
    MAE = tf.keras.losses.MeanAbsoluteError()
    data_generator = DataGenerator(train_path)
    # dataset = tf.data.Dataset.from_generator(train_generator, output_types=(tf.string, tf.float32)).batch(1)
    print("Number of audio: ", len(train_path))
    num_iters = 200
    z_vectors = []
    index = 0
    while index < BATCH_SIZE:
        labels, audio = data_generator.__getitem__(index)
        print("Batch: ", index)
        original_spectogram = convert_to_spectogram(audio)
        original_spectogram = tf.Variable(original_spectogram, tf.float32)
        original_spectogram = tf.cast(original_spectogram, tf.float32)

        print("Batch: {} original spectogram: {} - {}".format(index, original_spectogram.shape, type(original_spectogram)))

        _z = tf.convert_to_tensor((np.random.rand(1, 100) * 2.) - 1)
        G_z = wavegan(_z)
        
        _z = tf.Variable(_z, tf.float32)
        _z = tf.cast(_z, tf.float32)
        predicted_spectogram = convert_to_spectogram(G_z)
        print("predicted_spectogram: ", predicted_spectogram.shape, type(predicted_spectogram))


        loss = MAE(original_spectogram, predicted_spectogram)
        
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.constant(0.01)
        opt = tf.train.AdamOptimizer(learning_rate)
        train = opt.minimize(loss, var_list=_z, global_step=global_step)
        
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())


        #
        #start_time = time.time()
        #for i in range(num_iters):
            #print("step: ", i + 1)
            #_, loss_value, zp_val, eta = sess.run((train, loss, _z, learning_rate))
        #
        #     """Uncomment below lines if you want to add stochastic clipping"""
        #     # stochastic clipping
        #     # zp_val[zp_val > 1] = random.uniform(-1, 1)
        #     # zp_val[zp_val < -1] = random.uniform(-1, 1)
        #
        # print("-" * 50)
        # print("Time for one batch:  %s seconds" % (time.time() - start_time))
        # print("-" * 50)
        #
        # zp_val = sess.run(_z)
        # z_vectors.append(zp_val)
        #
        # recovered_audio = sess.run(G_z, {z: _z})
        # wavfile.write("recovered_audio/wavegan_digit/" + "audio_" + str(batch) + ".wav", 16384, recovered_audio[0])
        index += 1

if __name__ == "__main__":
    # generate_fake_audio()
    recover_latent_vector()