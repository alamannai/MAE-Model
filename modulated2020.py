#pip install tensorflow-gpu==2.4.* tensorflow-compression==2.0

import argparse
import glob
import sys
from absl import app
from absl.flags import argparse_flags
from keras.datasets import cifar100
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_compression as tfc
from tensorflow import keras
import numpy as np

def read_png(filename):
  """Loads a PNG image file."""
  string = tf.io.read_file(filename)
  return tf.image.decode_image(string, channels=3)


def write_png(filename, image):
  """Saves an image to a PNG file."""
  string = tf.image.encode_png(image)
  tf.io.write_file(filename, string)

def check_image_size(image, patchsize):
  shape = tf.shape(image)
  return shape[0] >= patchsize and shape[1] >= patchsize and shape[-1] == 3


def crop_image(image, patchsize):
  image = tf.image.random_crop(image, (patchsize, patchsize, 3))
  return tf.cast(image, tf.float32)


def get_dataset(name,split, patchsize, batchsize):
  """Creates input data pipeline from a TF Datasets dataset."""
  with tf.device("/cpu:0"):
    dataset = tfds.load(name,split=split, shuffle_files=True)
    #if split == "train":
      #dataset = dataset.repeat()
    dataset = dataset.filter(
        lambda x: check_image_size(x["image"], patchsize))
    dataset = dataset.map(
        lambda x: crop_image(x["image"], patchsize))
    dataset = dataset.batch(batchsize, drop_remainder=True)
  return dataset

modulated = keras.Input(shape=(1, 1))
x = tf.keras.layers.Dense(50)(modulated)
x = tf.nn.relu(x)
x = tf.keras.layers.Dense(128)(x)
output = tf.math.exp(x)
modulated_network_a = keras.Model(inputs=modulated, outputs=output, name="ma")
modulated_network_s = keras.Model(inputs=modulated, outputs=output, name="ms")

modulated_network_a.summary()

class HyperAnalysisTransform(tf.keras.Sequential):
  """The analysis transform for the entropy model parameters."""

  def __init__(self, num_filters):
    super().__init__(name="hyper_analysis")
    self.add(tfc.SignalConv2D(
        num_filters, (3, 3), name="layer_0", corr=True, strides_down=1,
        padding="same_zeros", use_bias=True, kernel_parameter="variable",
        activation=tf.nn.relu))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True, kernel_parameter="variable",
        activation=tf.nn.relu))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
        padding="same_zeros", use_bias=False, kernel_parameter="variable",
        activation=None))

class HyperSynthesisTransform(tf.keras.Sequential):
  """The synthesis transform for the entropy model parameters."""

  def __init__(self, num_filters):
    super().__init__(name="hyper_synthesis")
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True, kernel_parameter="variable",
        activation=tf.nn.relu))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True, kernel_parameter="variable",
        activation=tf.nn.relu))
    self.add(tfc.SignalConv2D(
        num_filters, (3, 3), name="layer_2", corr=False, strides_up=1,
        padding="same_zeros", use_bias=True, kernel_parameter="variable",
        activation=None))

class modulated_analysis_transform(tf.keras.Model):
    """Builds the modulated analysis transform."""
    def __init__(self, num_filters):
      super().__init__()
      self.layer1 = tfc.SignalConv2D(num_filters, (9, 9), corr=True, strides_down=4,padding="same_zeros",kernel_parameter="variable", use_bias=True,activation=None)
      self.layerx = tfc.SignalConv2D(num_filters, (5, 5), corr=True, strides_down=2,padding="same_zeros", kernel_parameter="variable",use_bias=True,activation=None)
      self.gdn = tfc.GDN()
    
    def call(self, input_tensor, conds):
      x = tf.keras.layers.Lambda(lambda x: x / 255.)(input_tensor)
      x = self.layer1(x)
      vector = conds
      modulated_tensor1 = x * vector
      a = self.gdn(modulated_tensor1)

      x = self.layerx(a)
      modulated_tensor2 = x * vector
      a = self.gdn(modulated_tensor2)

      x = self.layerx(modulated_tensor2)
      modulated_tensor = x * vector
      a = self.gdn(modulated_tensor)

      return a

class demodulated_synthesis_transform(tf.keras.Sequential):
    """Builds the demodulated synthesis transform."""
    def __init__(self, num_filters):
      super().__init__()
      self.layer1 = tfc.SignalConv2D(3, (9, 9), corr=True, strides_up=4,padding="same_zeros", kernel_parameter="variable", use_bias=True,activation=None)
      self.layerx = tfc.SignalConv2D(num_filters, (5, 5), corr=True, strides_up=2,padding="same_zeros",kernel_parameter="variable", use_bias=True,activation=None)
      self.gdn = tfc.GDN(inverse=True)
    
    def call(self, input_tensor, conds):
      x = self.gdn(input_tensor)
      vector = conds
      demodulated_tensor1 = x * vector
      a = self.layerx(demodulated_tensor1)

      x = self.gdn(a)
      demodulated_tensor2 = x * vector
      a = self.layerx(demodulated_tensor2)

      x = self.gdn(a)
      demodulated_tensor2 = x * vector
      a = self.layer1(demodulated_tensor2)
      x = tf.keras.layers.Lambda(lambda x: x * 255.)(a)
      return x

class mae(tf.keras.Model):
  """Main model class."""

  def __init__(self, lmbda, num_filters, num_scales, scale_min, scale_max, modulated_network_a , modulated_network_s):
    super().__init__()
    self.lmbda = lmbda
    self.num_scales = num_scales
    offset = tf.math.log(scale_min)
    factor = (tf.math.log(scale_max) - tf.math.log(scale_min)) / (
        num_scales - 1.)
    self.scale_fn = lambda i: tf.math.exp(offset + factor * i)
    self.modulated_network_a = modulated_network_a
    self.modulated_network_s = modulated_network_s
    self.analysis_transform = modulated_analysis_transform(num_filters)
    self.synthesis_transform = demodulated_synthesis_transform(num_filters)
    self.hyper_analysis_transform = HyperAnalysisTransform(num_filters)
    self.hyper_synthesis_transform = HyperSynthesisTransform(num_filters)
    self.hyperprior = tfc.NoisyDeepFactorized(batch_shape=(num_filters,))
    self.build((None, None, None, 3))
  
  @property

  def variables(self):
    return tf.Module.variables.fget(self)

  @property
  def trainable_variables(self):
    return tf.Module.trainable_variables.fget(self)

  weights = variables
  trainable_weights = trainable_variables

  # This seems to be necessary to prevent a comparison between class objects.
  _TF_MODULE_IGNORED_PROPERTIES = (
      tf.keras.Model._TF_MODULE_IGNORED_PROPERTIES.union(
          ("_compiled_trainable_state",)
      ))
  ############################################################################

  def call(self, x, training):
    """Computes rate and distortion losses."""
    entropy_model = tfc.LocationScaleIndexedEntropyModel(
        tfc.NoisyNormal, self.num_scales, self.scale_fn, coding_rank=3,
        compression=False)
    side_entropy_model = tfc.ContinuousBatchedEntropyModel(
        self.hyperprior, coding_rank=3, compression=False)
    total_loss, total_bpp, total_mse, total_mssim = list(), list(), list(), list()
    for i, _lmbda in enumerate(self.lmbda):
      p = _lmbda / max(self.lmbda)
      cond = tf.convert_to_tensor(p,dtype=tf.float32)
      cond = tf.expand_dims(cond,0)
      cond = tf.expand_dims(cond,0)
      conda = self.modulated_network_a(cond)
      y = self.analysis_transform(x,conda)
      z = self.hyper_analysis_transform(abs(y))
      z_hat, side_bits = side_entropy_model(z, training=training)
      indexes = self.hyper_synthesis_transform(z_hat)
      y_hat, bits = entropy_model(y, indexes, training=training)
      conds = self.modulated_network_s(cond)
      x_hat = self.synthesis_transform(y_hat,conds)

      # Total number of bits divided by total number of pixels.
      num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), bits.dtype)
      bpp = (tf.reduce_sum(bits) + tf.reduce_sum(side_bits)) / num_pixels
      # Mean squared error across pixels.
      mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
      mss_ssim = tf.squeeze(tf.image.ssim_multiscale(x, x_hat, 255, filter_size=11,filter_sigma=1.5, k1=0.01, k2=0.03))
      # The rate-distortion Lagrangian.
      loss = bpp + _lmbda * mse

      total_loss.append(loss)
      total_bpp.append(bpp)
      total_mse.append(mse)
      total_mssim.append(mss_ssim)
    
    loss = tf.math.add_n(total_loss) 
    bpp = tf.math.add_n(total_bpp) #/ len(self.lmbda)
    mse = tf.math.add_n(total_mse) #/ len(self.lmbda)
    mssim = tf.math.add_n(total_mssim) #/ len(self.lmbda)


    return loss, bpp, mse, mssim

  def train_step(self, x):
    with tf.GradientTape() as tape:
      loss, bpp, mse, mss_ssim = self(x, training=True)
    variables = self.trainable_variables
    gradients = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))
    self.loss.update_state(loss)
    self.bpp.update_state(bpp)
    self.mse.update_state(mse)
    self.mss_ssim.update_state(mss_ssim)
    return {m.name: m.result() for m in [self.loss, self.bpp, self.mse, self.mss_ssim]}
  
  def test_step(self, x):
    loss, bpp, mse = self(x, training=False)
    self.loss.update_state(loss)
    self.bpp.update_state(bpp)
    self.mse.update_state(mse)
    return {m.name: m.result() for m in [self.loss, self.bpp, self.mse]}

  def predict_step(self, x):
    raise NotImplementedError("Prediction API is not supported.")

  def compile(self, **kwargs):
    super().compile(
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        **kwargs,
    )
    self.loss = tf.keras.metrics.Mean(name="loss")
    self.bpp = tf.keras.metrics.Mean(name="bpp")
    self.mse = tf.keras.metrics.Mean(name="mse")
    self.mss_ssim = tf.keras.metrics.Mean(name="mss_ssim")

  def fit(self, *args, **kwargs):
    retval = super().fit(*args, **kwargs)
    # After training, fix range coding tables.
    self.entropy_model = tfc.LocationScaleIndexedEntropyModel(
        tfc.NoisyNormal, self.num_scales, self.scale_fn, coding_rank=3,
        compression=True)
    self.side_entropy_model = tfc.ContinuousBatchedEntropyModel(
        self.hyperprior, coding_rank=3, compression=True)
    return retval

  @tf.function(input_signature=[
      tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
      tf.TensorSpec(shape=(1,), dtype=tf.float32),
      tf.TensorSpec(shape=(1,), dtype=tf.float32),
  ])
  def compress(self, x, lmbda, norm):
    """Compresses an image."""
    p = lmbda / norm
    cond = tf.convert_to_tensor(p,dtype=tf.float32)
    cond = tf.expand_dims(cond,0)
    cond = tf.expand_dims(cond,0)
    conda = self.modulated_network_a(cond)
    # Add batch dimension and cast to float.
    x = tf.expand_dims(x, 0)
    x = tf.cast(x, dtype=tf.float32)
    y = self.analysis_transform(x,conda)
    z = self.hyper_analysis_transform(abs(y))
    # Preserve spatial shapes of image and latents.
    x_shape = tf.shape(x)[1:-1]
    y_shape = tf.shape(y)[1:-1]
    z_shape = tf.shape(z)[1:-1]
    z_hat, _ = self.side_entropy_model(z, training=False)
    indexes = self.hyper_synthesis_transform(z_hat)
    indexes = indexes[:, :y_shape[0], :y_shape[1], :]
    side_string = self.side_entropy_model.compress(z)
    string = self.entropy_model.compress(y, indexes)
    return string, side_string, x_shape, y_shape, z_shape, cond

  @tf.function(input_signature=[
      tf.TensorSpec(shape=(1,), dtype=tf.string),
      tf.TensorSpec(shape=(1,), dtype=tf.string),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
      tf.TensorSpec(shape=(1, 1, 1), dtype=tf.float32)
  ])
  def decompress(self, string, side_string, x_shape, y_shape, z_shape, cond):
    """Decompresses an image."""
    z_hat = self.side_entropy_model.decompress(side_string, z_shape)
    indexes = self.hyper_synthesis_transform(z_hat)
    indexes = indexes[:, :y_shape[0], :y_shape[1], :]
    y_hat = self.entropy_model.decompress(string, indexes)
    conds = self.modulated_network_s(cond)
    x_hat = self.synthesis_transform(y_hat,conds)
    # Remove batch dimension, and crop away any extraneous padding.
    x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]
    # Then cast back to 8-bit integer.
    return tf.saturate_cast(tf.round(x_hat), tf.uint8)

lmbda = [12,64,512,2048]
num_filters = 128
num_scales = 64.0
scale_min = 0.11
scale_max = 256.0

dataset_path =""
max_validation_steps = 16
epochs = 1000
verbose = 1.0
batchsize = 8
patchsize = 256

model = mae(lmbda, num_filters, num_scales, scale_min,scale_max, modulated_network_a,modulated_network_s)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)
)

from google.colab import drive
drive.mount("/content/gdrive")

import os
path = 'gdrive/MyDrive/training/'
images = os.listdir(path)
files = []
for i in images:
  if i != '.DS_Store':
    files.append(path+i)

short = []
vlist = []
for j in range(900):
  short.append(files[j])

#training dataset
dataset = tf.data.Dataset.from_tensor_slices(short)
dataset = dataset.shuffle(len(short), reshuffle_each_iteration=True)
#if split == "train":
  #dataset = dataset.repeat()
dataset = dataset.map(
    lambda x: crop_image(read_png(x), patchsize))
dataset = dataset.batch(batchsize, drop_remainder=True)

model.fit(
    dataset,
    epochs=epochs,
    #validation_data=valid_dataset.cache()
    #validation_freq=1,
    #verbose=int(verbose)
    callbacks=[
              tf.keras.callbacks.TerminateOnNaN(),
              #tf.keras.callbacks.TensorBoard(
                  #log_dir='/content/',
                  #histogram_freq=1, update_freq="epoch"),
              tf.keras.callbacks.experimental.BackupAndRestore('/content/gdrive/MyDrive/mae/'),
          ]
)
#model.save(args.model_path)

model.summary()

from google.colab import files

uploaded = files.upload()

x = read_png('00003_TE_4000x3000.png')

l = tf.constant([512.0])
l.shape
norm = tf.constant([2048.0])
string, side_string, x_shape, y_shape, z_shape, cond= model.compress(x,l, norm)
#packed = tfc.PackedTensors()
#packed.pack(tensors)

print(x_shape)

x_hat = model.decompress(string, side_string, x_shape, y_shape, z_shape, cond)

# Cast to float in order to compute metrics.
x = tf.cast(x, tf.float32)
x_hat = tf.cast(x_hat, tf.float32)
mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
psnr = tf.squeeze(tf.image.psnr(x, x_hat, 255))
msssim = tf.squeeze(tf.image.ssim_multiscale(x, x_hat, 255, filter_size=11,filter_sigma=1.5, k1=0.01, k2=0.03))
msssim_db = -10. * tf.math.log(1 - msssim) / tf.math.log(10.)

# The actual bits per pixel including entropy coding overhead.
num_pixels = tf.reduce_prod(tf.shape(x)[:-1])
bpp = (tf.strings.length(string[0])+ tf.strings.length(side_string[0])) * 8 / num_pixels

print(tf.strings.length(string[0]))
print(bpp)

print(f"Mean squared error: {mse:0.4f}")
print(f"PSNR (dB): {psnr:0.2f}")
print(f"Multiscale SSIM: {msssim:0.4f}")
print(f"Multiscale SSIM (dB): {msssim_db:0.2f}")
print(f"Bits per pixel: {bpp:0.4f}")

import matplotlib.pyplot as plt

plt.imshow(tf.dtypes.cast(x_hat, tf.int32))

plt.imshow(tf.dtypes.cast(x, tf.int32))