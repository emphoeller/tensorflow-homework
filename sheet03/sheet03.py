import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.strings as tfs
import numpy as np
import matplotlib.pyplot as plt

# analogous to onehotify from the sheet
def str_to_one_hot(str_tensor):
  for number, letter in enumerate('ACGT'):
    str_tensor = tfs.regex_replace(str_tensor, letter, str(number))
  labels = tf.cast(tfs.to_number(tfs.bytes_split(str_tensor)), tf.uint8)
  return tf.reshape(tf.one_hot(labels, 4), (-1,))

def prepare_dataset(ds):
  ds = ds.map(lambda seq, target: (str_to_one_hot(seq), tf.one_hot(target, 10)))
  ds = ds.cache()
  #ds = ds.shuffle(?)
  ds = ds.batch(8)
  ds = ds.prefetch(20)
  return ds

class MyDenseLayer(tf.keras.layers.Layer):
  
  def __init__(self, units, activation):
    super(MyDenseLayer, self).__init__()
    self.units = units
    self.activation = activation
  
  def build(self, input_shape):
    self.w = self.add_weight(shape = (input_shape[-1], self.units), initializer = 'random_normal', trainable = True)
    self.b = self.add_weight(shape = (self.units,), initializer = 'random_normal', trainable = True)
  
  def call(self, inputs):
    return self.activation(tf.matmul(inputs, self.w) + self.b)

class MyModel(tf.keras.Model):
  
  def __init__(self):
    super(MyModel, self).__init__()
    self.hidden_layer1 = MyDenseLayer(256, tf.nn.sigmoid)
    self.hidden_layer2 = MyDenseLayer(256, tf.nn.sigmoid)
    self.output_layer = MyDenseLayer(10, tf.nn.softmax)
  
  def call(self, inputs):
    return self.output_layer(self.hidden_layer2(self.hidden_layer1(inputs)))

ds_train, ds_test = tfds.load('genomics_ood', split = ('train', 'test'), shuffle_files = True, as_supervised = True)

ds_train = ds_train.take(100000)
ds_test = ds_test.take(1000)

ds_train = prepare_dataset(ds_train)
ds_test = prepare_dataset(ds_test)

model = MyModel()
optimizer = tf.keras.optimizers.SGD(learning_rate = 0.1)
loss_function = tf.keras.losses.CategoricalCrossentropy()

n_epochs = 10

training_loss_progression = []
test_loss_progression = []
test_accuracy_progression = []

for epoch in range(n_epochs):
  # Training step
  accumulated_training_loss = 0.0
  for seq, target in ds_train:
    with tf.GradientTape() as tape:
      prediction = model(seq)
      loss = loss_function(prediction, target)
      gradients = tape.gradient(loss, model.trainable_variables) # The lecture is inconsistent as to whether this should be inside or outside the with statement
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    accumulated_training_loss += loss.numpy()
  training_loss_progression.append(accumulated_training_loss / len(ds_train))
  # Test step
  accumulated_test_loss = 0.0
  accumulated_test_accuracy = 0.0
  for seq, target in ds_test:
    prediction = model(seq)
    accumulated_test_loss += loss_function(target, prediction).numpy()
    accumulated_test_accuracy += np.mean(np.argmax(target, axis = 1) == np.argmax(prediction, axis = 1))
  test_loss_progression.append(accumulated_test_loss / len(ds_test))
  test_accuracy_progression.append(accumulated_test_accuracy / len(ds_test))

# Since the training epoch has already finished at the point of testing, it makes sense to shift the x-coordinates.
# (This avoids the “test loss is lower than training loss” issue.)
x_training = range(0, n_epochs)
x_test = range(1, n_epochs + 1)

plt.figure()
training_loss_line, = plt.plot(x_training, training_loss_progression)
test_loss_line, = plt.plot(x_test, test_loss_progression)
test_accuracy_line, = plt.plot(x_test, test_accuracy_progression)
plt.xlabel('Epochs passed')
plt.legend((training_loss_line, test_loss_line, test_accuracy_line), ('Training Loss', 'Test Loss', 'Test Accuracy'))
plt.show()
