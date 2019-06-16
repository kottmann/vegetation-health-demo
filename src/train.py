import ee
import tensorflow as tf
import os
import numpy as np

outputBucket = "/home/burn/Downloads/ee-docs-demos-berlin"

# Names for output files.
trainFilePrefix = 'Training_demo_'
testFilePrefix = 'Testing_demo_'

fileNameSuffix = 'ee_export.tfrecord.gz'

trainFilePath = outputBucket + '/' + trainFilePrefix + fileNameSuffix
testFilePath = outputBucket + '/' + testFilePrefix + fileNameSuffix

# Check existence of the exported files
print('Found training file.' if tf.gfile.Exists(trainFilePath)
    else 'No training file found.')
print('Found testing file.' if tf.gfile.Exists(testFilePath)
    else 'No testing file found.')

# Create a dataset from the TFRecord file in Cloud Storage.
trainDataset = tf.data.TFRecordDataset(trainFilePath, compression_type='GZIP')

# Print the first record to check.
# print(iter(trainDataset).next())

# Use these bands for prediction.
bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']

label = 'landcover'

# This is list of all the properties we want to export.
featureNames = list(bands)
featureNames.append(label)


# List of fixed-length features, all of which are float32.
columns = [
    tf.io.FixedLenFeature(shape=[1], dtype=tf.float32) for k in featureNames
]

# Dictionary with names as keys, features as values.
featuresDict = dict(zip(featureNames, columns))

def parse_tfrecord(example_proto):
    """The parsing function.

    Read a serialized example into the structure defined by featuresDict.

    Args:
      example_proto: a serialized Example.

    Returns:
      A tuple of the predictors dictionary and the label, cast to an `int32`.
    """
    parsed_features = tf.io.parse_single_example(example_proto, featuresDict)
    labels = parsed_features.pop(label)
    return parsed_features, tf.cast(labels, tf.int32)

# Map the function over the dataset.
parsedDataset = trainDataset.map(parse_tfrecord, num_parallel_calls=5)

def normalizedDifference(a, b):
    """Compute normalized difference of two inputs.

    Compute (a - b) / (a + b).  If the denomenator is zero, add a small delta.

    Args:
      a: an input tensor with shape=[1]
      b: an input tensor with shape=[1]

    Returns:
      The normalized difference as a tensor.
    """
    nd = (a - b) / (a + b)
    nd_inf = (a - b) / (a + b + 0.000001)
    return tf.where(tf.is_finite(nd), nd, nd_inf)

def addNDVI(features, label):
    """Add NDVI to the dataset.
    Args:
      features: a dictionary of input tensors keyed by feature name.
      label: the target label

    Returns:
      A tuple of the input dictionary with an NDVI tensor added and the label.
    """
    features['NDVI'] = normalizedDifference(features['B5'], features['B4'])
    return features, label

from tensorflow import keras

# How many classes there are in the model.
nClasses = 3

# Add NDVI.
inputDataset = parsedDataset.map(addNDVI)

# Keras requires inputs as a tuple.  Note that the inputs must be in the
# right shape.  Also note that to use the categorical_crossentropy loss,
# the label needs to be turned into a one-hot vector.
def toTuple(dict, label):
    return tf.transpose(list(dict.values())), tf.one_hot(indices=label, depth=nClasses)

# Repeat the input dataset as many times as necessary in batches of 10.
inputDataset = inputDataset.map(toTuple).repeat().batch(10)

# Define the layers in the model.
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(nClasses, activation=tf.nn.softmax)
])

# Compile the model with the specified loss function.
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model to the training data.
# Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
model.fit(x=inputDataset, epochs=3, steps_per_epoch=100)


# Save the model to disk
# tf.keras.models.save_model(model, "./model_path", overwrite=True)
model.save("./model_path")

