import apache_beam as beam

class ParseTFRecord(beam.DoFn):

    def __init__(self, featuresDict, label):
        self.featuresDict = featuresDict
        self.label = label

    def process(self, example_proto):
        """The parsing function.

        Read a serialized example into the structure defined by featuresDict.

        Args:
          example_proto: a serialized Example.

        Returns:
          A tuple of the predictors dictionary and the label, cast to an `int32`.
        """
        parsed_features = tf.io.parse_single_example(example_proto, self.featuresDict)

        labels = parsed_features.pop(self.label)
        return parsed_features, tf.cast(labels, tf.int32)
