from tensorflow.keras.layers import Layer


# trims off last coordinate
# assumes shape is (None, n)
class TrimmingLayer(Layer):
    def __init__(self):
        super(TrimmingLayer, self).__init__()

    def build(self, input_shape):
        super(TrimmingLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        return x[:, 0:(x.shape[1] - 1)]

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1] - 1
