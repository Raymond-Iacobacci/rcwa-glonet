import tensorflow as tf
from tensorflow.keras import layers






class Generator(tf.keras.Model):
    def __init__(self, params):
        super(Generator, self).__init__()
        dim = 3
        self.noise_dim = params['noise_dims']
        self.gk = tf.random.normal((dim, dim), mean=0, stddev=4)
        self.FC = tf.keras.Sequential([
            layers.Dense(25, activation='linear', use_bias=True),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.2),
        ])
        self.conv1 = layers.Conv2DTranspose(16, kernel_size=6, padding='same', strides=1, use_bias=True)
        self.bn1 = layers.BatchNormalization(momentum=False)
        self.conv2 = layers.Conv2DTranspose(16, kernel_size=4, strides=1, use_bias=True)
        self.bn2 = layers.BatchNormalization(momentum=False)

    def call(self, noise, params):
        net = self.FC(noise)
        net = tf.reshape(net, (-1, 5, 5, 1))  # assuming input shape is (batch_size, 25)
        net = self.conv1(net)
        net = tf.reshape(net, (4, 4, 8, 8))
        net = self.bn1(net)
        net = tf.reshape(net, (16, 8, 8))
        net = layers.LeakyReLU(0.2)(net)
        net = self.conv2(net)
        net = tf.reshape(net, (4, 4, 11, 11))
        net = self.bn2(net)
        net = tf.reshape(net, (-1, 11, 11))
        net = tf.tanh(net * params['binary_amp']) * 0.5 + 0.5
        return net

# Example usage:
# params = {'noise_dims': 25, 'binary_amp': 1.0}
# generator = Generator(params)
# noise = tf.random.normal((batch_size, params['noise_dims']))
# output = generator(noise, params)