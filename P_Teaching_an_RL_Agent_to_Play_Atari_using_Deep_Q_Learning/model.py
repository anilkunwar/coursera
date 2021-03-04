import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense

class DQN(tf.keras.Model):
    def __init__(self, hidden_size: int, num_actions: int):
        super(DQN, self).__init__()
        self.conv1 = Conv2D(16, kernel_size=8, strides=4, activation='relu', use_bias=False)
        self.conv2 = Conv2D(32, kernel_size=4, strides=2, activation='relu', use_bias=False)
        self.flatten = Flatten()
        self.adv_dense = Dense(hidden_size, activation='relu', use_bias=False,
                                kernel_initializer=tf.keras.initializers.he_normal())
        self.adv_out = Dense(num_actions, activation='softmax',
                                kernel_initializer=tf.keras.initializers.he_normal())

    def call(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.flatten(x)

        adv = self.adv_dense(x)
        adv = self.adv_out(adv)
        return adv