import tensorflow.keras as keras
from tensorflow.keras.layers import Dense


class ACModel(keras.Model):

    def __init__(self, action_space):
        super(ACModel, self).__init__()
        self.fc1 = Dense(1024, activation='relu')
        self.fc2 = Dense(512, activation='relu')
        self.value_output = Dense(1, activation=None)
        self.policy_output = Dense(action_space, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        output = self.fc1(inputs)
        output = self.fc2(output)

        value = self.value_output(output)
        policy = self.policy_output(output)
        return value, policy


def create_model():
    pass
