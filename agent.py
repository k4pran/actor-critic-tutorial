import tensorflow as tf
from agent_frame import AgentBase
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from model import ACModel


class ACAgent(AgentBase):

    def __init__(self, learning_rate, discount, action_space, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate = learning_rate
        self.discount = discount
        self.action = None

        self.ac_model = ACModel(action_space)
        self.ac_model.compile(optimizer=Adam(learning_rate=learning_rate))

    def before(self, *args, **kwargs):
        pass

    def after(self, *args, **kwargs):
        pass

    def act(self, state) -> int:
        state = tf.convert_to_tensor([state])
        _, probs = self.ac_model(state)

        action_probs = tfp.distributions.Categorical(probs=probs)
        self.action = action_probs.sample()
        return self.action.numpy().item()

    def learn(self, *args, **kwargs):
        state = tf.convert_to_tensor([kwargs['state']], dtype=tf.float32)
        next_state = tf.convert_to_tensor([kwargs['next_state']], dtype=tf.float32)
        reward = tf.convert_to_tensor([kwargs['reward']], dtype=tf.float32)
        done = kwargs['done']

        with tf.GradientTape(persistent=False) as tape:
            state_val, probs = self.ac_model(state)
            next_state_val, _ = self.ac_model(next_state)

            state_val = tf.squeeze(state_val)
            next_state_val = tf.squeeze(next_state_val)

            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(self.action)

            exp_val = reward + self.discount * next_state_val * (1 - int(done)) - state_val

            actor_loss = -log_prob * exp_val
            critic_loss = exp_val**2
            total_loss = actor_loss + critic_loss

        gradient = tape.gradient(total_loss, self.ac_model.trainable_variables)
        self.ac_model.optimizer.apply_gradients(zip(gradient, self.ac_model.trainable_variables))

    def save_model(self):
        self.ac_model.save_weights("path/to/file")

    def load_model(self):
        self.ac_model.load_weights("path/to/file")


def create_agent(learning_rate=0.001, discount=0.8, action_space=2):
    return ACAgent(learning_rate, discount, action_space)
