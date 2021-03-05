import gym
from numpy.core.records import record
import tensorflow as tf
import random
import numpy as np
import datetime as dt
import imageio
import os
from util import image_preprocess
from model import DQN
from memory import Memory

def choose_action(state, primary_network, eps, step):
    if step < DELAY_TRAINING:
        return random.randint(0, num_actions - 1)
    else:
        if random.random() < eps:
            return random.randint(0, num_actions - 1)
        else:
            return np.argmax(primary_network(tf.reshape(state, (1, POST_PROCESS_IMAGE_SIZE[0], POST_PROCESS_IMAGE_SIZE[1], NUM_FRAMES)).numpy()))

def process_state_stack(state_stack, state):
    for i in range(1, state_stack.shape[-1]):
        state_stack[:, :, i - 1].assign(state_stack[:, :, i])
    state_stack[:, :, -1].assign(state[:, :, 0])
    return state_stack

def train(primary_network, memory):
    states, actions, rewards, next_states, terminal = memory.sample()
    prim_q = primary_network(states)
    prim_qn = primary_network(next_states)
    target_q = prim_q.numpy()
    updates = rewards
    valid_idxs = terminal != True
    batch_idxs = np.arange(BATCH_SIZE)
    updates[valid_idxs] += GAMMA * np.amax(prim_qn.numpy()[valid_idxs, :], axis=1)
    target_q[batch_idxs, actions] = updates
    loss = primary_network.train_on_batch(states, target_q)
    return loss

MAX_EPS = 1
MIN_EPS = 0.1
EPS_MIN_ITER = 500000

GAMMA = 0.98
BATCH_SIZE = 64

POST_PROCESS_IMAGE_SIZE = (105, 84, 1)

# For avoiding catastrophic forgetting
DELAY_TRAINING = 50000

NUM_FRAMES = 4
GIF_RECORDING_FREQ = 100
MODEL_SAVE_FREQ = 100

env = gym.make('Phoenix-v4')
num_actions = env.action_space.n

primary_network = DQN(1024, num_actions)
primary_network.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber())

#  max_memory, batch_size, frame_size, total_frame
memory = Memory(100000, BATCH_SIZE, POST_PROCESS_IMAGE_SIZE, NUM_FRAMES)

eps = MAX_EPS
render = True
steps = 0

def record_gif(frame_list, episode, fps=30):
    imageio.mimsave(f"./Phoenix-{episode}.gif", frame_list, fps=fps)

for i in range(1000000):
    state = env.reset()
    state = image_preprocess(state)
    state_stack = tf.Variable(np.repeat(state.numpy(), NUM_FRAMES).reshape((POST_PROCESS_IMAGE_SIZE[0],
                                                                            POST_PROCESS_IMAGE_SIZE[1],
                                                                            NUM_FRAMES)))
    cnt = 1
    avg_loss = 0
    total_reward = 0

    if i % GIF_RECORDING_FREQ == 0:
        frame_list = []

    while True:
        if render:
            env.render()
        action = choose_action(state_stack, primary_network, eps, steps)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        if i % GIF_RECORDING_FREQ == 0:
            frame_list.append(tf.cast(tf.image.resize(next_state, (480, 320)), tf.uint8).numpy())
        next_state = image_preprocess(next_state)
        state_stack = process_state_stack(state_stack, next_state)
        memory.add_sample(next_state, action, reward, done)

        if steps > DELAY_TRAINING:
            loss = train(primary_network, memory)
        else:
            loss = -1
        avg_loss += loss

        # linearly decay the eps value
        if steps > DELAY_TRAINING:
            eps = MAX_EPS - ((steps - DELAY_TRAINING) / EPS_MIN_ITER) * \
                  (MAX_EPS - MIN_EPS) if steps < EPS_MIN_ITER else \
                MIN_EPS
        steps += 1

        if done:
            if steps > DELAY_TRAINING:
                avg_loss /= cnt
                print(f"Episode: {i}, Reward: {total_reward}, avg loss: {avg_loss:.5f}, eps: {eps:.3f}")
            else:
                print(f"Pre-training...Episode: {i}")

            if i % MODEL_SAVE_FREQ == 0:
                primary_network.save_weights(f'./weights-{i}/', overwrite=True)

            if i % GIF_RECORDING_FREQ == 0:
                record_gif(frame_list, i)
            break

        cnt += 1

env.close()