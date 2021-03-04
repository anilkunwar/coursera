from P_Teaching_an_RL_Agent_to_Play_Atari_using_Deep_Q_Learning.main import DELAY_TRAINING, POST_PROCESS_IMAGE_SIZE
import numpy as np

def policy(state, t):
    p = np.array(q[(state, x)] / t for x in range(env.action_space.n))
    prob_actions = np.exp(p) / np.sum(np.exp(p))
    cumulative_prob = 0.0
    choice = random.uniform(0, 1)

    for a, pr in enumerate(prob_actions):
        cumulative_prob += pr
        if cumulative_prob > choice:
            return a

def choose_action(state, primary_network, eps, step):
    if step < DELAY_TRAINING:
        return random.randint(0, num_actions - 1)
    else:
        if random.random() < eps:
            return random.randint(0, num_actions - 1)
        else:
            return np.argmax(primary_network(tf.reshape(state, (1, POST_PROCESS_IMAGE_SIZE[0], POST_PROCESS_IMAGE_SIZE[1], NUM_FRAMES))))