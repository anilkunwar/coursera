import numpy as np

class Memory:
    def __init__(self, max_memory, batch_size, frame_size, total_frame):
        self._max_memory = max_memory
        self._batch_size = batch_size
        self._total_frame = total_frame
        self._actions = np.zeros(max_memory, dtype=np.int32)
        self._rewards = np.zeros(max_memory, dtype=np.float32)
        self._frames = np.zeros((frame_size[0], frame_size[1], max_memory), dtype=np.float32)
        self._terminal = np.zeros(max_memory, dtype=np.bool)
        self._i = 0

    def add_sample(self, frame, action, reward, terminal):
        self._actions[self._i] = action
        self._rewards[self._i] = reward
        self._frames[:, :, self._i] = frame[:, :, 0]
        self._terminal[self._i] = terminal

        if self._i % (self._max_memory - 1) == 0 and self._i != 0:
            self._i = self._batch_size + self._total_frame + 1
        else:
            self._i += 1

    def sample(self):
        if self._i < self._batch_size + self._total_frame + 1:
            raise ValueError('Not Enough memory to extract a batch')
        else:
            idxs = np.random.randint(self._total_frame + 1, self._i, size=self._batch_size)
            states = np.zeros((self._batch_size, self._frames.shape[0], self._frames.shape[1], self._total_frame), dtype=np.float32)
            next_states = np.zeros((self._batch_size, self._frames.shape[0], self._frames.shape[1], self._total_frame), dtype=np.float32)
            for i, idx in enumerate(idxs):
                states[i] = self._frames[:, :, idx - 1 - self._total_frame:idx - 1]
                next_states[i] = self._frames[:, :, idx - self._total_frame:idx]
            return states, self._actions[idxs], self._rewards[idxs], next_states, self._terminal[idxs]