import numpy as np
import torch

class ExperienceReplay():
  def __init__(self, size, observation_size, action_size, device='cuda'):
    self.device = device
    self.size = size
    self.observations = np.zeros((size, observation_size), dtype=np.float32)
    self.actions = np.zeros((size, ), dtype=np.int64)
    self.rewards = np.zeros((size, ), dtype=np.float32)
    self.next_observations = np.zeros((size, observation_size), dtype=np.float32)
    self.idx = 0
    self.full = False  # Tracks if memory has been filled/all slots are valid
    self.steps, self.episodes = 0, 0  # Tracks how much experience has been used in total

  def append(self, observation, action, reward, next_observation):
    self.observations[self.idx] = np.array(observation)
    self.actions[self.idx] = np.array(action)
    self.rewards[self.idx] = np.array(reward)
    self.next_observations[self.idx] = np.array(next_observation)
    self.idx = (self.idx + 1) % self.size
    self.full = self.full or self.idx == 0
    self.steps += 1

  def get_state(self, frame_size):
    idx = np.asarray([self.size + i + self.idx - frame_size + 1 for i in range(frame_size - 1)]) % self.size
    _pre_observation = self.observations[idx].reshape(-1)
    _observation = self.next_observations[self.idx - 1]
    return torch.from_numpy(np.concatenate([_pre_observation, _observation]))

  # Returns an index for a valid single sequence chunk uniformly sampled from the memory
  def _sample_idx(self, frame_size):
    valid_idx = False
    while not valid_idx:
      idx = np.random.randint(0, self.size if self.full else self.idx - frame_size)
      idxs = np.arange(idx, idx + frame_size) % self.size
      valid_idx = not self.idx in idxs[1:]  # Make sure data does not cross the memory index
    return idxs

  def _retrieve_batch(self, idxs, n, L):
    vec_idxs = idxs.transpose().reshape(-1)  # Unroll indices
    observations = torch.as_tensor(self.observations[vec_idxs].astype(np.float32))
    return observations.reshape(L, n, *observations.shape[1:]), self.actions[vec_idxs].reshape(L, n, -1), self.rewards[vec_idxs].reshape(L, n), self.next_observations[vec_idxs].reshape(L, n, 1)

  # Returns a batch of sequence chunks uniformly sampled from the memory
  def sample(self, n, frame_size):
    idx = np.asarray([self._sample_idx(frame_size) for _ in range(n)])
    _observations = self.observations[idx].reshape(n, -1)
    _rewards = self.rewards[idx[:, -1]]
    _action = self.actions[idx[:, -1]]
    _next_observations = self.next_observations[idx].reshape(n, -1)
    # batch = self._retrieve_batch(np.asarray([self._sample_idx(frame_size) for _ in range(n)]), n, frame_size)
    # print(np.asarray([self._sample_idx(L) for _ in range(n)]))
    # [1578 1579 1580 ... 1625 1626 1627]                                                                                                                                        | 0/100 [00:00<?, ?it/s]
    # [1049 1050 1051 ... 1096 1097 1098]
    # [1236 1237 1238 ... 1283 1284 1285]
    # ...
    # [2199 2200 2201 ... 2246 2247 2248]
    # [ 686  687  688 ...  733  734  735]
    # [1377 1378 1379 ... 1424 1425 1426]]
    return [torch.from_numpy(_observations), torch.unsqueeze(torch.from_numpy(_action), 1), torch.unsqueeze(torch.from_numpy(_rewards), 1), torch.from_numpy(_next_observations)]



# buffer = ExperienceReplay(1000, 10, 1)
# buffer.sample(100, 4)