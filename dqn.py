import torch
from torch import jit, nn
from torch.nn import functional as F
import torch.distributions
from torch.distributions.normal import Normal
import numpy as np
from memory import ExperienceReplay
from env import DogfightEnv
import argparse
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--tau', default=0.005, type=float)  # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--checkpoints', default=250000, type=int)
parser.add_argument('--memory-checkpoints', default=500000, type=int)
parser.add_argument('--experience-replay', default='./experience.pth', type=str)
parser.add_argument('--load-experience', action='store_true')
parser.add_argument('--load-model', default=0, type=int)
parser.add_argument('--test-interval', default=1000, type=int)
parser.add_argument('--gradient_steps', default=1, type=int)
parser.add_argument('--learning_rate', default=1e-5, type=float)
parser.add_argument('--action-repeat', default=4, type=int)
parser.add_argument('--gamma', default=0.95, type=float)  # discount gamma
parser.add_argument('--epsilon', default=0.9, type=float)
parser.add_argument('--capacity', default=500000, type=int)  # replay buffer size
parser.add_argument('--iteration', default=100000, type=int)  # num of  games
parser.add_argument('--batch_size', default=256, type=int)  # mini batch size
parser.add_argument('--frames_size', default=1, type=int)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--episodes', default=100000, type=int)
parser.add_argument('--observation_size', default=40, type=float)
parser.add_argument('--update-interval', default=10000, type=int)
parser.add_argument('--action_size', default=12, type=int)
parser.add_argument('--num_hidden_layers', default=2, type=int)
parser.add_argument('--num_hidden_units_per_layer', default=256, type=int)
parser.add_argument('--sample_frequency', default=256, type=int)
parser.add_argument('--activation', default='Relu', type=str)
parser.add_argument('--renderless', action='store_true')  # show UI or not
parser.add_argument('--log_interval', default=50, type=int)  #
parser.add_argument('--load', default=False, type=bool)  # load model
parser.add_argument('--render_interval', default=100, type=int)  # after render_interval, the env.render() will work
args = parser.parse_args()

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, min_log_std=-10, max_log_std=2):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.mu_head = nn.Linear(512, action_dim)
        self.log_std_head = nn.Linear(512, action_dim)
        self.max_action = max_action

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std_head = F.relu(self.log_std_head(x))
        log_std_head = torch.clamp(log_std_head, self.min_log_std, self.max_log_std)
        return mu, log_std_head


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Net(nn.Module):
    """docstring for Net"""
    def __init__(self, observation_size, action_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(observation_size * args.frames_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, action_size)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        out = self.fc3(x)
        return out

class DQN():
    """docstring for DQN"""
    def __init__(self, observation_size, action_size):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Net(observation_size, action_size).to(device), Net(observation_size, action_size).to(device)
        if args.load_model:
            self.eval_net.load_state_dict(torch.load(f'models/{args.load_model}.model'))
        self.learn_step_counter = 0
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=args.learning_rate)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state, exploration=True):
        epsilon = args.epsilon
        state = torch.unsqueeze(state, 0).cuda() # get a 1D array
        if np.random.uniform() <= epsilon or not exploration:# greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value.cpu(), 1)[1].data.numpy()
            action = action[0]
        else: # random policy
            action = np.random.randint(0, args.action_size)
        return action


    def learn(self, memory):
        self.eval_net.train()
        #update the parameters
        if self.learn_step_counter % args.update_interval ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter+=1

        #sample batch(o,a,r,_o) from memory
        batch = memory.sample(args.batch_size, args.frames_size)

        #q_eval
        q_eval = self.eval_net(batch[0].cuda()).gather(1, batch[1].cuda())
        q_next = self.target_net(batch[-1].cuda()).detach()
        q_target = batch[2].cuda() + args.gamma * q_next.max(1)[0].view(args.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # save model
        if self.learn_step_counter % args.checkpoints == 0:
            if not os.path.exists("dqn_models"):
                os.makedirs("dqn_models")
            model_save_path = f"dqn_models/{self.learn_step_counter}.model"
            torch.save(self.eval_net.state_dict(), model_save_path)


def main():
    dqn = DQN(args.observation_size, args.action_size)
    env = DogfightEnv(action_repeat=args.action_repeat)
    writer = SummaryWriter('dqn')
    if os.path.exists(args.experience_replay) and args.load_experience:
        memory = torch.load(args.experience_replay)
        memory_counter = args.capacity
        _episode = memory.episodes
    else:
        _episode = 0
        memory_counter = 0  # 总步数
        memory = ExperienceReplay(args.capacity, args.observation_size, args.action_size, device)
    print("Collecting Experience....")
    pbar = tqdm(range(args.episodes - _episode), ncols=100)
    for episode in pbar:
        observation = env.reset()
        observation = torch.FloatTensor(env.reset())
        done = False
        # state = torch.cat([torch.zeros(args.observation_size * (args.frames_size - 1)), torch.FloatTensor(observation)], 0)
        total_reward = 0
        while not done:
            action = dqn.choose_action(observation, exploration=True)
            # action = env.sample_random_action()
            next_observation, reward, done = env.step(action)
            # print(next_observation)
            memory.append(observation, action, reward, next_observation)
            observation = next_observation
            observation = torch.FloatTensor(observation)
            memory_counter += 1
            total_reward += reward
            if memory_counter >= args.capacity:
                dqn.learn(memory)
                if done:
                    print(f" 😍Total reward is {total_reward:.5f}")
                    writer.add_scalar("total reward", total_reward, episode)
            if memory_counter % args.memory_checkpoints == 0:
                torch.save(memory, args.experience_replay)
            pbar.set_postfix({"step": memory_counter, "reward": f"{reward:.5f}"})
            # state = memory.get_state(args.frames_size)


        if episode % args.test_interval == 0:
            observation = env.reset()
            observation = torch.FloatTensor(env.reset())
            done = False
            test_reward = 0
            while not done:
                action = dqn.choose_action(observation, exploration=False)
                next_observation, reward, done = env.step(action)
                observation = next_observation
                observation = torch.FloatTensor(observation)
                test_reward += reward
                # state = memory.get_state(args.frames_size)
            print(f" ✈Test reward is {test_reward:.5f}")
            writer.add_scalar("test reward", test_reward, episode)

if __name__ == '__main__':
    main()

