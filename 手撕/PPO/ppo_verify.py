import gymnasium as gym
import torch
from ppo_agent import PPOAgent

scenario = 'Pendulum-v1'
MODE_PATH = 'ppo_policy_pendulum_v1.para'
NUM_EVAL_EPISODE = 10

env = gym.make(scenario, render_mode='human')
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]

agent = PPOAgent(STATE_DIM, ACTION_DIM, batch_size=25)

agent.actor.load_state_dict(torch.load(MODE_PATH))
agent.actor.eval()

eval_rewards = []
for episode_i in range(NUM_EVAL_EPISODE):
    state, others = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action, value = agent.get_action(state)
        next_state, reward, done, truncated, info= env.step(action)
        episode_reward += reward
        done = True if truncated else done
        state = next_state

    eval_rewards.append(episode_reward)
    print(f'Eval Episode: {episode_i}, Reward: {round(episode_reward, 2)}')

env.close()
