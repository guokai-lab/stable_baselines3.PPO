import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np
import torch

# 创建并监控训练环境
env = Monitor(gym.make("Pendulum-v1"))

# 创建用于周期性评估的环境（不渲染）
eval_env = Monitor(gym.make("Pendulum-v1"))
eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path='C:/Users/GuoKai/Desktop/python_demo/PI自整定-强化学习/PI整定-SAC-simulink/SAC_PI_Mode',
    eval_freq=5000,
    deterministic=True, 
    render=False
)

# 定义PPO模型，优化超参数
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1,
    learning_rate=1e-3,      # 提高学习率，加速训练
    n_steps=2048,            # 每个更新批次收集的步数
    batch_size=64,           # 降低batch_size以增加更新频率
    n_epochs=10,             # 每批数据的训练轮数
    gamma=0.99,              # 折扣因子
    gae_lambda=0.95,         # GAE参数
    clip_range=0.2,          # PPO裁剪范围
    ent_coef=0.01,           # 熵系数，促进探索
    vf_coef=0.5,             # 价值函数系数
    max_grad_norm=0.5,       # 梯度裁剪
    device='cuda',           # 使用GPU训练
    policy_kwargs=dict(
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],  # 更大、更分离的网络结构
        activation_fn=torch.nn.ReLU
    )
)

# print("开始训练PPO模型...")
# model.learn(total_timesteps=50000, callback=eval_callback)
# print("训练完成！")

# model.save('C:/Users/GuoKai/Desktop/python_demo/PI自整定-强化学习/PI整定-SAC-simulink/SAC_PI_Mode.zip')
model = PPO.load('C:/Users/GuoKai/Desktop/python_demo/PI自整定-强化学习/PI整定-SAC-simulink/SAC_PI_Mode.zip')

print("\n开始评估模型...")
n_eval_episodes = 10 # 评估10个回合
episode_rewards = []

# 创建带渲染的环境进行可视化评估
visual_eval_env = gym.make("Pendulum-v1", render_mode='human')

for episode in range(n_eval_episodes):
    obs, info = visual_eval_env.reset()
    done = False
    total_reward = 0
    steps = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = visual_eval_env.step(action)
        total_reward += reward
        steps += 1
        if done or truncated:
            break

    episode_rewards.append(total_reward)
    print(f"评估回合 {episode + 1}: 奖励 = {total_reward:.2f}, 步数 = {steps}")

# 计算并打印平均评估奖励
mean_reward = np.mean(episode_rewards)
std_reward = np.std(episode_rewards)
print(f"\n模型在 {n_eval_episodes} 个评估回合中的平均奖励: {mean_reward:.2f} +/- {std_reward:.2f}")

# 展示最终评估结果的奖励图
plt.figure(figsize=(10, 5))
plt.bar(range(1, n_eval_episodes + 1), episode_rewards)
plt.xlabel('episode')
plt.ylabel('total reward')
plt.title('PPO algorithm in Pendulum-v1 environment')
plt.grid(axis='y', linestyle='--')
plt.show()

# 关闭环境
env.close()
eval_env.close()
visual_eval_env.close()
