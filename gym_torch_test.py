import gymnasium as gym
import torch

from gym_torch_classes import RLAgent


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# How to create an environment from scratch
# How to use the created policy

# env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("CartPole-v1")


rl_agent = RLAgent(env, plot=True, render=False,
                   num_episodes=128, BATCH_SIZE=32)

rl_agent.run()
# rl_agent.print_model()

env1 = gym.make("CartPole-v1", render_mode="human")
episodes = 10

for epi in range(1, episodes):

    state, info = env1.reset()
    state = torch.tensor(state, dtype=torch.float32,
                         device=device).unsqueeze(0)

    done = False
    score = 0
    while not done:

        # action = rl_agent.select_action(state)
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            action = rl_agent.policy_net(state).max(1)[1].view(1, 1)

        state, reward, terminated, truncated, _ = env1.step(action.item())
        state = torch.tensor(state, dtype=torch.float32,
                             device=device).unsqueeze(0)
        done = terminated or truncated

        env1.render()
        score += reward

    print(f"Episode {epi}: score = {score}")

env1.close()
