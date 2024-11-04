# train.py
import torch
from snake_env import SnakeEnv
from model import DQN
from replay_memory import ReplayMemory, Transition
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pygame
from torch.utils.tensorboard import SummaryWriter

# 預處理函數
def preprocess(state):
    # 將狀態轉換為 PyTorch tensor 並正規化
    state = torch.from_numpy(state).float()
    return state

# 測試函數
def test_model(env, model, device='cpu'):
    state = env.reset()
    print("Environment reset.")
    state = preprocess(state).unsqueeze(0).to(device)
    done = False
    total_reward = 0
    while not done:
        with torch.no_grad():
            q_values = model(state)
            action = q_values.max(1)[1].item()
        print(f"Action chosen: {action}")  # 調試打印
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = preprocess(next_state).unsqueeze(0).to(device)
        env.render()
    print(f"Test Game - Total Reward: {total_reward}")
    
    # 保持窗口打開直到用戶手動關閉
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return

# 訓練函數
def train_dqn(env, num_episodes=1000, batch_size=64, gamma=0.94, 
             epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=3000, 
             target_update=10, memory_capacity=50000, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_actions = env.action_space.n
    input_size = env.observation_space.shape[0]
    
    policy_net = DQN(input_size, n_actions).to(device)
    target_net = DQN(input_size, n_actions).to(device)
    
    highest_score = 0
    
    #加載已保存的模型
    try:
        policy_net.load_state_dict(torch.load("dqn_snake_model.pth", weights_only=True))
        print("Loaded saved model parameters.")
    except FileNotFoundError:
        print("No saved model found. Training from scratch.")
    
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = ReplayMemory(memory_capacity)
    
    steps_done = 0
    
    # 初始化 TensorBoard
    writer = SummaryWriter('runs/snake_dqn')
    
    for episode in range(num_episodes):
        state = env.reset()
        state = preprocess(state).to(device)
        total_reward = 0
        done = False
        
        while not done:
            # Epsilon-greedy 策略
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                      np.exp(-1. * steps_done / epsilon_decay)
            steps_done += 1
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = policy_net(state.unsqueeze(0))
                    action = q_values.max(1)[1].item()
            
            # 執行動作
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            next_state = preprocess(next_state)
            
            # 儲存轉移
            memory.push(state, action, reward, next_state, done)
            state = next_state.to(device)
            
            # 訓練步驟
            if len(memory) >= batch_size:
                transitions = memory.sample(batch_size)
                batch = Transition(*zip(*transitions))
                
                # 將批次中的元素轉換為張量
                non_final_mask = ~torch.tensor(batch.done, dtype=torch.bool).to(device)
                non_final_next_states = torch.stack([s for s, d in zip(batch.next_state, batch.done) if not d]).to(device)
                
                state_batch = torch.stack(batch.state).to(device)
                action_batch = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).to(device)
                reward_batch = torch.tensor(batch.reward, dtype=torch.float).to(device)
                
                # 計算 Q(s_t, a)
                state_action_values = policy_net(state_batch).gather(1, action_batch)
                
                # Double DQN
                with torch.no_grad():
                    # 使用策略網路選擇動作
                    next_actions = policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                    # 使用目標網路評估這些動作的 Q 值
                    next_state_values = torch.zeros(batch_size, device=device)
                    next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_actions).squeeze()
                
                # 計算期望的 Q 值
                expected_state_action_values = reward_batch + (gamma * next_state_values)
                
                # 計算損失
                loss = F.smooth_l1_loss(state_action_values.squeeze(), expected_state_action_values)
                
                # 優化網路
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 記錄損失到 TensorBoard
                writer.add_scalar('Loss', loss.item(), steps_done)
        
        # 每個 episode 結束後記錄獎勵到 TensorBoard
        writer.add_scalar('Total Reward', total_reward, episode)
        
        # 每隔 target_update 個 episode 更新一次目標網路
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        if episode % 100 == 0:
            torch.save(policy_net.state_dict(), "dqn_snake_model.pth")
            print(f"Model saved at episode {episode}.")
        
        score = env.score    
        if score > highest_score:
            highest_score = score
        
        print(f"Episode {episode} - Score: {score} - Hight Score {highest_score} - Total Reward: {int(total_reward)}")
    
    writer.close()
    print("訓練完成")
    return policy_net

# 主程式
if __name__ == "__main__":
    # 訓練時啟用渲染
    env = SnakeEnv(grid_size=12, block_size=40, render_mode=True)
    trained_model = train_dqn(env, num_episodes=10000)
    
    # 保存模型
    torch.save(trained_model.state_dict(), "dqn_snake_model.pth")
