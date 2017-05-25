import traffice_sys
import tensorflow as tf
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

GAMMA = 0.9
INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.01
REPLAY_SIZE = 10000
BATCH_SIZE = 32


class DQN():
    def __init__(self):
        self.replay_buffer = deque() # 新建缓存历史队列
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = 16
        self.action_dim = 16

        self.create_Q_network()
        self.create_training_method()

        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def create_Q_network(self):
        # 神经网络权重
        W1 = self.weight_variable([self.state_dim, 20])    # 权重的维度 {state_dim：输入神经元， 20：hidden layer 神经元数量}
        b1 = self.bias_variable([20])                      # 偏置向量维度
        W2 = self.weight_variable([20, self.action_dim])
        b2 = self.bias_variable([self.action_dim])
        # 输入层
        self.state_input = tf.placeholder("float", [None, self.state_dim])  # 定义容器存放输入（张量）

        h_layer = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)  # input --> hidden
        # Q network
        self.Q_value = tf.matmul(h_layer, W2) + b2                  # hidden --> output(Q_value)

    # 为了避免神经元输出恒为0的问题，加入少量噪声来打破对称，避免0梯度。通常用一个较小的正数来初始化偏置项
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)
    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)
    # ----------------------------------------------------------------------------------------------

    def create_training_method(self):
        self.action_input = tf.placeholder("float", [None, self.action_dim])
        self.y_input = tf.placeholder("float", [None])
        Q_action = tf.reduce_sum(tf.mul(self.Q_value, self.action_input), reduction_indices = 1)
        # cost function J(theta)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        # optimize J(theta)
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def train_Q_network(self):
        self.time_step += 1
        # obtain random minibatch from replay memory
        # 随机抽取避免出现时序性
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        # calculate y
        y_batch = []
        # 每一次批量train，update state.
        Q_value_batch = self.Q_value.eval(feed_dict = {self.state_input:next_state_batch})
        for i in range(0,BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                # Q(s, a) = Reward + max(Q(next_state, a))
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))
                # 由此慢慢探索，get到label，继而能够去更新cost，从而优化Q_network

        self.optimizer.run(feed_dict = {
            self.y_input:y_batch,
            self.action_input:action_batch,
            self.state_input:state_batch})

    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1  # [1, 0] & [0, 1] means two different actions.
        # input some situations into replay buffer.
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()
        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

    def egreedy_action(self, state):
        try:
            Q_value = self.Q_value.eval(feed_dict = {self.state_input:[state]})[0]
        except:
            print(state)
            return 0
        if random.random() <= self.epsilon:
            # explore new routine.
            return random.randint(0, self.action_dim - 1)
        else:
            # choose the best known routine.
            return np.argmax(Q_value)
        # self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000

    def action(self, state):
        try:
            best_action =  np.argmax(self.Q_value.eval(feed_dict = {self.state_input:[state]})[0])
            return best_action
        except:
            print(state)
            return 0

EPISODE = 2810
STEP = 100


def main():
    agent = DQN()
    x_episode = []
    y_v = []
    y_wait = []
    y_length = []


    for episode in range(EPISODE):
        print('=========NO. %d Episode==========' % episode)
        env = traffice_sys.reset()
        avg_reward = 0

        for step in range(STEP):
            # print(env.state)
            action = agent.egreedy_action(env.state)
            new_action = action_transform(action)
            all_red = []
            for i in range(4):
                if env.action[i] == new_action[i]:all_red.append(new_action[i])
                else:all_red.append(-1)
            state = env.state
            env.action = new_action
            for _ in range(10):
                env.step()
            env.action = all_red
            for _ in range(3):
                next_state, reward, done, _ = env.step()
            env.action = new_action
            avg_reward += reward
            agent.perceive(state, action, reward, next_state, done)
            if done:
                break
        avg_reward = avg_reward / STEP
        print(avg_reward)

        if (episode % 2800 == 0):
            print('-=-=-=-=-= TEST -=-=-=-=-')
            total_reward = 0
            avg_length = 0
            avg_waiting_time = 0
            for i in range(3):
                avg_reward = 0
                queue_length = 0
                env = traffice_sys.reset()
                for j in range(STEP):
                    action = agent.action(env.state)
                    new_action = action_transform(action)
                    env.action = new_action
                    all_red = []
                    for i in range(4):
                        if env.action[i] == new_action[i]:
                            all_red.append(new_action[i])
                        else:
                            all_red.append(-1)
                    for _ in range(10):
                        env.step()
                    env.action = all_red
                    for _ in range(3):
                        next_state, reward, done, _ = env.step()
                        queue_length += (sum(next_state)/3)
                    env.action = new_action
                    avg_reward += reward
                    if done:
                        break
                queue_length /= STEP
                avg_reward /= STEP
                avg_length += (queue_length / 3)
                total_reward += (avg_reward / 3)

                car_num = 0
                avg_wait = 0
                for i in range(16,24):
                    for j in range(3):
                        for car in env.roads[i][j]:
                            car_num += 1
                            avg_wait += car.info[6][2]
                if car_num != 0:avg_wait /= (car_num * 3)
                avg_waiting_time += avg_wait

            x_episode.append(episode)
            y_v.append(total_reward)
            y_wait.append(avg_waiting_time)
            y_length.append(avg_length / 16)




    plt.figure(1)
    plt.plot(x_episode, y_v, "b", linewidth=1)
    plt.xlabel("episode")
    plt.ylabel("the mean velocity of vehicles")
    plt.title("DQN train process")

    plt.figure(2)
    plt.plot(x_episode, y_wait, "b", linewidth=1)
    plt.ylabel('the mean delay of vehicles')
    plt.xlabel("episode")
    plt.title("DQN train process")

    plt.figure(3)
    plt.plot(x_episode, y_length, "b", linewidth=1)
    plt.ylabel('the mean queue length of vehicles')
    plt.xlabel("episode")
    plt.title("DQN train process")

    plt.show()
    print(x_episode)
    print('====')
    print(y_length)
    print('====')
    print(y_wait)
    print('====')
    print(y_v)

def action_transform(action):
    switch = {
        0:[0,0,0,0],1:[0,0,0,1],
        2:[0,0,1,0],3:[0,1,0,0],
        4:[1,0,0,0],5:[0,0,1,1],
        6:[0,1,0,1],7:[1,0,0,1],
        8:[0,1,1,0],9:[1,0,1,0],
        10:[1,1,0,0],11:[0,1,1,1],
        12:[1,1,1,0],13:[1,0,1,1],
        14:[1,1,0,1],15:[1,1,1,1]
    }
    return switch[action]

if __name__ == '__main__':
    main()