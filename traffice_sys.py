import numpy as np
import vehicles
import random as rd
import math

TICK = 0
THETA = 0.11
INTERVAL = 1

def throw():
    result = exponential(THETA)
    dice = rd.random()
    for i in range(10):
        if dice < result[i]:
            return i
# 构建反指数分布
def exponential(k):
    n = np.arange(10)
    expon = []
    for item in n:
        expon.append(1 - math.exp(-(k * item)))
    return expon

def reset():
    env = Traffic()
    step_num = rd.randint(1, 50)
    for i in range(step_num):
        env.normal_actions(i)
        next_state, reward, done, _ = env.step()
    return env

def recover_vehicle(info):
    vehicle = vehicles.Vehicles()
    vehicle.info = info
    return vehicle

class Traffic():

    def __init__(self):
        self.roads = []
        self.action = [0,0,0,0]
        self.state = [] # lines length
        self.delay = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]

        self.create_traffic_net()
        self.create_traffic_signals()

    def create_traffic_net(self):
        # 井字形路网，12条6车道大马路~
        for _ in range(24):
            self.roads.append([[],[],[]])

    def create_traffic_signals(self):
        self.lights = [[0,12,11,7],[1,2,15,8],[13,10,5,6],[9,3,4,14]]
        self.str_dir = [[13,23,16,8],[9,12,17,18],[21,22,11,14],[20,10,15,19]]
        # self.left_dir = [[8,13,23,16],[18,9,12,17],[14,21,22,11],[19,20,10,15]]
        # self.right_dir = [[23,16,8,13],[12,17,18,9],[22,11,14,21],[10,15,19,20]]
        # self.lights_info = [40, 30, 20] # 直行，左转，右转路程

    def generate_vehicles(self):
        for i in range(8):
            for j in range(3):
                if self.delay[i][j] != 0:
                    self.delay[i][j] -= 1
                    break
                dice = throw()
                if dice == None: dice = 10
                if dice > INTERVAL:
                    self.roads[i][j].append(vehicles.Vehicles())
                    self.delay[i][j] = self.delay[i][j] - 1 + dice

    def step(self):
        # part1: generation.
        self.generate_vehicles()
        # part2: step forward.
        for i in range(16):
            # find property signal
            for j in range(4):
                for k in range(4):
                    if self.lights[j][k] == i:
                        mark = [j,k]

            action = self.action[mark[0]]
            str_direction = self.str_dir[mark[0]][mark[1]]
            # left_direction = self.left_dir[mark[0]][mark[1]]
            # right_direction = self.right_dir[mark[0]][mark[1]]
            if mark[1] % 2 == 0: # N--S direction
                if action == 0:
                    red = True
                elif action == 1:
                    red = False
                else:
                    red = True
            else:
                if action == 1:
                    red = True
                elif action == 0:
                    red = False
                else:
                    red = True
            # 直行
            for go_line in range(3):
                front_one = None
                if len(self.roads[i][go_line]) != 0:
                    bellwether = self.roads[i][go_line][0]
                    if bellwether.info[3] > 300 and red != True and len(self.roads[str_direction][go_line]) != 0:
                        bottom = self.roads[str_direction][go_line][-1]
                        bellwether.info[0] = bottom.info[1]
                        bellwether.info[2] = bottom.info[3] + 400
                    elif bellwether.info[3] > 300 and red != True:
                        bellwether.info[0] = 0
                        bellwether.info[2] = 800
                    else:
                        bellwether.info[0] = 0
                        bellwether.info[2] = 400

                for n in self.roads[i][go_line]:
                    n.refollow(front_one)
                    n.update_speed()
                    n.move()
                    n.history()
                    front_one = n

                for each in self.roads[i][go_line]:
                    if each.flow():
                        each.info[3] -= 440
                        self.roads[str_direction][go_line].append(each)
                        self.roads[i][go_line].remove(each)
            # turn left.
            # turn right just turn.
        for park in range(16,24):
            for park_line in range(3):
                one = None
                for m in self.roads[park][park_line]:
                    m.update_speed()
                    m.move()
                    if one == None:
                        m.info[0] = 0
                        m.info[2] = 1000
                    else:
                        m.refollow(one)
                        one = m
        # part3: update state.
        total_avg_speed = 0
        vehicles_num = 0
        total_waiting_tick = 0
        self.state = np.zeros(16)
        for i in range(16):
            length = 0
            for each_line in range(3):
                for ve in self.roads[i][each_line]:
                    if ve.info[1] < 0.3:
                        length += 1
                    total_avg_speed += (ve.info[6][1] / ve.info[6][0])
                    vehicles_num += 1
                    total_waiting_tick += ve.info[6][2]
            self.state[i] = length
        sp_info = []
        for i in range(24):
            num = 0
            for l in range(3):
                num += len(self.roads[i][l])
            sp_info.append(num)
        # part4: return next_state, reward, done, _.
        reward = total_avg_speed / vehicles_num
        if reward > 2: done = False
        else: done = True
        return self.state, reward, done, sp_info

    def normal_actions(self, tick):
        if tick % 6 < 3:
            self.action = [0,0,1,1]
        elif tick % 6 <= 6:
            self.action = [1,1,0,0]


if __name__ == '__main__':
    r = 0
    en = reset()
    for i in range(100):
        en.normal_actions(i)
        for _ in range(10):
            en.step()
        if i % 3 == 0:
            en.action = [-1,-1,-1,-1]
            for _ in range(3):
                en.step()
            next_state, reward, done, info = en.step()
        r += reward
    print(en.state)
    r = r / 100
    print(reward)
    print(r)
    print(info)
    num = 0
    sum_time = 0
    for i in range(16,24):
        for j in range(3):
            for car in en.roads[i][j]:
                num += 1
                sum_time += car.info[6][2]
    print(sum_time/num)

