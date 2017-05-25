import math

LENGTH = 400
V0 = 10
L = 5
VMAX = 15 # 限速
AMAX = 2 # 最大加速度
ROUTINE = [1,1]

class Vehicles(object):

    def __init__(self):
        self.info = [0,10,405,0,L,ROUTINE,[0, 0, 0]] # [v0, v1, s0, s1, length, routine]

    def initialize_routine(self, birth, destination):
        self.info[5] = birth
        self.info[6] = destination

    def update_speed(self):
        v0 = self.info[0]
        v1 = self.info[1]
        s0 = self.info[2]
        s1 = self.info[3]

        v_a = v1 + 2.5 * AMAX * (1 - v1 / VMAX) * math.sqrt(0.025 + v1 / VMAX)

        if v0 >= 0:
            temp = (AMAX ** 2) + AMAX * (2 * (s0 - 5 - s1) - v1 + (v0 ** 2) / AMAX)
            if temp < 0:
                v_b = 0
            else:
                v_b = -AMAX + math.sqrt(temp)
            v = min(v_b, v_a, VMAX)
        else:
            v = v_a
        if v < 0:
            v = 0
        self.info[1] = v
        return v

    def move(self):
        self.info[3] += self.info[1]

    def flow(self):
        if self.info[3] > 400:
            return True
        else:
            return False

    def refollow(self, front_vehicle):
        if front_vehicle != None:
            self.info[0] = front_vehicle.info[1]
            self.info[2] = front_vehicle.info[3]
        # what if front vehicle has gone???

    def history(self):
        self.info[6][0] += 1 # time_sum
        self.info[6][1] += self.info[1] # speed_sum
        if self.info[1] <= 1 and self.info[3] >= 200:
            self.info[6][2] += 1







