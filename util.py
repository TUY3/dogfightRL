import math

# 角度转换
def angle_trans(x):
    return x / 180 * math.pi


# 计算偏航角
def cal_yaw(vx, vz):
    yaw = 0
    if vz != 0:
        # yaw=180 / math.pi * math.atan(Main.p1_aircraft.v_move.x / Main.p1_aircraft.v_move.z)
        # print(yaw)
        if vz > 0:
            yaw = 180 / math.pi * math.atan(vx / vz)
        elif vx > 0:
            yaw = 180 / math.pi * math.atan(vx / vz) + 180
        else:
            yaw = 180 / math.pi * math.atan(vx / vz) - 180
    return yaw

class AircraftState:
    def __init__(self, state):
        # p1
        self.p1_posX = state[0]
        self.p1_posZ = state[1]
        self.p1_altitude = state[2]
        self.p1_hspeed = state[3]
        self.p1_vspeed = state[4]
        self.p1_linearspeed = state[5]
        self.p1_linearacc = state[6]
        self.p1_health_level = state[7]
        self.p1_cap = state[8]
        self.p1_pitch = state[9]
        self.p1_roll = state[10]
        self.p1_numMissiles = state[11]
        self.p1_target_dis = state[12]
        self.p1_target_cap = state[13]
        self.p1_target_angle = state[14]
        self.p1_target_lockingstate = state[15]
        self.p1_landed = state[16]
        # p2
        self.p2_posX = state[17]
        self.p2_posZ = state[18]
        self.p2_altitude = state[19]
        self.p2_hspeed = state[20]
        self.p2_vspeed = state[21]
        self.p2_linearspeed = state[22]
        self.p2_cap = state[23]
        self.p2_missile0_state = state[24]
        self.p2_missile0_posX = state[25]
        self.p2_missile0_posZ = state[26]
        self.p2_missile0_posY = state[27]
        self.p2_missile1_state = state[28]
        self.p2_missile1_posX = state[29]
        self.p2_missile1_posZ = state[30]
        self.p2_missile1_posY = state[31]
        self.p2_missile2_state = state[32]
        self.p2_missile2_posX = state[33]
        self.p2_missile2_posZ = state[34]
        self.p2_missile2_posY = state[35]
        self.p2_missile3_state = state[36]
        self.p2_missile3_posX = state[37]
        self.p2_missile3_posZ = state[38]
        self.p2_missile3_posY = state[39]
        self.p2_target_dis = state[40]
        self.p2_target_cap = state[41]
        self.p2_target_alti = state[42]
        self.p2_target_angle = state[43]
        self.p2_target_lockingstate = state[44]
        self.p2_health_level = state[45]


# reward
coef_alti = 0.00002
coef_health = 100
coef_fire = 1
coef_miss_dis = 0.000001
coef_speed = 0.0001
coef_locked = 0.02
# 对抗
# 1.导弹：锁定/进入锁定范围
#        与导弹之间的距离
def cal_r_missile(state):
    state = AircraftState(state)
    reward = 0
    # 与导弹之间的距离
    dis_miss0 = math.sqrt(pow(state.p1_posX - state.p2_missile0_posX, 2) +
                          pow(state.p1_altitude - state.p2_missile0_posY, 2) +
                          pow(state.p1_posZ - state.p2_missile0_posZ, 2))
    dis_miss1 = math.sqrt(pow(state.p1_posX - state.p2_missile1_posX, 2) +
                          pow(state.p1_altitude - state.p2_missile1_posY, 2) +
                          pow(state.p1_posZ - state.p2_missile1_posZ, 2))
    dis_miss2 = math.sqrt(pow(state.p1_posX - state.p2_missile2_posX, 2) +
                          pow(state.p1_altitude - state.p2_missile2_posY, 2) +
                          pow(state.p1_posZ - state.p2_missile2_posZ, 2))
    dis_miss3 = math.sqrt(pow(state.p1_posX - state.p2_missile3_posX, 2) +
                          pow(state.p1_altitude - state.p2_missile3_posY, 2) +
                          pow(state.p1_posZ - state.p2_missile3_posZ, 2))
    if dis_miss0 < 1500 or dis_miss1 < 1500 or dis_miss2 < 1500 or dis_miss3 < 1500:
        reward += ((dis_miss0 - 1500) * state.p2_missile0_state +
                   (dis_miss1 - 1500) * state.p2_missile1_state +
                   (dis_miss2 - 1500) * state.p2_missile2_state +
                   (dis_miss3 - 1500) * state.p2_missile3_state)
    # if state.p2_missile0_state or state.p2_missile1_state or state.p2_missile2_state or state.p2_missile3_state:
    #     reward /= (state.p2_missile0_state + state.p2_missile1_state + state.p2_missile2_state + state.p2_missile3_state)
    reward = reward * coef_miss_dis
    # 锁定与被锁定
    reward += -state.p2_target_lockingstate * coef_locked
    reward += state.p1_target_lockingstate * coef_locked
    return reward

# 2.子弹：进入攻击范围
def cal_r_bullet(state: AircraftState):
    reward = 0
    return reward
# 3.血量
def cal_r_health(p1_health, p2_health, _p1_health, _p2_health):
    reward = (_p1_health - p1_health) + (p2_health - _p2_health)
    return reward * coef_health

# 4.发射导弹/子弹
# def cal_r_fire(a_missile, a_bullet):
#     reward = 0
#     if a_missile: reward += -0.1
#     if a_bullet: reward += -0.01
#     return reward * coef_fire

def cal_r_fire(action, state):
    state = AircraftState(state)
    reward = 0
    if action == 9: reward += (-1 / (state.p1_numMissiles + 4))
    if action == 8: reward += -0.001
    return reward * coef_fire


# 飞控
# 1.高度 安全高度2000/最小高度500/最大高度10000
def cal_r_alti(state):
    state = AircraftState(state)
    reward = 0
    if state.p1_altitude < 500:
        reward += (state.p1_altitude - 500)
    elif state.p1_altitude < 2000:
        reward += (state.p1_altitude - 2000) * 0.1
    elif state.p1_altitude > 10000:
        reward += (10000 - state.p1_altitude)
    return reward * coef_alti

# 2.速度
def cal_r_speed(state):
    state = AircraftState(state)
    reward = 0
    if state.p1_linearspeed < 80 and not state.p1_landed:
        reward += (state.p1_linearspeed - 80)
    elif state.p1_linearspeed < 50 and state.p1_landed:
        reward += (state.p1_linearspeed - 50)
    elif state.p1_linearspeed > 340:
        reward += (340 - state.p1_linearspeed)
    return reward * coef_speed
# 3.时间
def cal_r_time():
    return 0.01





