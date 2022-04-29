import sys
sys.path.append('')
from main import *
import harfang as hg
import json
import random, time
from util import *
import numpy as np
import torch



class UAV:
    resolution = hg.Vector2(1600, 900)
    antialiasing = 2
    screenMode = hg.FullscreenMonitor1

    main_node = hg.Node()

    controller = None

    scene = None
    camera = None
    satellite_camera = None
    camera_matrix = None
    camera_v_move = hg.Vector3(0, 0, 0)  # Camera velocity for sfx
    fps = None
    sea_render = None
    ligth_sun = None
    ligth_sky = None

    sea_render_script = None
    clouds_render_script = None

    water_reflection = None

    p1_aircraft = None
    p2_aircraft = None

    p1_success = False

    carrier = None
    carrier_radar = None
    island = None

    p1_missiles = [None] * 4
    p2_missiles = [None] * 4
    p1_missiles_smoke_color = hg.Color(1, 1, 1, 1)
    p2_missiles_smoke_color = hg.Color(1, 1, 1, 1)

    p1_targets = []

    bullets = None
    ennemy_bullets = None

    title_font = "../source/assets/fonts/destroy.ttf"
    hud_font = "../source/assets/fonts/Furore.otf"
    texture_hud_plot = None
    texture_noise = None

    fading_cptr = 0
    fading_start_saturation = 0
    fadout_flag = False
    fadout_cptr = 0

    audio = None
    p1_sfx = None
    p2_sfx = None

    title_music = 0
    title_music_settings = None

    clouds = None
    render_volumetric_clouds = True

    show_debug_displays = False
    display_gui = False

    satellite_view = False

    HSL_postProcess = None
    MotionBlur_postProcess = None
    RadialBlur_postProcess = None

    flag_MotionBlur = False

    radial_blur_strength = 0.5
    deceleration_blur_strength = 1 / 6
    acceleration_blur_strength = 1 / 3

    gun_sight_2D = None

    current_view = None



class DogfightEnv():
    """
    action_space = 11 dim one hot
    action = {
            thrust level up:0, thrust level down: 1, Pitch up:2, pitch down:3
            roll up: 4, roll down:5, yaw up: 6, yaw down: 7
            gun: 8, missile: 9, post combustion: 10, noop: 11
    }
    """
    def __init__(self, renderless=False, seed=7, max_episode_length=1000, action_repeat=4):
        self.symbolic = renderless
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.plus = hg.GetPlus()
        self.action_space_size = 12
        self.steps = 0
        # self.action_space_size = 11
        hg.LoadPlugins()
        hg.MountFileDriver(hg.StdFileDriver())
        _, scr_mod, scr_res = request_screen_mode()
        UAV.resolution.x, UAV.resolution.y = scr_res.x, scr_res.y
        # UAV.resolution.x, UAV.resolution.y = 10, 10
        UAV.screenMode = scr_mod

        self.plus.RenderInit(int(UAV.resolution.x), int(UAV.resolution.y), UAV.antialiasing, UAV.screenMode)
        self.plus.SetBlend2D(hg.BlendAlpha)
        self.plus.SetBlend3D(hg.BlendAlpha)
        self.plus.SetCulling2D(hg.CullNever)

        # ------------ Add loading screen -------------
        self.plus.Clear()
        self.plus.Flip()
        self.plus.EndFrame()
        # ---------------------------------------------

        init_game(UAV, self.plus)

        # plus.UpdateScene(UAV.scene)
        UAV.scene.Commit()
        UAV.scene.WaitCommit()

    def reset(self):
        self.steps = 0
        init_start_phase(UAV, self.plus)

        delta_t = self.plus.UpdateClock()
        # dts = hg.time_to_sec_f(delta_t)

        _, init_state = start_phase(UAV, self.plus, delta_t)
        self.plus.Flip()
        self.plus.EndFrame()
        return self.state_normalize(init_state[:-6])


    def step(self, action):
        reward = 0
        done = False
        delta_t = self.plus.UpdateClock()
        for k in range(self.action_repeat):
            # action repeat时不重复发射导弹
            # if k > 0 and action[9] == 1: action[9] = 0
            if k > 0 and action == 9: action = 11
            # dts = hg.time_to_sec_f(delta_t)
            p1_health, p2_health = UAV.p1_aircraft.health_level, UAV.p2_aircraft.health_level
            # end, next_state = main_phase(UAV, self.plus, delta_t, action.tolist())
            end, next_state = main_phase(UAV, self.plus, delta_t, action)
            self.steps += 1
            _p1_health, _p2_health = UAV.p1_aircraft.health_level, UAV.p2_aircraft.health_level
            # calculate reward
            reward += (cal_r_time() +
                       cal_r_speed(next_state) +
                       cal_r_alti(next_state) +
                       # cal_r_fire(action[9], action[8]) +
                       cal_r_fire(action, next_state) +
                       cal_r_health(p1_health, p2_health, _p1_health, _p2_health) +
                       cal_r_missile(next_state))
            if (self.steps > 1000 and next_state[16] == 1) or self.steps > 10000:
                reward += -100
                done = True
                break
            # print(f"""
            #         cal_r_time() {cal_r_time()}\n
            #         cal_r_speed(next_state) {cal_r_speed(next_state)}\n
            #         cal_r_alti(next_state) {cal_r_alti(next_state)}\n
            #         cal_r_fire(action[9], action[8]) {cal_r_fire(action)}\n
            #         cal_r_health(p1_health, p2_health, _p1_health, _p2_health) {cal_r_health(p1_health, p2_health, _p1_health, _p2_health)}\n
            #         cal_r_missile(next_state) {cal_r_missile(next_state)}\n
            #         """)
            # print(next_state[:-9])
            if end == True:
                done = True
                break
        self.plus.Flip()
        self.plus.EndFrame()
        return self.state_normalize(next_state[:-6]), reward, done

    def sample_random_action(self):
        # return np.random.randint(0, 2, size=self.action_space_size)
        return np.random.randint(0, 12)

    def state_normalize(self, state):
        normal = np.array([
            1000, 1000, 1000, 300, 100, 300, 10, 1, 360, 90, 90, 4, 1000, 360, 90, 1, 1,
            1000, 1000, 1000, 300, 100, 300, 360, 1, 1000, 1000, 1000, 1, 1000, 1000, 1000, 1, 1000, 1000, 1000, 1, 1000, 1000, 1000
        ])
        state = np.array(state) / normal
        return state


def main():
    env = DogfightEnv()
    for i in range(10):
        init = env.reset()
        done = False
        step = 0
        while not done:
            print(step)
            # action = env.sample_random_action()
            step+=1
            next_state, _, done = env.step(action)
            # print(next_state)
            state = next_state
            print(state[7])
        print(step)



if __name__ == '__main__':
    main()

        # print(f"""
        #
        #
        #     """)
        # print(_state)
        # print(action)

    # env = gym.make('CartPole-v0')
    # print(env.observation_space)

