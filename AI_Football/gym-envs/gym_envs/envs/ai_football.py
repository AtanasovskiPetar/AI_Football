__credits__ = ["Petar Atanasovski"]

# import math
from os import path
from typing import Optional
import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
import os
import platform
import numpy as np
import pygame
# import time
import random

# from Team_name import Defense as offence_script

game_name = 'AI Football'
fps = 60
dt = 1 / fps

resolution = 1366, 768
resolution_rect = [0, 0, resolution[0], resolution[1]]
# [50, 203, 1366, 768]
ground = [0, int(resolution[1] / 5), resolution[0], resolution[1]]
ground_rect = [ground[0], ground[1], resolution[0], resolution[1]]
# [50, 203, 1316, 718]
playground = [50, 50 + int(resolution[1] / 5), resolution[0] - 50, resolution[1] - 50]
playground_rect = [playground[0], playground[1], playground[2] - playground[0], playground[3] - playground[1]]
half_playground_rect = [playground_rect[0], playground_rect[1], int(playground_rect[2] / 2), playground_rect[3]]
center = [int((playground[2] - playground[0]) / 2) + playground[0],
          int((playground[3] - playground[1]) / 2) + playground[1]]

post_radius = 10
post_screen_top = 343
post_screen_bottom = 578
post_screen_left = playground[0]
post_screen_right = playground[2]

player_1_initial_position = [int((center[0] - playground[0]) / 2) + playground[0] - 10, post_screen_top]
player_2_initial_position = [int((center[0] - playground[0]) / 2) + playground[0] - 10, center[1]]
player_3_initial_position = [int((center[0] - playground[0]) / 2) + playground[0] - 10, post_screen_bottom]
player_4_initial_position = [player_1_initial_position[0] + half_playground_rect[2] + 10, post_screen_top]
player_5_initial_position = [player_2_initial_position[0] + half_playground_rect[2] + 10, center[1]]
player_6_initial_position = [player_3_initial_position[0] + half_playground_rect[2] + 10, post_screen_bottom]

initial_positions_team_left = [player_1_initial_position, player_2_initial_position, player_3_initial_position]
initial_positions_team_right = [player_4_initial_position, player_5_initial_position, player_6_initial_position]

# Colors:
black = [0, 0, 0]
white = [255, 255, 255]
red = [255, 0, 0]
yellow = [255, 255, 0]
green = [0, 255, 0]
sky_blue = [135, 206, 250]
blue = [0, 0, 255]
grass = [1, 142, 14]

cursor_width = 2

ball_restitution = 0.6
player_player_restitution = 0.5
ball_restitution_under_player_control = 0.4
player_post_restitution = 0.5
half_time_duration = 45
no_render_half_time_duration = 0.460640802538643
new_no_render_half_time_duration = 5.11532040126

short_pause_countdown_time = 5
goal_pause_countdown_time = 5

shift = 230
team_left_logo_position = [22 + shift, 2, 153, 153]
team_right_logo_position = [527 + shift, 2, 153, 153]
team_left_color_position = [173 + shift, 0, 50, 153]
team_right_color_position = [477 + shift, 0, 50, 153]
post_mass = 1e99

# Player stats: Kylian Mbappé, Joshua Kimmich, Joško Gvardiol
weights = [73, 75, 80]
radiuses = [22, 23, 24]
accelerations = [100, 60, 70]
speeds = [100, 60, 85]
shot_powers = [65, 70, 50]


class Circle:
    def __init__(self, x=0, y=0, radius=0, mass=1, alpha=0):
        self.x = x
        self.y = y
        self.radius = radius
        self.mass = mass
        self.alpha = alpha
        self.v = 0


class Player(Circle):
    a_fifa = 0.75
    v_fifa = 0.88
    shot_power_fifa = 0.95
    a_max_coeff = 22
    v_max_coeff = 7
    shot_power_max_coeff = 200
    a_max = a_max_coeff * a_fifa
    v_max = v_max_coeff * v_fifa
    shot_power_max = shot_power_max_coeff * shot_power_fifa
    shot_power = shot_power_max
    shot_request = False

    def __init__(self, name, weight, radius, acceleration, speed, shot_power):
        self.name = name
        self.mass = int(weight)
        self.radius = int(radius)
        self.a_fifa = int(acceleration)
        self.v_fifa = int(speed)
        self.shot_power_fifa = int(shot_power)
        self.v_max = self.v_max_coeff * self.v_fifa
        self.a_max = self.a_max_coeff * self.a_fifa
        self.shot_power_max = self.shot_power_max_coeff * self.shot_power_fifa

    def move(self, manager_decision):
        force = np.clip(manager_decision['force'], -0.5 * self.a_max * self.mass, self.a_max * self.mass)
        self.alpha = manager_decision['alpha']
        self.shot_power = np.clip(manager_decision['shot_power'], 0, self.shot_power_max)
        self.shot_request = manager_decision['shot_request']
        self.v += force / self.mass * dt
        self.v = np.clip(self.v, 0, self.v_max)
        self.x += np.cos(self.alpha) * self.v * dt
        self.y += np.sin(self.alpha) * self.v * dt
        self.x = np.clip(self.x, ground_rect[0], ground_rect[2])
        self.y = np.clip(self.y, ground_rect[1], ground_rect[3])

    def draw(self, screen, color):
        pygame.draw.circle(screen, color, [int(self.x), int(self.y)], self.radius)
        new_x = self.x + self.radius * np.cos(self.alpha)
        new_y = self.y + self.radius * np.sin(self.alpha)
        pygame.draw.line(screen, black, [self.x, self.y], [new_x, new_y], cursor_width)

    def data(self):
        player_data = {'x': self.x, 'y': self.y, 'alpha': self.alpha,
                       'mass': self.mass, 'radius': self.radius,
                       'a_max': self.a_max, 'v_max': self.v_max, 'shot_power_max': self.shot_power_max,
                       }
        return player_data

    def snelius(self):
        if self.y + self.radius >= ground[3] and np.sin(self.alpha) > 0:
            self.alpha = -self.alpha
            self.v *= np.abs(np.cos(self.alpha))
        if self.y - self.radius <= ground[1] and np.sin(self.alpha) < 0:
            self.alpha = -self.alpha
            self.v *= np.abs(np.cos(self.alpha))
        if self.x + self.radius >= ground[2] and np.cos(self.alpha) > 0:
            self.alpha = np.pi - self.alpha
            self.v *= np.abs(np.sin(self.alpha))
        if self.x - self.radius <= ground[0] and np.cos(self.alpha) < 0:
            self.alpha = -np.pi - self.alpha
            self.v *= np.abs(np.sin(self.alpha))

    def reset(self, initial_position, alpha):
        self.x = initial_position[0]
        self.y = initial_position[1]
        self.alpha = alpha
        self.v = 0

    def clip_velocity(self):
        self.v = np.clip(self.v, 0, self.v_max)


class Ball(Circle):
    v_max = 850
    radius = 15
    mass = 0.5

    def move(self):
        self.x += np.cos(self.alpha) * self.v * dt
        self.y += np.sin(self.alpha) * self.v * dt
        self.v *= 0.99

    def draw(self, screen):
        pygame.draw.circle(screen, black, [int(self.x), int(self.y)], self.radius)
        pygame.draw.circle(screen, white, [int(self.x), int(self.y)], self.radius - 2)

    def snelius(self):
        goal = post_screen_top < self.y < post_screen_bottom
        if self.y + self.radius >= playground[3] and np.sin(self.alpha) > 0:
            self.alpha = -self.alpha
            self.y = playground[3] - self.radius
        if self.y - self.radius <= playground[1] and np.sin(self.alpha) < 0:
            self.alpha = -self.alpha
            self.y = playground[1] + self.radius
        if self.x + self.radius >= playground[2] and np.cos(self.alpha) > 0 and not goal:
            self.alpha = np.pi - self.alpha
            self.x = playground[2] - self.radius
        if self.x - self.radius <= playground[0] and np.cos(self.alpha) < 0 and not goal:
            self.alpha = -np.pi - self.alpha
            self.x = playground[0] + self.radius

    def reset(self):
        self.x = center[0]
        self.y = center[1]
        self.alpha = 0
        self.v = 0

    def data(self):
        ball_data = {'x': self.x, 'y': self.y, 'alpha': self.alpha, 'mass': self.mass, 'radius': self.radius}
        return ball_data

    def clip_velocity(self):
        self.v = np.clip(self.v, 0, self.v_max)


class Post(Circle):
    def draw(self, screen):
        pygame.draw.circle(screen, white, [int(self.x), int(self.y)], self.radius)


def collision(circle_1, circle_2):
    return (circle_1.x - circle_2.x) ** 2 + (circle_1.y - circle_2.y) ** 2 <= (circle_1.radius + circle_2.radius) ** 2


def resolve_collision(circle_1, circle_2):
    collision_angle = np.arctan2(circle_2.y - circle_1.y, circle_2.x - circle_1.x)

    new_x_speed_1 = circle_1.v * np.cos(circle_1.alpha - collision_angle)
    new_y_speed_1 = circle_1.v * np.sin(circle_1.alpha - collision_angle)
    new_x_speed_2 = circle_2.v * np.cos(circle_2.alpha - collision_angle)
    new_y_speed_2 = circle_2.v * np.sin(circle_2.alpha - collision_angle)

    final_x_speed_1 = ((circle_1.mass - circle_2.mass) * new_x_speed_1
                       + (circle_2.mass + circle_2.mass) * new_x_speed_2) / (circle_1.mass + circle_2.mass)
    final_x_speed_2 = ((circle_1.mass + circle_1.mass) * new_x_speed_1
                       + (circle_2.mass - circle_1.mass) * new_x_speed_2) / (circle_1.mass + circle_2.mass)
    final_y_speed_1 = new_y_speed_1
    final_y_speed_2 = new_y_speed_2

    cos_gamma = np.cos(collision_angle)
    sin_gamma = np.sin(collision_angle)
    circle_1.v_x = cos_gamma * final_x_speed_1 - sin_gamma * final_y_speed_1
    circle_1.v_y = sin_gamma * final_x_speed_1 + cos_gamma * final_y_speed_1
    circle_2.v_x = cos_gamma * final_x_speed_2 - sin_gamma * final_y_speed_2
    circle_2.v_y = sin_gamma * final_x_speed_2 + cos_gamma * final_y_speed_2

    x_difference = circle_1.x - circle_2.x
    y_difference = circle_1.y - circle_2.y
    d = np.linalg.norm([x_difference, y_difference])

    # minimum translation distance to push balls apart after intersecting
    mtd_x = x_difference * (((circle_1.radius + circle_2.radius) - d) / d)
    mtd_y = y_difference * (((circle_1.radius + circle_2.radius) - d) / d)
    im1 = 1 / circle_1.mass if circle_1.mass > 0 else 0
    im2 = 1 / circle_2.mass if circle_2.mass > 0 else 0

    # push-pull them apart based off their mass
    circle_1.x += mtd_x * (im1 / (im1 + im2))
    circle_1.y += mtd_y * (im1 / (im1 + im2))
    circle_2.x -= mtd_x * (im2 / (im1 + im2))
    circle_2.y -= mtd_y * (im2 / (im1 + im2))

    if isinstance(circle_1, Player) and isinstance(circle_2, Player):
        circle_1.v = player_player_restitution * np.sqrt(circle_1.v_x ** 2 + circle_1.v_y ** 2)
        circle_2.v = player_player_restitution * np.sqrt(circle_2.v_x ** 2 + circle_2.v_y ** 2)
    if isinstance(circle_1, Player) and isinstance(circle_2, Ball):
        circle_1.v = np.sqrt(circle_1.v_x ** 2 + circle_1.v_y ** 2)
        if circle_1.shot_request:
            circle_2.v = np.sqrt(circle_2.v_x ** 2 + circle_2.v_y ** 2)
            circle_2.v = circle_1.shot_power * circle_1.mass / (circle_1.mass + circle_2.mass) * (1 + ball_restitution)
        else:
            circle_2.v = ball_restitution_under_player_control * np.sqrt(circle_2.v_x ** 2 + circle_2.v_y ** 2)
    if isinstance(circle_1, Player) and isinstance(circle_2, Post):
        circle_1.v = player_post_restitution * np.sqrt(circle_1.v_x ** 2 + circle_1.v_y ** 2)
        circle_2.v = 0
    if isinstance(circle_1, Ball) and isinstance(circle_2, Post):
        circle_1.v = np.sqrt(circle_1.v_x ** 2 + circle_1.v_y ** 2)
        circle_2.v = 0

    circle_1.alpha = np.arctan2(circle_1.v_y, circle_1.v_x)
    circle_2.alpha = np.arctan2(circle_2.v_y, circle_2.v_x)

    for circle in [circle_1, circle_2]:
        if isinstance(circle, Player) or isinstance(circle, Ball):
            circle.clip_velocity()
            circle.snelius()

    return circle_1, circle_2


def team_properties():
    properties = dict()
    player_names = ["Венко", "Филипче", "Пашето"]
    properties['team_name'] = "Пелистер"
    properties['player_names'] = player_names
    properties['image_name'] = 'pelister.png'  # use image resolution 153x153
    return properties


team_1_properties = team_properties()
team_2_properties = team_properties()
team_1 = [Player(team_1_properties['player_names'][0], weights[0], radiuses[0], accelerations[0], speeds[0],
                 shot_powers[0]),
          Player(team_1_properties['player_names'][1], weights[1], radiuses[1], accelerations[1], speeds[1],
                 shot_powers[1]),
          Player(team_1_properties['player_names'][2], weights[2], radiuses[2], accelerations[2], speeds[2],
                 shot_powers[2])]
team_2 = [Player(team_2_properties['player_names'][0], weights[0], radiuses[0], accelerations[0], speeds[0],
                 shot_powers[0]),
          Player(team_2_properties['player_names'][1], weights[1], radiuses[1], accelerations[1], speeds[1],
                 shot_powers[1]),
          Player(team_2_properties['player_names'][2], weights[2], radiuses[2], accelerations[2], speeds[2],
                 shot_powers[2])]

# MAIN PROPERTIES
the_ball = Ball(420, 250, 15, 0.5)
the_posts = [Post(post_screen_left, post_screen_top, post_radius, post_mass),
             Post(post_screen_left, post_screen_bottom, post_radius, post_mass),
             Post(post_screen_right, post_screen_top, post_radius, post_mass),
             Post(post_screen_right, post_screen_bottom, post_radius, post_mass)]

if platform.system() == "Windows":
    red_logo = pygame.image.load(os.getcwd() + '\\Team_name\\' + team_1_properties['image_name'])
    blue_logo = pygame.image.load(os.getcwd() + '\\Team_name\\' + team_2_properties['image_name'])
else:
    # red_logo = pygame.image.load(os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/Team_name/' + team_1_properties['image_name'])
    red_logo = pygame.image.load(path.join(path.dirname(__file__), "assets/pelister.png"))
    # blue_logo = pygame.image.load(os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/Team_name/' + team_1_properties['image_name'])
    blue_logo = pygame.image.load(path.join(path.dirname(__file__), "assets/pelister.png"))
logos = {team_1_properties['team_name']: red_logo, team_2_properties['team_name']: blue_logo}


# This function gathers game information and controls each one of your three players
def getRoles(players):
    roles = dict()
    sorted_by_raduis = sorted(players, key=lambda x: x['radius'])
    roles['attacker'] = sorted_by_raduis[0]
    roles['mid'] = sorted_by_raduis[1]
    roles['defender'] = sorted_by_raduis[2]
    return roles


def get_angle_from_points(x1, y1, x2, y2):
    if x1 > x2:
        return np.arctan((y2 - y1) / (x1 - x2)) - np.pi
    elif x1 < x2:
        return np.arctan((y1 - y2) / (x1 - x2))
    else:
        return 0


def get_angle_attacker(player_x, player_y, ball_x, ball_y):
    direction = get_angle(player_x, player_y, ball_x, ball_y)
    return direction


def get_angle_goal_keeper(x1, y1, x2, y2, your_side, max_force):
    goal_y = y2
    if y2 <= 363:
        goal_y = max(323, y2)
    else:
        goal_y = min(598, y2)
    if your_side == 'right':
        goal_x = 1281
    else:
        goal_x = 85
    direction = get_angle(x1, y1, goal_x, goal_y)
    return direction, max_force


def get_angle_mid(player_x, player_y, other_player_x, other_player_y, your_side):
    if your_side == 'left' and other_player_x > 100:
        x_coor = other_player_x - 50
    elif your_side == 'right' and other_player_x < 1266:
        x_coor = other_player_x + 50
    else:
        x_coor = other_player_x
    return get_angle(player_x, player_y, x_coor, other_player_y)
    # return direction


def get_angle(x1, y1, x2, y2):
    return 2 * np.pi - np.arctan2((y1 - y2), (x2 - x1))


def get_side_attacker(player_x, player_y, ball_x, alpha, your_side):
    goal_top = 578
    goal_bottom = 343
    if your_side == 'left':
        goal_x = 1300
        return player_x < ball_x
    # else:
    #     goal_x = 50
    #     b = player_x>ball_x
    #
    # angle_top = get_angle(player_x, player_y, goal_x, goal_top)
    # angle_bot = get_angle(player_x, player_y, goal_x, goal_bottom)
    # return angle_bot<alpha<angle_top and b
    return player_x > ball_x


def get_closest_player_to_your_goal(their_team, your_side):
    if your_side == 'left':
        g = np.array([50, 460])
    else:
        g = np.array([1316, 460])
    min_dist = 20000
    i = 0
    index = 0
    for player in their_team:
        pl = their_team[player]
        p = np.array([pl['x'], pl['y']])
        dist = np.linalg.norm(g - p)
        if dist < min_dist:
            min_dist = dist
            index = player
        i += 1
    return their_team[index]


def get_closest_player_to_their_goal(their_team, your_side):
    if your_side == 'right':
        g = np.array([50, 460])
    else:
        g = np.array([1316, 460])
    min_dist = 20000
    i = 0
    index = 0
    for player in their_team:
        pl = their_team[player]
        p = np.array([pl['x'], pl['y']])
        dist = np.linalg.norm(g - p)
        if dist < min_dist:
            min_dist = dist
            index = player
        i += 1
    return their_team[index]


def get_random_angle_attacker(player_x, player_y, your_side):
    if your_side == 'left':
        goal = [1316, random.randint(343, 578)]
    else:
        goal = [50, random.randint(343, 578)]
    return get_angle(player_x, player_y, goal[0], goal[1])


def min_distance_from_opponents(player, their_team):
    g = np.array([player['x'], player['y']])
    min_dist = 20000
    i = 0
    for player in their_team:
        pl = their_team[player]
        p = np.array([pl['x'], pl['y']])
        dist = np.linalg.norm(g - p)
        if dist < min_dist:
            min_dist = dist
        i += 1
    return min_dist


def attacker_not_directed_at_side(player, your_side, alpha, ball):
    if your_side == 'left':
        goal_x = 1300
        return player['x'] < ball['x']
    return player['x'] > ball['x']


def attacker_not_directed_at_goal(player, your_side, alpha):
    if your_side == 'left':
        goal_x = 1316
    else:
        goal_x = 50

    y_coordinate_attacker_goal = player['y'] - (np.tan(2 * np.pi - alpha) * (player['x'] - goal_x))

    return 343 < 768 - y_coordinate_attacker_goal < 578


def get_shot_angle_attacker(player, your_side, alpha):
    if your_side == 'left':
        goal_x = 1316
    else:
        goal_x = 50

    y_coordinate_attacker_goal = player['y'] - (np.tan(2 * np.pi - alpha) * (player['x'] - goal_x))
    y_coordinate_attacker_goal = 768 - y_coordinate_attacker_goal
    if 343 < y_coordinate_attacker_goal < 578:
        if abs(y_coordinate_attacker_goal - 343) < abs(y_coordinate_attacker_goal - 578):
            return get_angle(player['x'], player['y'], goal_x, 353)
    return get_angle(player['x'], player['y'], goal_x, 568)


def get_angle_mid_between(player, ball, their_team, your_side):
    if your_side == 'left':
        g = np.array([50, 460])
    else:
        g = np.array([1316, 460])

    dict_by_distances = {k: np.linalg.norm(g - np.array([their_team[k]['x'], their_team[k]['y']])) for k in
                         their_team.keys()}

    sorted_dict = sorted(dict_by_distances.items(), reverse=True)

    p1 = their_team[sorted_dict[0][0]]['x'], their_team[sorted_dict[0][0]]['y']
    p2 = their_team[sorted_dict[1][0]]['x'], their_team[sorted_dict[1][0]]['y']

    x_avg = (p1[0] + p2[0]) / 2
    y_avg = (p1[1] + p2[1]) / 2

    if your_side == 'left' and player['x'] > 450:
        x_avg -= 100
    elif your_side == 'right' and player['x'] < 916:
        x_avg += 100

    return get_angle(player['x'], player['y'], x_avg, y_avg)


def randomize_initial_positions():
    displacement = random.randint(0, 10)
    player_1 = [int((center[0] - playground[0]) / 2) + playground[0] - displacement, post_screen_top]
    player_2 = [int((center[0] - playground[0]) / 2) + playground[0] - displacement, center[1]]
    player_3 = [int((center[0] - playground[0]) / 2) + playground[0] - displacement,
                post_screen_bottom]
    player_4 = [player_1[0] + half_playground_rect[2] + displacement, post_screen_top]
    player_5 = [player_2[0] + half_playground_rect[2] + displacement, center[1]]
    player_6 = [player_3[0] + half_playground_rect[2] + displacement,
                post_screen_bottom]

    initial_left = [player_1, player_2, player_3]
    initial_right = [player_4, player_5, player_6]
    return initial_left, initial_right


def decision(our_team, their_team, ball, your_side, half, time_left, our_score, their_score):
    manager_decision = [dict(), dict(), dict()]
    players = our_team
    opponents = their_team

    # pass?, shoot?
    players = getRoles(players)
    their_team = getRoles(their_team)
    i = 0
    for p in players:
        player = players[p]
        ball_x, ball_y = ball['x'], ball['y']
        player_x, player_y = player['x'], player['y']

        w = np.array([player_x, player_y])
        b = np.array([ball_x, ball_y])

        distance_from_ball = np.linalg.norm(w - b)

        shot_request = bool
        force = player['a_max'] * player['mass']
        shot_power = player['shot_power_max']

        if p == 'attacker':  # attacker
            alpha = get_angle_attacker(player_x, player_y, ball_x, ball_y)
            wow = np.array([player_x, player_y])
            goa = np.array([1316, 460]) if your_side == 'left' else np.array([50, 460])

            distance_from_goal = np.linalg.norm(wow - goa)

            manager_decision[i]['shot_request'] = True
            if distance_from_ball < 50 and not attacker_not_directed_at_side(player, your_side, alpha, ball):
                manager_decision[i]['alpha'] = alpha + 3 * np.pi / 8
                manager_decision[i]['shot_request'] = False
            else:
                if distance_from_ball < 50 and distance_from_goal < 300 and attacker_not_directed_at_goal(player,
                                                                                                          your_side,
                                                                                                          alpha):
                    manager_decision[i]['alpha'] = get_shot_angle_attacker(player, your_side, alpha)
                else:
                    manager_decision[i]['alpha'] = get_angle_attacker(player_x, player_y, ball_x, ball_y)
            manager_decision[i]['force'] = force


        elif p == 'mid':  # mid
            alpha = get_angle_attacker(player_x, player_y, ball_x, ball_y)
            manager_decision[i]['shot_request'] = True  # choose if you want to shoot
            opponent_attacker = get_closest_player_to_your_goal(their_team, your_side)
            if our_score < their_score:
                opponent_defender = get_closest_player_to_their_goal(their_team, your_side)
                manager_decision[i]['alpha'] = get_angle(player_x, player_y, opponent_defender['x'],
                                                         opponent_defender['y'])
            else:
                if (your_side == 'left' and opponent_attacker['x'] < 683 and ball_x < 683) or (
                        your_side == 'right' and opponent_attacker['x'] > 683 and ball_x > 683):
                    manager_decision[i]['alpha'] = get_angle_mid(player_x, player_y, opponent_attacker['x'],
                                                                 opponent_attacker['y'], your_side)
                    manager_decision[i]['shot_request'] = True

                else:
                    if distance_from_ball < 50 and not attacker_not_directed_at_side(player, your_side, alpha, ball):
                        manager_decision[i]['alpha'] = alpha + np.pi / 2
                        manager_decision[i]['shot_request'] = False
                    else:
                        manager_decision[i]['alpha'] = get_angle_attacker(player_x, player_y, ball_x, ball_y)
            manager_decision[i]['force'] = force

        else:  # defender
            manager_decision[i]['shot_request'] = True  # choose if you want to shoot
            manager_decision[i]['alpha'] = get_angle_goal_keeper(player_x, player_y, ball_x, ball_y, your_side, force)[
                0]
            manager_decision[i]['force'] = get_angle_goal_keeper(player_x, player_y, ball_x, ball_y, your_side, force)[
                1]
        manager_decision[i]['shot_power'] = shot_power  # use different shot power: (0, 'shot_power_max')
        i += 1
    return manager_decision


def decision_not_move(our_team, their_team, ball, your_side, half, time_left, our_score, their_score):
    manager_decision = [dict(), dict(), dict()]
    for i in range(3):
        manager_decision[i]['shot_request'] = False
        manager_decision[i]['force'] = 0
        manager_decision[i]['shot_power'] = 0
        manager_decision[i]['alpha'] = np.pi
    return manager_decision


class AiFootballEnv(gym.Env):
    """
    ## Description
    ## Action Space
    ## Observation Space
    ## Rewards
    ## Starting State
    ## Episode Truncation
    ## Arguments
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.velocity_step = 10000
        self.angle_step = 0.1
        self.angle_change = 0
        self.velocity_change = 0
        self.render_mode = render_mode
        normalized_low = -1.0
        normalized_high = 1.0

        # Define your action space
        # 'shot_request_1', 'force_1', 'shot_power_1', 'alpha_1',
        # 'shot_request_2', 'force_2', 'shot_power_2', 'alpha_2',
        # 'shot_request_3', ,'force_3', 'shot_power_3' 'alpha_3'

        # self.original_action_low = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # self.original_action_high = np.array([1, 160600, 13000, 2 * np.pi, 1, 160600, 14000,
        #                                       2 * np.pi, 1, 160600, 10000, 2 * np.pi])
        # self.action_space = spaces.Box(low=normalized_low, high=normalized_high, shape=(12,), dtype=np.float32)
        # DISCRETE ACTION SPACE
        self.max_force = 160600
        self.max_shot_powers = [13000, 14000, 10000]
        self.action_space = spaces.MultiDiscrete([1, 6, 6, 32] * 3)

        # Define your state space
        self.original_players_low = np.array([50, 203, 0, 0] * 6)
        self.original_players_high = np.array([1316, 718, 2 * np.pi, 700] * 6)
        self.original_ball_low = np.array([50, 203, 0, 0])
        self.original_ball_high = np.array([1316, 718, 2 * np.pi, 850])
        self.observation_space = spaces.Dict({
            'players': spaces.Box(low=normalized_low, high=normalized_high, shape=(24,), dtype=np.float32),
            'ball': spaces.Box(low=normalized_low, high=normalized_high, shape=(4,), dtype=np.float32),
            'time_left': spaces.Discrete(46),
            'our_score': spaces.Discrete(100),
            'their_score': spaces.Discrete(100)
        })

    def _get_obs(self):
        return self.observation

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.iteration = 1600
        self.done = False
        self.screen = None
        self.team_1 = [
            Player(team_1_properties['player_names'][0], weights[0], radiuses[0], accelerations[0], speeds[0],
                   shot_powers[0]),
            Player(team_1_properties['player_names'][1], weights[1], radiuses[1], accelerations[1], speeds[1],
                   shot_powers[1]),
            Player(team_1_properties['player_names'][2], weights[2], radiuses[2], accelerations[2], speeds[2],
                   shot_powers[2])]
        self.team_2 = [
            Player(team_2_properties['player_names'][0], weights[0], radiuses[0], accelerations[0], speeds[0],
                   shot_powers[0]),
            Player(team_2_properties['player_names'][1], weights[1], radiuses[1], accelerations[1], speeds[1],
                   shot_powers[1]),
            Player(team_2_properties['player_names'][2], weights[2], radiuses[2], accelerations[2], speeds[2],
                   shot_powers[2])]
        self.ball = Ball(420, 250, 15, 0.5)
        self.posts = [Post(post_screen_left, post_screen_top, post_radius, post_mass),
                      Post(post_screen_left, post_screen_bottom, post_radius, post_mass),
                      Post(post_screen_right, post_screen_top, post_radius, post_mass),
                      Post(post_screen_right, post_screen_bottom, post_radius, post_mass)]
        self.team_1_properties = team_properties()
        self.team_2_properties = team_properties()
        self.team_1_score, self.team_2_score = 0, 0

        initial_left, initial_right = randomize_initial_positions()
        for i, player in enumerate(self.team_1):
            player.reset(initial_left[i], 0)
        for i, player in enumerate(self.team_2):
            player.reset(initial_right[i], np.pi)
        self.ball.reset()

        self.circles = [self.team_1[0], self.team_1[1], self.team_1[2], self.team_2[0], self.team_2[1], self.team_2[2],
                        self.ball, self.posts[0], self.posts[1], self.posts[2], self.posts[3]]

        self.time_to_play = max(0, int(self.iteration / 35.55555))
        player_info = []
        for player in self.team_1 + self.team_2:
            player_info.append(player.x)
            player_info.append(player.y)
            player_info.append(np.mod(player.alpha, 2 * np.pi))
            player_info.append(player.v)

        p = np.array(player_info, dtype=np.float32)
        b = np.array([self.ball.x, self.ball.y, np.mod(self.ball.alpha, 2 * np.pi), self.ball.v],
                     dtype=np.float32)
        normalized_players = 2 * ((p - self.original_players_low) /
                                  (self.original_players_high - self.original_players_low)) - 1
        normalized_ball = 2 * ((b - self.original_ball_low) / (self.original_ball_high - self.original_ball_low)) - 1

        self.observation = {
            'players': normalized_players.astype(np.float32),
            'ball': normalized_ball.astype(np.float32),
            'time_left': self.time_to_play,
            'our_score': self.team_1_score,
            'their_score': self.team_2_score
        }
        return self.observation, {}

    def step(self, action):

        # action = ((action + 1) / 2) * (self.original_action_high - self.original_action_low) + self.original_action_low

        self.iteration -= 1
        if self.iteration <= 0:
            if self.ball.v <= 50:
                self.done = True
            if self.ball.x <= center[0] and np.cos(self.ball.alpha) >= 0:
                self.done = True
            if self.ball.x >= center[0] and np.cos(self.ball.alpha) <= 0:
                self.done = True

        # {'shot_request': False, 'force': 0, 'shot_power': 0, 'alpha': 3.141592653589793}
        manager_1_decision = [
            {'shot_request': True if action[0] == 1 else False, 'force': action[1] * (self.max_force / 6),
             'shot_power': action[2] * (self.max_shot_powers[0] / 6), 'alpha': action[3] * (np.pi / 16)},
            {'shot_request': True if action[4] == 1 else False, 'force': action[5] * (self.max_force / 6),
             'shot_power': action[6] * (self.max_shot_powers[1] / 6), 'alpha': action[7] * (np.pi / 16)},
            {'shot_request': True if action[8] == 1 else False, 'force': action[9] * (self.max_force / 6),
             'shot_power': action[10] * (self.max_shot_powers[2] / 6), 'alpha': action[11] * (np.pi / 16)}]

        manager_2_decision = decision_not_move(
            our_team=[self.team_2[0].data(), self.team_2[1].data(), self.team_2[2].data()],
            their_team=[self.team_1[0].data(), self.team_1[1].data(), self.team_1[2].data()],
            ball=self.ball.data(),
            your_side='right',
            half=1,
            time_left=self.time_to_play,
            our_score=self.team_2_score,
            their_score=self.team_1_score)

        manager_decision = [manager_1_decision[0], manager_1_decision[1], manager_1_decision[2],
                            manager_2_decision[0], manager_2_decision[1], manager_2_decision[2]]

        self.time_to_play = max(0, int(self.iteration / 35.55555))

        manager_decision[0]['alpha'] += self.angle_change
        manager_decision[0]['force'] += self.velocity_change

        # self.original_players_low = np.array([50, 203, 0, 0] * 6)
        # self.original_players_high = np.array([1316, 718, 2 * np.pi, 700] * 6)
        for i, player in enumerate(self.team_1 + self.team_2):
            player.move(manager_decision[i])
            player.x = np.clip(player.x, 49, 1315)
            player.y = np.clip(player.y, 202, 717)
        self.ball.move()

        goal = False
        if not goal:
            goal_team_right = post_screen_top < self.ball.y < post_screen_bottom and self.ball.x < post_screen_left
            goal_team_left = post_screen_top < self.ball.y < post_screen_bottom and self.ball.x > post_screen_right
            if goal_team_left and self.team_1_score < 100:
                self.team_1_score += 1
            if goal_team_right and self.team_2_score < 100:
                self.team_2_score += 1
            goal = goal_team_left or goal_team_right

        if goal:
            initial_left, initial_right = randomize_initial_positions()
            for i, player in enumerate(self.team_1):
                player.reset(initial_left[i], 0)
            for i, player in enumerate(self.team_2):
                player.reset(initial_right[i], np.pi)
            self.ball.reset()

        if not goal:
            for i in range(len(self.circles[:-4])):
                self.circles[i].snelius()
                for j in range(i + 1, len(self.circles)):
                    if collision(self.circles[i], self.circles[j]):
                        self.circles[i], self.circles[j] = resolve_collision(self.circles[i], self.circles[j])

        player_info = []
        for player in self.team_1 + self.team_2:
            player_info.append(player.x)
            player_info.append(player.y)
            player_info.append(np.mod(player.alpha, 2 * np.pi))
            player_info.append(player.v)

        p = np.array(player_info, dtype=np.float32)
        b = np.array([self.ball.x, self.ball.y, np.mod(self.ball.alpha, 2 * np.pi), self.ball.v],
                     dtype=np.float32)
        normalized_players = 2 * ((p - self.original_players_low) /
                                  (self.original_players_high - self.original_players_low)) - 1
        normalized_ball = 2 * ((b - self.original_ball_low) / (self.original_ball_high - self.original_ball_low)) - 1

        self.observation = {
            'players': normalized_players.astype(np.float32),
            'ball': normalized_ball.astype(np.float32),
            'time_left': self.time_to_play,
            'our_score': self.team_1_score,
            'their_score': self.team_2_score
        }

        reward = self.reward()

        if self.render_mode == "human":
            self.render()

        return self.observation, reward, self.done, False, {}

    def reward(self):
        if self.team_1_score - self.team_2_score > 10:
            return 1
        elif self.team_2_score - self.team_1_score > 10:
            return -1
        r = self.team_1_score - self.team_2_score
        score = 2 * ((r - (-10)) / (10 - (-10))) - 1
        x1, y1 = 1316, 350
        x2, y2 = 1316, 571
        x3, y3 = self.ball.x, self.ball.y
        px = x2 - x1
        py = y2 - y1
        norm = px * px + py * py
        u = ((x3 - x1) * px + (y3 - y1) * py) / float(norm)
        if u > 1:
            u = 1
        elif u < 0:
            u = 0
        x = x1 + u * px
        y = y1 + u * py
        dx = x - x3
        dy = y - y3
        dist = (dx * dx + dy * dy) ** .5
        low = 0
        high = 1280
        norm_dist = 2 * ((dist - high) / (low - high)) - 1
        norm_dist /= 20
        score += norm_dist
        return score

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode(resolution, pygame.RESIZABLE)
                pygame.display.set_caption(game_name)
                pygame.mixer.init(22050, -16, 2, 2048)
                pygame.mixer.music.load(path.join(path.dirname(__file__), "assets/football_crowd.ogg"))
                pygame.mixer.music.play(10, 14)

        pygame.draw.rect(self.screen, white, resolution_rect)
        pygame.draw.rect(self.screen, grass, ground_rect)
        pygame.draw.rect(self.screen, black, resolution_rect, 2)
        pygame.draw.rect(self.screen, black, ground_rect, 2)
        pygame.draw.rect(self.screen, white, playground_rect, 2)
        pygame.draw.rect(self.screen, white, half_playground_rect, 2)
        pygame.draw.circle(self.screen, white, center, 100, 2)
        pygame.draw.circle(self.screen, white, center, 5)

        team_left_logo = logos['Пелистер']
        team_right_logo = logos['Пелистер']

        font = pygame.font.SysFont("roboto", 50)
        img = font.render('Пелистер', True, (0, 0, 0))
        self.screen.blit(img, (10, 60))

        font = pygame.font.SysFont("roboto", 50)
        img = font.render('Пелистер', True, (0, 0, 0))
        self.screen.blit(img, (915, 60))

        team_left_color = [255, 0, 0]
        team_right_color = [0, 255, 0]

        self.screen.blit(team_left_logo, team_left_logo_position)
        self.screen.blit(team_right_logo, team_right_logo_position)

        pygame.draw.rect(self.screen, team_left_color, team_left_color_position)
        pygame.draw.rect(self.screen, team_right_color, team_right_color_position)

        for player in self.team_1:
            player.draw(self.screen, [255, 0, 0])
            font = pygame.font.SysFont("roboto", 30)
        for player in self.team_2:
            player.draw(self.screen, [0, 255, 0])
            font = pygame.font.SysFont("roboto", 30)
        for player in self.team_1 + self.team_2:
            img = font.render(player.name, True, (0, 0, 0))
            self.screen.blit(img, (player.x - 50, player.y - 50))
        self.ball.draw(self.screen)
        for post in self.posts:
            post.draw(self.screen)

        myfont = pygame.font.SysFont("monospace", 120)
        time_screen_value = self.time_to_play
        if time_screen_value < 0:
            time_screen_value = 0
        message = "{}s".format(time_screen_value)
        label = myfont.render(message, 1, (0, 0, 0))
        self.screen.blit(label, (1150, 0))

        myfont = pygame.font.SysFont("roboto", 40)

        label = myfont.render("полувреме: 1", 1, (0, 0, 0))
        self.screen.blit(label, (1175, 125))

        myfont = pygame.font.SysFont("monospace", 150)
        message = "{}:{}".format(self.team_1_score, self.team_2_score)

        label = myfont.render(message, 1, (0, 0, 0))
        self.screen.blit(label, (215 + shift, 0))

        pygame.display.flip()
        pygame.time.Clock().tick()

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
