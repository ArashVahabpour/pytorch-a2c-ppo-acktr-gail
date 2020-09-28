import gym
import torch
import numpy as np
import numpy.linalg as linalg
from utilities import *
import pickle
import random
from matplotlib import pyplot as plt
from typing import List
from numbers import Real
from tqdm.auto import tqdm

# FIXME: all of these are to be tested


def generate_start_state(loc_xy, std_var, mode="moving",
                         radius=None, cx=None, cy=None, delta_theta=None):
    """
    Generate one start state for one trajectory
    Args:
        cx, cy: the center of the circular outline
        mode:
            1. mode; moving: 
            2. mode: static: (1,1)-(1,1)–(1,1)-(1,1)-(1,1)
    """
    # If not
    noise_loc = np.random.normal(0, std_var, 2)
    #state_arr = np.zeros((num_traj, traj_len, 2, 5))
    #action_arr = np.zeros((num_traj, traj_len, 2, 1))
    new_state = np.zeros((2, 5))
    cur_loc = loc_xy + noise_loc
    new_state[:, 0] = cur_loc
    if mode == "moving":
        for i in range(4):
            speed = compute_speed_vec(
                radius, cx, cy, cur_loc[0], cur_loc[1], delta_theta)
            #print(state[:,1:].shape, next_loc.reshape(-1,1).shape)
            #new_state = np.concatenate([state[:,1:], next_loc.reshape(-1,1)],axis=1)
            new_state[:, i+1] = cur_loc + speed
            cur_loc += speed
    elif mode == "static":
        for i in range(4):
            new_state[:, i+1] = cur_loc.copy()
    return new_state


def generate_pt_in_circle(radius, cx, cy, num_traj):
    pt_arr = np.zeros((num_traj, 2))
    random_angle = np.random.uniform(-np.pi, np.pi, num_traj)
    pt_arr[:, 0] = cx + radius * np.cos(random_angle)
    pt_arr[:, 1] = cy + radius * np.sin(random_angle)
    return pt_arr


def generate_traj(radius, cx, cy, num_traj=50, traj_len=1000,
                  std_var=1, mode="circle", motion_mode="moving"):
    """
    num_traj: traj for this mode
    traj_len: length of each traj (number of steps)
    state: 2*5 tuples for (x,y) in 5 steps
    """
    delta_theta = 2*np.pi/traj_len
    # seed the pseudorandom number generator
    start_loc = np.zeros((num_traj, 2, 5))
    if mode == "circle":
        start_loc_pt = generate_pt_in_circle(radius, cx, cy, num_traj)
    else:
        start_loc_pt = np.random.normal(0, std_var, (num_traj, 2))

    state_arr = np.zeros((num_traj, traj_len, 2, 5))
    action_arr = np.zeros((num_traj, traj_len, 2, 1))
    for j in range(num_traj):
        start_loc[j, :, :] = generate_start_state(
            start_loc_pt[j], std_var, radius, cx, cy, delta_theta, motion_mode)
        state = start_loc[j, :, :]
        for i in range(traj_len):
            new_state, action = _step(radius, cx, cy, state, delta_theta)
            state_arr[j, i, :, :] = new_state
            action_arr[j, i, :, :] = action.reshape(2, 1)
            state = new_state
    return state_arr, action_arr


def compute_speed_vec(radius, cx, cy, loc_x, loc_y, delta_theta):
    dis_vect = np.array([loc_x, loc_y]) - np.array([cx, cy])
    circ_vect = np.array([dis_vect[1],  -dis_vect[0]])
    length = linalg.norm(dis_vect)
    speed = (circ_vect/length)*radius*delta_theta - \
        (dis_vect/length)*(length - radius)
    #print("dis_vect, circ_vect, speed", dis_vect, circ_vect, speed)
    return speed


def clip_speed(speed: torch.Tensor, max_ac_mag: float):
    if linalg.norm(speed) > max_ac_mag:
        speed = speed / linalg.norm(speed) * max_ac_mag
    return speed


def compute_speed_vector(loc_x, loc_y, radius, max_ac_mag, cx, cy, delta_theta=2*np.pi/100):
    dis_vect = np.array([loc_x, loc_y]) - np.array([cx, cy])
    circ_vect = np.array([dis_vect[1],  -dis_vect[0]])
    length = linalg.norm(dis_vect)
    speed = (circ_vect/length)*radius*delta_theta - \
        (dis_vect/length)*(length - radius)
    if linalg.norm(speed) > max_ac_mag:
        speed = speed / linalg.norm(speed) * max_ac_mag
    return speed


def compute_speed_vector_new(loc_x, loc_y, radius, max_ac_mag, cx, cy, delta_theta=2*np.pi/100):
    start = np.array([loc_x, loc_y])
    center = np.array([cx, cy])
    rot_mat_T = np.array([
        [np.cos(delta_theta), -np.sin(delta_theta)],
        [np.sin(delta_theta), np.cos(delta_theta)]
    ]).T
    radial_dist = (start - center).dot(rot_mat_T)
    circ_dest = radial_dist + center
    circ_speed = circ_dest - start
    length = linalg.norm(radial_dist)
    speed = circ_speed - (radial_dist/length)*(length - radius)
    if linalg.norm(speed) > max_ac_mag:
        speed = speed / linalg.norm(speed) * max_ac_mag
    return speed


def _step(radius, cx, cy, state, delta_theta):
    # delta_theta = np.pi/1000 ## finish one circle in 200 steps
    cur_loc = state[:, -1]
    speed = compute_speed_vec(
        radius, cx, cy, cur_loc[0], cur_loc[1], delta_theta)
    next_loc = cur_loc + speed
    #print(state[:,1:].shape, next_loc.reshape(-1,1).shape)
    new_state = np.concatenate([state[:, 1:], next_loc.reshape(-1, 1)], axis=1)
    #print(speed.shape, new_state.shape)
    return new_state, speed


def test():
    plt.figure()
    state = np.array(np.repeat(np.arange(5), 2).reshape(2, -1))
    #state = np.zeros(2,5)
    radius = 10
    cx = 0
    cy = 0
    delta_theta = np.pi/1000
    for i in range(2000):
        new_state, action = _step(radius, cx, cy, state, delta_theta)
        plt.plot(new_state[0, -1], new_state[1, -1], "*")
        state = new_state
        #print(i, new_state)
    plt.show()


def visualize_trajs(state_arr, action_arr, traj_len, sel_indx=0):
    # one trajectory visualization
    plt.figure()
    for i in range(traj_len):
        plt.plot(state_arr[sel_indx, i, 0, -1],
                 state_arr[sel_indx, i, 1, -1], "*")
    # Action visualization
    plt.figure()
    for i in range(traj_len):
        plt.plot(action_arr[sel_indx, i, 0, -1],
                 action_arr[sel_indx, i, 1, -1], "*")


# pylint: disable=not-callable
# radii = np.linspace(20, -10, 100)  # any radius between 10, -5
# radii = [20, 10, -10]
radii = [20]
state_len = 5
num_traj = 500  # number of trajectories


def generate_one_traj_env(traj_len: int, state_len: int, radius: Real,
                          actor, noise: bool, render: bool = False):
    """Create a new environment and generate a trajectory
    If the trajectory is prematurely ended before the length `traj_len`,
    start over again to make sure the trajectory has length `traj_len`
    Returns:
        states (List[np.array]): The list of states in the trajectory
        actions (List[np.array]): The list of actions in the trajectory
        length (int): The length of the trajectory
    """
    assert traj_len >= 1000, "WARNING: DO NOT CHANGE THIS OR LOWER VALUES CAN CAUSE ISSUES IN GAIL RUN"
    length = traj_len
    env = gym.make("Circles-v0", radii=[radius],
                   state_len=state_len, no_render=False)
    max_ac_mag = env.max_ac_mag  # max action magnitude
    states, actions = [], []
    done = False
    step = 0
    observation = env.reset()
    while True:
        states.append(observation)
        action = actor(observation, radius, max_ac_mag)
        actions.append(action)
        if render:
            env.render()
        observation, reward, done, info = env.step(action, noise)
        step += 1
        if step >= traj_len:
            break
        elif done:
            # start over a new trajectory hoping that this time it completes
            observation = env.reset()
            step = 0
            states[:] = []
            actions[:] = []
            print('warning: an incomplete trajectory occured.')
    env.close()
    return states, actions, length


def generate_traj_env_dataset(num_traj: int, traj_len: int, state_len: int, radii: List[Real],
                              save_path="gail_experts/circle/trajs_circles.pt",
                              noise=True, render=False):
    # traj_len = 1000  # length of each trajectory --- WARNING: DO NOT CHANGE THIS OR LOWER VALUES CAN CAUSE ISSUES IN GAIL RUN
    expert_data = {'states': [],
                   'actions': [],
                   'radii': [],
                   'lengths': []}

    def circular_actor(states, radius, max_ac_mag):
        x, y = states[-2:]
        action = compute_speed_vector_new(
            x, y, abs(radius), max_ac_mag, 0, radius)
        return action

    for traj_id in tqdm(range(num_traj)):
        radius = np.random.choice(radii)
        states, actions, length = generate_one_traj_env(
            traj_len, state_len, radius, circular_actor, noise, render
        )
        expert_data['states'].append(torch.FloatTensor(np.array(states)))
        expert_data['actions'].append(torch.FloatTensor(np.array(actions)))
        expert_data['radii'].append(radius)
        expert_data['lengths'].append(length)

    expert_data['states'] = torch.stack(expert_data['states'])
    expert_data['actions'] = torch.stack(expert_data['actions'])
    expert_data['radii'] = torch.tensor(expert_data['radii'])
    expert_data['lengths'] = torch.tensor(expert_data['lengths'])

    if save_path is not None:
        create_dir(os.path.dirname(save_path))
        torch.save(expert_data, save_path)
    return expert_data


# specify filename by argument? multiple
# def generate_traj_env(num_traj, state_len, radii, save_dir="gail_experts/circle"):
#     traj_len = 1000  # length of each trajectory --- WARNING: DO NOT CHANGE THIS OR LOWER VALUES CAN CAUSE ISSUES IN GAIL RUN
#     env = gym.make("Circles-v0", radii=radii, state_len=5, no_render=False)

#     expert_data = {'states': [],
#                    'actions': [],
#                    'radii': [],
#                    'lengths': torch.tensor([traj_len] * num_traj, dtype=torch.int32)}

#     max_ac_mag = env.max_ac_mag  # max action magnitude

#     for traj_id in range(num_traj):
#         print('traj #{}'.format(traj_id + 1))
#         done = False

#         observation = env.reset()
#         step = 0
#         states = []
#         actions = []
#         while step < traj_len:
#             #         env.render()  # uncomment for visualisation purposes
#             radius = env.radius
#             x, y = env.state[-2:]
#             action = compute_speed_vector(x, y, abs(radius), max_ac_mag, 0, radius) + np.random.randn(2) * max_ac_mag * 0.1

#             states.append(observation)
#             observation, reward, done, info = env.step(action)
#             actions.append(action)

#             step += 1

#             if done:
#                 # start over a new trajectory hoping that this time it completes
#                 observation = env.reset()
#                 step = 0
#                 states = []
#                 actions = []
#                 print('warning: an incomplete trajectory occured.')

#         expert_data['states'].append(torch.FloatTensor(np.array(states)))
#         expert_data['actions'].append(torch.FloatTensor(np.array(actions)))
#         expert_data['radii'].append(radius)
#     env.close()

#     expert_data['states'] = torch.stack(expert_data['states'])
#     expert_data['actions'] = torch.stack(expert_data['actions'])
#     expert_data['radii'] = torch.tensor(expert_data['radii'])
#     create_dir(save_dir)
#     torch.save(expert_data, 'trajs_circles.pt')
#     print('expert data saved successfully.')


def flat_to_nested(states, state_len=None, dim_state=None):
    """Restructure the flattened states to nested states
    [num_traj] x num_batch x dim_flat_state -> [num_traj] x num_batch x history_length x dim_state 
    """
    assert (state_len is not None) or (
        dim_state is not None), "At least one of the `history_length` and `dim_state` needs to be specified"
    if (state_len is not None) and (dim_state is not None):
        assert state_len * \
            dim_state == states.shape[-1], f"Dimensions don't match: history_length ({state_len}) x dim_state ({dim_state}) and states.shape[-1] ({states.shape[-1]})"
    if state_len is not None:
        return states.reshape(*states.shape[:-1], state_len, -1)
    else:
        return states.reshape(*states.shape[:-1], -1, dim_state)


def nested_to_flat(states):
    """Restructure the nested states to flattened states
    [num_traj] x num_batch x history_length x dim_state -> [num_traj] x num_batch x dim_flat_state 
    """
    return states.reshape(*states.shape[:-2], -1)


if __name__ == "__main__":
    generate_traj_env_dataset(
        500, 1000, 5, [-10, 10, 20], save_path="/tmp/trajs_circles.pt", noise=True, render=False)
