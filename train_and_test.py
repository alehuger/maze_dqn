import time
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

from agent import Agent
from util import draw_heat_map, draw_state


def test_config(environment, config_path, flag_display=True, flag_plot=True):

    os.makedirs(config_path)

    state = environment.init_state
    init_state = state

    agent = Agent(init_state, config_path)

    # Determine the time at which training will stop, i.e. in 10 minutes (600 seconds) time
    start_time = time.time()
    end_time = start_time + 600

    initialization_steps = []
    distance_per_episode = []
    epsilon_per_episode = []

    map_show = environment.show(state)
    cv2.imwrite(config_path + "/map.png", map_show)

    # Train the agent, until the time is up
    while time.time() < end_time:
        if flag_display:
            environment.show(state)
        # If the action is to start a new episode, then reset the state
        if agent.has_finished_episode():
            print("There is {} seconds left!".format(int(end_time-time.time())))
            state = environment.init_state

            initialization_steps.append(agent.initialisation_steps)
            distance_per_episode.append(agent.final_distance)
            epsilon_per_episode.append(agent.epsilon_end)

        # Get the state and action from the agent
        action = agent.get_next_action(state)
        # Get the next state and the distance to the goal
        next_state, distance_to_goal = environment.step(state, action)
        # Return this to the agent
        agent.set_next_state_and_distance(next_state, distance_to_goal)
        # Set what the new state is
        state = next_state

    if flag_plot:
        x_axis = range(agent.nb_episodes)
        plt.plot(x_axis, initialization_steps)
        plt.title('initialization_steps')
        plt.savefig(config_path + "/initialization_steps.png")
        plt.close()

        plt.plot(x_axis, distance_per_episode)
        plt.title('distance_per_episode')
        plt.savefig(config_path + "/distance_per_episode.png")
        plt.close()

        plt.plot(x_axis, epsilon_per_episode)
        plt.title('epsilon_per_episode')
        plt.savefig(config_path + "/epsilon_per_episode.png")
        plt.close()

        np.savez(config_path + "/buffer.npz",
                 buffer=np.array(agent.replay_buffer.transition_container),
                 probabilities=np.array(agent.transition_probabilities))

        heat_map_data = agent.dqn.visualize_dqn()
        draw_heat_map(heat_map_data, init_state, config_path, False, -1)

    # Test the agent for 100 steps, using its greedy policy
    state = environment.init_state
    has_reached_goal = False
    step_num, distance_to_goal = None, None
    image = environment.show(state)
    for step_num in range(100):
        action = agent.get_greedy_action(state)
        next_state, distance_to_goal = environment.step(state, action)
        image = draw_state(image, state, next_state, step_num / 100)
        state = next_state

        # The agent must achieve a maximum distance of 0.03 for use to consider it "reaching the goal"
        if distance_to_goal < 0.03:
            has_reached_goal = True
            break

    # Print out the result
    if has_reached_goal:
        print('Reached goal in ' + str(step_num) + ' steps.')
    else:
        print('Did not reach goal. Final distance = ' + str(distance_to_goal))

    cv2.imwrite(config_path + "/final_step_map.png", image)
