import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2


def draw_heat_map(data, start_point, configuration_path, flag_show, i):
    x_s, y_s = [int(100*value) for value in start_point]
    data[y_s, x_s] = 4
    values = ['Up', 'Right', 'Down', 'Left', 'Origin']
    plt.figure(figsize=(10, 8))
    im = plt.imshow(data, interpolation='none', origin='lower')
    colors = [im.cmap(im.norm(color)) for color in range(5)]
    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=colors[i], label=values[i]) for i in range(len(values))]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.grid(True)
    plt.title("DQN Network")
    if flag_show:
        plt.savefig(configuration_path + "/dqn_visualization_episode_" + str(i) + ".png")
    else:
        plt.savefig(configuration_path + "/dqn_visualization.png")
    plt.close()


def draw_state(init_image, state_t_1, state_t, percent_steps):
    color = int(255*percent_steps)
    x = int(500 * state_t[0])
    y = int(500 * (1-state_t[1]))
    xp = int(500 * state_t_1[0])
    yp = int(500 * (1 - state_t_1[1]))

    cv2.circle(init_image, (x, y), 4, [0, color, 255-color], thickness=cv2.FILLED)
    cv2.line(init_image, (x, y), (xp, yp), [0, color, 255-color], thickness=10)
    return init_image

