import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
from utils import to_numpy


def print_graph_with_same_agent_states(agent, args=None, run_directory=str, transitions=True, width=5.512, height=4, blue_ordered=False):

    if args.format=='pdf':
        matplotlib.use('pdf')
        plt.rc('font', family='serif', serif='Times')
        plt.rc('text', usetex=True)

    action1 = torch.tensor([0]).to(agent.device)
    action2 = torch.tensor([1]).to(agent.device)
    action3 = torch.tensor([2]).to(agent.device)
    action4 = torch.tensor([3]).to(agent.device)

    if agent.onehot:
        action1 = F.one_hot(action1.long(), num_classes=4)[0]
        action2 = F.one_hot(action2.long(), num_classes=4)[0]
        action3 = F.one_hot(action3.long(), num_classes=4)[0]
        action4 = F.one_hot(action4.long(), num_classes=4)[0]

    dictionary_with_agent_states = agent.states_same_agent
    env1 = dictionary_with_agent_states['1']
    env2 = dictionary_with_agent_states['2']
    env3 = dictionary_with_agent_states['3']
    env4 = dictionary_with_agent_states['4']

    transition_function = agent.agent_forward_state_action

    encoded_batch1, featuremaps1 = agent.encoder(env1)
    encoded_batch_numpy1 = to_numpy(encoded_batch1)
    encoded_batch2, featuremaps2 = agent.encoder(env2)
    encoded_batch_numpy2 = to_numpy(encoded_batch2)
    encoded_batch3, featuremaps3 = agent.encoder(env3)
    encoded_batch_numpy3 = to_numpy(encoded_batch3)
    encoded_batch4, featuremaps4 = agent.encoder(env4)
    encoded_batch_numpy4 = to_numpy(encoded_batch4)

    fig = plt.figure()
    fig.set_size_inches(w=width, h=height)

    plt.subplot(4, 3, 1)
    plt.axis([-1.1, 1.1, -1.1, 1.1])
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    for i in range(len(encoded_batch_numpy1)):
        color = 'bo' if i==0 else 'ro'
        if blue_ordered:
            plt.plot([encoded_batch_numpy1[i][0]], [encoded_batch_numpy1[i][1]], color, zorder=3 if color=='bo' else 0)
        else:
            plt.plot([encoded_batch_numpy1[i][0]], [encoded_batch_numpy1[i][1]], color)

        if transitions:
            state_after_action_0 = to_numpy(
                transition_function(torch.cat((encoded_batch1[i], featuremaps1[i].flatten(0), action1), 0)) + agent.prediction_delta *
                encoded_batch1[i])
            state_after_action_1 = to_numpy(
                transition_function(torch.cat((encoded_batch1[i], featuremaps1[i].flatten(0), action2), 0)) + agent.prediction_delta *
                encoded_batch1[i])
            state_after_action_2 = to_numpy(
                transition_function(torch.cat((encoded_batch1[i], featuremaps1[i].flatten(0), action3), 0)) + agent.prediction_delta *
                encoded_batch1[i])
            state_after_action_3 = to_numpy(
                transition_function(torch.cat((encoded_batch1[i], featuremaps1[i].flatten(0), action4), 0)) + agent.prediction_delta *
                encoded_batch1[i])
            # Plot every action's transition as a line from each state
            plt.plot([encoded_batch_numpy1[i][0], state_after_action_0[0]],
                     [encoded_batch_numpy1[i][1], state_after_action_0[1]], 'r', alpha=0.3)
            plt.plot([encoded_batch_numpy1[i][0], state_after_action_1[0]],
                     [encoded_batch_numpy1[i][1], state_after_action_1[1]], 'g', alpha=0.3)
            plt.plot([encoded_batch_numpy1[i][0], state_after_action_2[0]],
                     [encoded_batch_numpy1[i][1], state_after_action_2[1]], 'b', alpha=0.3)
            plt.plot([encoded_batch_numpy1[i][0], state_after_action_3[0]],
                     [encoded_batch_numpy1[i][1], state_after_action_3[1]], 'y', alpha=0.3)

    plt.draw()
    plt.subplot(4, 3, 2)
    img1 = plt.imshow(to_numpy(featuremaps1[0][0]), cmap='gray')
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    plt.draw()
    plt.subplot(4, 3, 3)
    img12 = plt.imshow(to_numpy(env1[0][0]), cmap='gray')
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    plt.yticks(np.arange(0, 48, 20))

    plt.draw()

    plt.subplot(4, 3, 4)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    for i in range(len(encoded_batch_numpy2)):
        color = 'bo' if i==0 else 'ro'
        if blue_ordered:
            plt.plot([encoded_batch_numpy2[i][0]], [encoded_batch_numpy2[i][1]], color, zorder=3 if color=='bo' else 0)
        else:
            plt.plot([encoded_batch_numpy2[i][0]], [encoded_batch_numpy2[i][1]], color)
        if transitions:
            state_after_action_0 = to_numpy(
                transition_function(torch.cat((encoded_batch2[i], featuremaps2[i].flatten(0), action1), 0)) + agent.prediction_delta *
                encoded_batch2[i])
            state_after_action_1 = to_numpy(
                transition_function(torch.cat((encoded_batch2[i], featuremaps2[i].flatten(0), action2), 0)) + agent.prediction_delta *
                encoded_batch2[i])
            state_after_action_2 = to_numpy(
                transition_function(torch.cat((encoded_batch2[i], featuremaps2[i].flatten(0), action3), 0)) + agent.prediction_delta *
                encoded_batch2[i])
            state_after_action_3 = to_numpy(
                transition_function(torch.cat((encoded_batch2[i], featuremaps2[i].flatten(0), action4), 0)) + agent.prediction_delta *
                encoded_batch2[i])
            # Plot every action's transition as a line from each state
            plt.plot([encoded_batch_numpy2[i][0], state_after_action_0[0]],
                     [encoded_batch_numpy2[i][1], state_after_action_0[1]], 'r', alpha=0.3)
            plt.plot([encoded_batch_numpy2[i][0], state_after_action_1[0]],
                     [encoded_batch_numpy2[i][1], state_after_action_1[1]], 'g', alpha=0.3)
            plt.plot([encoded_batch_numpy2[i][0], state_after_action_2[0]],
                     [encoded_batch_numpy2[i][1], state_after_action_2[1]], 'b', alpha=0.3)
            plt.plot([encoded_batch_numpy2[i][0], state_after_action_3[0]],
                     [encoded_batch_numpy2[i][1], state_after_action_3[1]], 'y', alpha=0.3)

    plt.axis([-1.1, 1.1, -1.1, 1.1])
    plt.draw()
    plt.subplot(4, 3, 5)
    img1 = plt.imshow(to_numpy(featuremaps2[0][0]), cmap='gray')
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    plt.draw()
    plt.subplot(4, 3, 6)
    img12 = plt.imshow(to_numpy(env2[0][0]), cmap='gray')
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    plt.yticks(np.arange(0, 48, 20))
    plt.draw()

    plt.subplot(4, 3, 7)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    for i in range(len(encoded_batch_numpy3)):
        color = 'bo' if i==0 else 'ro'
        if blue_ordered:
            plt.plot([encoded_batch_numpy3[i][0]], [encoded_batch_numpy3[i][1]], color, zorder=3 if color=='bo' else 0)
        else:
            plt.plot([encoded_batch_numpy3[i][0]], [encoded_batch_numpy3[i][1]], color)
        if transitions:
            state_after_action_0 = to_numpy(
                transition_function(torch.cat((encoded_batch3[i], featuremaps3[i].flatten(0), action1), 0)) + agent.prediction_delta *
                encoded_batch3[i])
            state_after_action_1 = to_numpy(
                transition_function(torch.cat((encoded_batch3[i], featuremaps3[i].flatten(0), action2), 0)) + agent.prediction_delta *
                encoded_batch3[i])
            state_after_action_2 = to_numpy(
                transition_function(torch.cat((encoded_batch3[i], featuremaps3[i].flatten(0), action3), 0)) + agent.prediction_delta *
                encoded_batch3[i])
            state_after_action_3 = to_numpy(
                transition_function(torch.cat((encoded_batch3[i], featuremaps3[i].flatten(0), action4), 0)) + agent.prediction_delta *
                encoded_batch3[i])
            # Plot every action's transition as a line from each state
            plt.plot([encoded_batch_numpy3[i][0], state_after_action_0[0]],
                     [encoded_batch_numpy3[i][1], state_after_action_0[1]], 'r', alpha=0.3)
            plt.plot([encoded_batch_numpy3[i][0], state_after_action_1[0]],
                     [encoded_batch_numpy3[i][1], state_after_action_1[1]], 'g', alpha=0.3)
            plt.plot([encoded_batch_numpy3[i][0], state_after_action_2[0]],
                     [encoded_batch_numpy3[i][1], state_after_action_2[1]], 'b', alpha=0.3)
            plt.plot([encoded_batch_numpy3[i][0], state_after_action_3[0]],
                     [encoded_batch_numpy3[i][1], state_after_action_3[1]], 'y', alpha=0.3)

    plt.axis([-1.1, 1.1, -1.1, 1.1])
    plt.draw()
    plt.subplot(4, 3, 8)
    img1 = plt.imshow(to_numpy(featuremaps3[0][0]), cmap='gray')
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    plt.draw()
    plt.subplot(4, 3, 9)
    img12 = plt.imshow(to_numpy(env3[0][0]), cmap='gray')
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    plt.yticks(np.arange(0, 48, 20))
    plt.draw()

    plt.subplot(4, 3, 10)
    for i in range(len(encoded_batch_numpy4)):
        color = 'bo' if i==0 else 'ro'
        if blue_ordered:
            plt.plot([encoded_batch_numpy4[i][0]], [encoded_batch_numpy4[i][1]], color, zorder=3 if color=='bo' else 0)
        else:
            plt.plot([encoded_batch_numpy4[i][0]], [encoded_batch_numpy4[i][1]], color)
        if transitions:
            state_after_action_0 = to_numpy(
                transition_function(torch.cat((encoded_batch4[i], featuremaps4[i].flatten(0), action1), 0)) + agent.prediction_delta *
                encoded_batch4[i])
            state_after_action_1 = to_numpy(
                transition_function(torch.cat((encoded_batch4[i], featuremaps4[i].flatten(0), action2), 0)) + agent.prediction_delta *
                encoded_batch4[i])
            state_after_action_2 = to_numpy(
                transition_function(torch.cat((encoded_batch4[i], featuremaps4[i].flatten(0), action3), 0)) + agent.prediction_delta *
                encoded_batch4[i])
            state_after_action_3 = to_numpy(
                transition_function(torch.cat((encoded_batch4[i], featuremaps4[i].flatten(0), action4), 0)) + agent.prediction_delta *
                encoded_batch4[i])
            # Plot every action's transition as a line from each state
            plt.plot([encoded_batch_numpy4[i][0], state_after_action_0[0]],
                     [encoded_batch_numpy4[i][1], state_after_action_0[1]], 'r', alpha=0.3)
            plt.plot([encoded_batch_numpy4[i][0], state_after_action_1[0]],
                     [encoded_batch_numpy4[i][1], state_after_action_1[1]], 'g', alpha=0.3)
            plt.plot([encoded_batch_numpy4[i][0], state_after_action_2[0]],
                     [encoded_batch_numpy4[i][1], state_after_action_2[1]], 'b', alpha=0.3)
            plt.plot([encoded_batch_numpy4[i][0], state_after_action_3[0]],
                     [encoded_batch_numpy4[i][1], state_after_action_3[1]], 'y', alpha=0.3)

    plt.axis([-1.1, 1.1, -1.1, 1.1])
    plt.draw()
    plt.subplot(4, 3, 11)
    img1 = plt.imshow(to_numpy(featuremaps4[0][0]), cmap='gray')
    plt.draw()
    plt.subplot(4, 3, 12)
    img12 = plt.imshow(to_numpy(env4[0][0]), cmap='gray')
    plt.xticks(np.arange(0, 48, 20))
    plt.yticks(np.arange(0, 48, 20))
    plt.draw()

    if args.showplot:
        plt.show()
    if args.format=='png':
        plt.savefig(str(run_directory) + '/same_agent_states' + str(agent.iterations) + '_iterations.png', bbox_inches='tight')
    elif args.format=='pdf':
        plt.savefig(str(run_directory) + '/same_agent_states' + str(agent.iterations) + '_iterations.pdf', bbox_inches='tight')
    plt.close()


def print_featuremaps_halfagent_halfball(agent, args=None, run_directory=str, width_inches=5.5107/2,
                                        h_inches=4, extra_name='', markersize=3):

    if args.format=='pdf':
        matplotlib.use('pdf')
        plt.rc('font', family='serif', serif='Times')
        plt.rc('text', usetex=True)

    state1, state2, state3 = agent.env.observe_three_agents()
    state4, state5, state6 = agent.env.observe_three_balls()

    state1 = torch.from_numpy(state1[0]).unsqueeze(0).unsqueeze(0).float().to(agent.device)
    state2 = torch.from_numpy(state2[0]).unsqueeze(0).unsqueeze(0).float().to(agent.device)
    state3 = torch.from_numpy(state3[0]).unsqueeze(0).unsqueeze(0).float().to(agent.device)
    state4 = torch.from_numpy(state4[0]).unsqueeze(0).unsqueeze(0).float().to(agent.device)
    state5 = torch.from_numpy(state5[0]).unsqueeze(0).unsqueeze(0).float().to(agent.device)
    state6 = torch.from_numpy(state6[0]).unsqueeze(0).unsqueeze(0).float().to(agent.device)

    latent1, features1 = agent.encoder(state1)
    latent2, features2 = agent.encoder(state2)
    latent3, features3 = agent.encoder(state3)
    latent4, features4 = agent.encoder(state4)
    latent5, features5 = agent.encoder(state5)
    latent6, features6 = agent.encoder(state6)

    features1_numpy = to_numpy(features1)
    features2_numpy = to_numpy(features2)
    features3_numpy = to_numpy(features3)
    features4_numpy = to_numpy(features4)
    features5_numpy = to_numpy(features5)
    features6_numpy = to_numpy(features6)

    latent1_numpy = to_numpy(latent1)
    latent2_numpy = to_numpy(latent2)
    latent3_numpy = to_numpy(latent3)
    latent4_numpy = to_numpy(latent4)
    latent5_numpy = to_numpy(latent5)
    latent6_numpy = to_numpy(latent6)

    fig = plt.figure()

    plt.subplot(6, 3, 1)
    if agent.agent_dim == 2:
        plt.plot(latent1_numpy[0][0], latent1_numpy[0][1], 'ro', markersize=markersize)
        plt.axis([-1.1, 1.1, -1.1, 1.1])
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
    elif agent.agent_dim == 1:
        plt.axis([-1.1, 1.1, -1.1, 1.1])
        y = 0  # Make all y values the same
        plt.plot(latent1_numpy[0][0], y, '|', ms=40)  # Plot a line at each location specified in a
        ax = plt.gca()
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)

    plt.subplot(6, 3, 2)
    plt.imshow(np.flipud(features1_numpy[0][0]), cmap='gray')
    # plt.axis([0, agent.ball_grid, 0, agent.ball_grid,])
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)

    plt.subplot(6, 3, 3)
    plt.imshow(to_numpy(state1[0][0]), cmap='gray')
    plt.axis([0, 50, 0 , 50])
    plt.yticks(np.arange(0, 51, 50))
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)

    plt.subplot(6, 3, 4)
    if agent.agent_dim == 2:
        plt.plot(latent2_numpy[0][0], latent2_numpy[0][1], 'ro', markersize=markersize)
        plt.axis([-1.1, 1.1, -1.1, 1.1])
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
    elif agent.agent_dim == 1:
        plt.axis([-1.1, 1.1, -1.1, 1.1])
        y = 0  # Make all y values the same
        plt.plot(latent2_numpy[0][0], y, '|', ms=40)  # Plot a line at each location specified in a
        ax = plt.gca()
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)

    plt.subplot(6, 3, 5)
    plt.imshow(np.flipud(features2_numpy[0][0]), cmap='gray')
    # plt.axis([0, agent.ball_grid, 0, agent.ball_grid])
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)

    plt.subplot(6, 3, 6)
    plt.imshow(to_numpy(state2[0][0]), cmap='gray')
    plt.axis([0, 50, 0 , 50])
    plt.yticks(np.arange(0, 51, 50))
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)

    plt.subplot(6, 3, 7)
    if agent.agent_dim == 2:
        plt.plot(latent3_numpy[0][0], latent3_numpy[0][1], 'ro', markersize=markersize)
        plt.axis([-1.1, 1.1, -1.1, 1.1])
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
    elif agent.agent_dim == 1:
        plt.axis([-1.1, 1.1, -1.1, 1.1])
        y = 0  # Make all y values the same
        plt.plot(latent3_numpy[0][0], y, '|', ms=40)  # Plot a line at each location specified in a
        ax = plt.gca()
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)

    plt.subplot(6, 3, 8)
    plt.imshow(np.flipud(features3_numpy[0][0]), cmap='gray')
    # plt.axis([0, agent.ball_grid, 0, agent.ball_grid])
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)

    plt.subplot(6, 3, 9)
    plt.imshow(to_numpy(state3[0][0]), cmap='gray')
    plt.axis([0, 50, 0 , 50])
    plt.yticks(np.arange(0, 51, 50))
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)

    plt.subplot(6, 3, 10)
    if agent.agent_dim == 2:
        plt.plot(latent4_numpy[0][0], latent4_numpy[0][1], 'ro', markersize=markersize)
        plt.axis([-1.1, 1.1, -1.1, 1.1])
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
    elif agent.agent_dim == 1:
        plt.axis([-1.1, 1.1, -1.1, 1.1])
        y = 0  # Make all y values the same
        plt.plot(latent4_numpy[0][0], y, '|', ms=40)  # Plot a line at each location specified in a
        ax = plt.gca()
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)

    plt.subplot(6, 3, 11)
    plt.imshow(np.flipud(features4_numpy[0][0]), cmap='gray')
    # plt.axis([0, agent.ball_grid, 0, agent.ball_grid])
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)

    plt.subplot(6, 3, 12)
    plt.imshow(to_numpy(state4[0][0]), cmap='gray')
    plt.axis([0, 50, 0 , 50])
    plt.yticks(np.arange(0, 51, 50))
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)

    plt.subplot(6, 3, 13)
    if agent.agent_dim == 2:
        plt.plot(latent5_numpy[0][0], latent5_numpy[0][1], 'ro', markersize=markersize)
        plt.axis([-1.1, 1.1, -1.1, 1.1])
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
    elif agent.agent_dim == 1:
        plt.axis([-1.1, 1.1, -1.1, 1.1])
        y = 0  # Make all y values the same
        plt.plot(latent5_numpy[0][0], y, '|', ms=40)  # Plot a line at each location specified in a
        ax = plt.gca()
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)

    plt.subplot(6, 3, 14)
    plt.imshow(np.flipud(features5_numpy[0][0]), cmap='gray')
    # plt.axis([0, agent.ball_grid, 0, agent.ball_grid])
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)

    plt.subplot(6, 3, 15)
    plt.imshow(to_numpy(state5[0][0]), cmap='gray')
    plt.axis([0, 50, 0 , 50])
    plt.yticks(np.arange(0, 51, 50))
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)

    plt.subplot(6, 3, 16)
    if agent.agent_dim == 2:
        plt.plot(latent6_numpy[0][0], latent6_numpy[0][1], 'ro', markersize=markersize)
        plt.axis([-1.1, 1.1, -1.1, 1.1])
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
    elif agent.agent_dim == 1:
        plt.axis([-1.1, 1.1, -1.1, 1.1])
        y = 0  # Make all y values the same
        plt.plot(latent6_numpy[0][0], y, '|', ms=40)  # Plot a line at each location specified in a
        ax = plt.gca()
        ax.get_yaxis().set_visible(False)
        # ax.get_xaxis().set_visible(False)
    plt.subplot(6, 3, 17)
    plt.imshow(np.flipud(features6_numpy[0][0]), cmap='gray')
    # plt.axis([0, agent.ball_grid, 0, agent.ball_grid])

    plt.subplot(6, 3, 18)
    plt.imshow(to_numpy(state6[0][0]), cmap='gray')
    plt.axis([0, 50, 0 , 50])
    plt.yticks(np.arange(0, 51, 50))
    plt.xticks(np.arange(0, 51, 50))

    fig.set_size_inches(w=width_inches, h=h_inches)
    # plt.subplots_adjust(hspace=2)
    plt.tight_layout()

    plt.draw()

    if args.showplot:
        plt.show()
    else:
        if args.format == 'png':
            plt.savefig(str(run_directory)+'/different_catcher_states'+str(agent.iterations)+extra_name+'_iterations.png', bbox_inches='tight')
        elif args.format=='pdf':
            plt.savefig(str(run_directory)+'/different_catcher_states'+str(agent.iterations)+extra_name+'_iterations.pdf', bbox_inches='tight')
        plt.close()


def print_graph_with_all_states_and_transitions_3D_fourmaze(agent, device, encoder, transition_function, batch_with_every_state1,
                                                   batch_with_every_state2, batch_with_every_state3,
                                                            batch_with_every_state4, delta=True, args=None, run_directory=str):

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    node_size = 2
    edge_size = 1

    encoded_batch = encoder(batch_with_every_state1)
    encoded_batch2 = encoder(batch_with_every_state2)
    encoded_batch3 = encoder(batch_with_every_state3)
    encoded_batch4 = encoder(batch_with_every_state4)

    encoded_batch_numpy = to_numpy(encoded_batch)
    encoded_batch_numpy2 = to_numpy(encoded_batch2)
    encoded_batch_numpy3 = to_numpy(encoded_batch3)
    encoded_batch_numpy4 = to_numpy(encoded_batch4)

    action1 = torch.tensor([0]).long()
    action2 = torch.tensor([1]).long()
    action3 = torch.tensor([2]).long()
    action4 = torch.tensor([3]).long()

    if agent.onehot:
        action1 = F.one_hot(action1, num_classes=4)
        action2 = F.one_hot(action2, num_classes=4)
        action3 = F.one_hot(action3, num_classes=4)
        action4 = F.one_hot(action4, num_classes=4)

    for i in range(len(encoded_batch)):
        ax.scatter3D(encoded_batch_numpy[i][0], encoded_batch_numpy[i][1], encoded_batch_numpy[i][2], color='red', s=node_size )
        # Collect the states after each action
        state_after_action_0 = to_numpy(transition_function(torch.cat((encoded_batch[i], action1[0].to(device)), 0)) + delta * encoded_batch[i][0:2])
        state_after_action_1 = to_numpy(
            transition_function(torch.cat((encoded_batch[i], action2[0].to(device)), 0)) + delta * encoded_batch[i][0:2])
        state_after_action_2 = to_numpy(
            transition_function(torch.cat((encoded_batch[i], action3[0].to(device)), 0)) + delta * encoded_batch[i][0:2])
        state_after_action_3 = to_numpy(
            transition_function(torch.cat((encoded_batch[i], action4[0].to(device)), 0)) + delta * encoded_batch[i][0:2])
        # Plot every action's transition as a line from each state
        ax.plot3D([encoded_batch_numpy[i][0], state_after_action_0[0]],
                  [encoded_batch_numpy[i][1], state_after_action_0[1]],
                  [encoded_batch_numpy[i][2], encoded_batch_numpy[i][2]], 'red', alpha=0.3, linewidth=edge_size)
        ax.plot3D([encoded_batch_numpy[i][0], state_after_action_1[0]],
                  [encoded_batch_numpy[i][1], state_after_action_1[1]],
                  [encoded_batch_numpy[i][2], encoded_batch_numpy[i][2]], 'green', alpha=0.3, linewidth=edge_size)
        ax.plot3D([encoded_batch_numpy[i][0], state_after_action_2[0]],
                  [encoded_batch_numpy[i][1], state_after_action_2[1]],
                  [encoded_batch_numpy[i][2], encoded_batch_numpy[i][2]], 'blue', alpha=0.3, linewidth=edge_size)
        ax.plot3D([encoded_batch_numpy[i][0], state_after_action_3[0]],
                  [encoded_batch_numpy[i][1], state_after_action_3[1]],
                  [encoded_batch_numpy[i][2], encoded_batch_numpy[i][2]], 'yellow', alpha=0.3, linewidth=edge_size)

    for i in range(len(encoded_batch2)):
        ax.scatter3D(encoded_batch_numpy2[i][0], encoded_batch_numpy2[i][1], encoded_batch_numpy2[i][2], color='green', s=node_size)
        state_after_action_0 = to_numpy(
            transition_function(torch.cat((encoded_batch2[i], action1[0].to(device)), 0)) + delta *
            encoded_batch2[i][0:2])
        state_after_action_1 = to_numpy(
            transition_function(torch.cat((encoded_batch2[i], action2[0].to(device)), 0)) + delta *
            encoded_batch2[i][0:2])
        state_after_action_2 = to_numpy(
            transition_function(torch.cat((encoded_batch2[i], action3[0].to(device)), 0)) + delta *
            encoded_batch2[i][0:2])
        state_after_action_3 = to_numpy(
            transition_function(torch.cat((encoded_batch2[i], action4[0].to(device)), 0)) + delta *
            encoded_batch2[i][0:2])
        # Plot every action's transition as a line from each state
        ax.plot3D([encoded_batch_numpy2[i][0], state_after_action_0[0]], [encoded_batch_numpy2[i][1], state_after_action_0[1]], [encoded_batch_numpy2[i][2], encoded_batch_numpy2[i][2]], 'red', alpha=0.3, linewidth=edge_size)
        ax.plot3D([encoded_batch_numpy2[i][0], state_after_action_1[0]], [encoded_batch_numpy2[i][1], state_after_action_1[1]], [encoded_batch_numpy2[i][2], encoded_batch_numpy2[i][2]], 'green', alpha=0.3, linewidth=edge_size)
        ax.plot3D([encoded_batch_numpy2[i][0], state_after_action_2[0]], [encoded_batch_numpy2[i][1], state_after_action_2[1]], [encoded_batch_numpy2[i][2], encoded_batch_numpy2[i][2]], 'blue', alpha=0.3, linewidth=edge_size)
        ax.plot3D([encoded_batch_numpy2[i][0], state_after_action_3[0]], [encoded_batch_numpy2[i][1], state_after_action_3[1]], [encoded_batch_numpy2[i][2], encoded_batch_numpy2[i][2]], 'yellow', alpha=0.3, linewidth=edge_size)

    for i in range(len(encoded_batch3)):
        ax.scatter3D(encoded_batch_numpy3[i][0], encoded_batch_numpy3[i][1], encoded_batch_numpy3[i][2], color='blue', s=node_size)
        state_after_action_0 = to_numpy(
            transition_function(torch.cat((encoded_batch3[i], action1[0].to(device)), 0)) + delta *
            encoded_batch3[i][0:2])
        state_after_action_1 = to_numpy(
            transition_function(torch.cat((encoded_batch3[i], action2[0].to(device)), 0)) + delta *
            encoded_batch3[i][0:2])
        state_after_action_2 = to_numpy(
            transition_function(torch.cat((encoded_batch3[i], action3[0].to(device)), 0)) + delta *
            encoded_batch3[i][0:2])
        state_after_action_3 = to_numpy(
            transition_function(torch.cat((encoded_batch3[i], action4[0].to(device)), 0)) + delta *
            encoded_batch3[i][0:2])
        # Plot every action's transition as a line from each state

        ax.plot3D([encoded_batch_numpy3[i][0], state_after_action_0[0]], [encoded_batch_numpy3[i][1], state_after_action_0[1]], [encoded_batch_numpy3[i][2], encoded_batch_numpy3[i][2]], 'red', alpha=0.3, linewidth=edge_size)
        ax.plot3D([encoded_batch_numpy3[i][0], state_after_action_1[0]], [encoded_batch_numpy3[i][1], state_after_action_1[1]], [encoded_batch_numpy3[i][2], encoded_batch_numpy3[i][2]], 'green', alpha=0.3, linewidth=edge_size)
        ax.plot3D([encoded_batch_numpy3[i][0], state_after_action_2[0]], [encoded_batch_numpy3[i][1], state_after_action_2[1]], [encoded_batch_numpy3[i][2], encoded_batch_numpy3[i][2]], 'blue', alpha=0.3, linewidth=edge_size)
        ax.plot3D([encoded_batch_numpy3[i][0], state_after_action_3[0]], [encoded_batch_numpy3[i][1], state_after_action_3[1]], [encoded_batch_numpy3[i][2], encoded_batch_numpy3[i][2]], 'yellow', alpha=0.3, linewidth=edge_size)

    for i in range(len(encoded_batch4)):
        ax.scatter3D(encoded_batch_numpy4[i][0], encoded_batch_numpy4[i][1], encoded_batch_numpy4[i][2], color='orange', s=node_size)
        state_after_action_0 = to_numpy(
            transition_function(torch.cat((encoded_batch4[i], action1[0].to(device)), 0)) + delta *
            encoded_batch4[i][0:2])
        state_after_action_1 = to_numpy(
            transition_function(torch.cat((encoded_batch4[i], action2[0].to(device)), 0)) + delta *
            encoded_batch4[i][0:2])
        state_after_action_2 = to_numpy(
            transition_function(torch.cat((encoded_batch4[i], action3[0].to(device)), 0)) + delta *
            encoded_batch4[i][0:2])
        state_after_action_3 = to_numpy(
            transition_function(torch.cat((encoded_batch4[i], action4[0].to(device)), 0)) + delta *
            encoded_batch4[i][0:2])
        # Plot every action's transition as a line from each state

        ax.plot3D([encoded_batch_numpy4[i][0], state_after_action_0[0]], [encoded_batch_numpy4[i][1], state_after_action_0[1]], [encoded_batch_numpy4[i][2], encoded_batch_numpy4[i][2]], 'red', alpha=0.3, linewidth=edge_size)
        ax.plot3D([encoded_batch_numpy4[i][0], state_after_action_1[0]], [encoded_batch_numpy4[i][1], state_after_action_1[1]], [encoded_batch_numpy4[i][2], encoded_batch_numpy4[i][2]], 'green', alpha=0.3, linewidth=edge_size)
        ax.plot3D([encoded_batch_numpy4[i][0], state_after_action_2[0]], [encoded_batch_numpy4[i][1], state_after_action_2[1]], [encoded_batch_numpy4[i][2], encoded_batch_numpy4[i][2]], 'blue', alpha=0.3, linewidth=edge_size)
        ax.plot3D([encoded_batch_numpy4[i][0], state_after_action_3[0]], [encoded_batch_numpy4[i][1], state_after_action_3[1]], [encoded_batch_numpy4[i][2], encoded_batch_numpy4[i][2]], 'yellow', alpha=0.3, linewidth=edge_size)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])

    # ax.set_xlabel('Controllable latent 1')
    # ax.set_ylabel('Controllable latent 2')
    # ax.set_zlabel('Uncontrollable latent')

    ax.view_init(elev=8)
    fig.set_size_inches(w=5.50107/2, h=3)

    if args.showplot:
        plt.show()
    else:
        if args.format =='png':
            plt.savefig(str(run_directory)+'/fourmaze'+str(agent.iterations)+'_iterations.png')
        elif args.format =='pdf':
            plt.savefig(str(run_directory)+'/fourmaze'+str(agent.iterations)+'_iterations.pdf')

        plt.close()
