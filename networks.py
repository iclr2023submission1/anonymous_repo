import torch.nn as nn
import torch


class EncoderDMC_8x8wall(nn.Module):
    """Convolutional encoder for image-based observations."""
    def __init__(self, obs_channels, latent_dim, tanh=False, scale=16, neuron_dim=100):
        super().__init__()

        self.latent_dim = latent_dim
        self.obs_channels = obs_channels
        self.outputs = dict()
        self.tanh = tanh
        self.scale = scale

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=self.obs_channels, out_channels=int(32), kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(32), out_channels=int(32), kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )

        self.conv_walls = nn.Sequential(
            nn.Conv2d(in_channels=int(32), out_channels=int(1), kernel_size=(4, 4), stride=(1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=6),  # Todo

        )

        self.dummy_input = torch.ones((1, 1, 48, 48))
        self.dummy_output = self.convs(self.dummy_input)
        self.fc_size = self.dummy_output.flatten(1).shape[1]
        self.wall_output = self.conv_walls(self.dummy_output)
        self.wall_size = self.wall_output.shape[2]**2

        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.fc_size, out_features=neuron_dim),
            nn.Tanh(),
            nn.Linear(in_features=neuron_dim, out_features=self.latent_dim),
            nn.Tanh())

    def forward(self, obs, detach=''):

        if len(obs.shape) == 3:
            obs = obs.unsqueeze(1)
        features = self.convs(obs)

        if detach:
            if detach =='base':
                features = features.detach()


        wall_features = self.conv_walls(features)
        self.outputs['wall_features'] = wall_features

        features = features.flatten(1)

        latent_agent = self.mlp(features)

        self.outputs['output_MlP'] = latent_agent

        if detach:
            if detach =='wall':
                wall_features = wall_features.detach()
            elif detach =='agent':
                latent_agent = latent_agent.detach()
            elif detach =='both':
                wall_features = wall_features.detach()
                latent_agent = latent_agent.detach()

        return latent_agent, wall_features

    
class EncoderDMC_half_features_catcher(nn.Module):
    """Convolutional encoder for image-based observations."""
    def __init__(self, obs_channels, latent_dim, scale=1, neuron_dim=100):
        super().__init__()

        self.latent_dim = latent_dim
        self.obs_channels = obs_channels
        self.outputs = dict()
        self.scale = scale

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=self.obs_channels, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )

        self.conv_ball = nn.Sequential(
            # nn.Conv2d(in_channels=int(32), out_channels=int(32), kernel_size=(1, 1), stride=(1, 1)),
            # nn.ReLU(),
            nn.Conv2d(in_channels=int(32), out_channels=int(1), kernel_size=(4, 4), stride=(1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=6),  # Todo
        )

        self.mlp = nn.Sequential(
            nn.Linear(in_features=16928, out_features=neuron_dim),
            nn.Tanh(),
            nn.Linear(in_features=neuron_dim, out_features=self.latent_dim),
            nn.Tanh())

    def forward(self, obs, detach=str):

        if len(obs.shape) == 3:
            obs = obs.unsqueeze(1)
        features = self.convs(obs)
        self.outputs['features'] = features

        if detach:
            if detach =='base':
                features = features.detach()

        ball_features = self.conv_ball(features)

        features = features.flatten(1)

        latent_agent = self.mlp(features)

        if detach:                  # TODO check if algorithm works with the new detachment.
            if detach =='ball':
                ball_features = ball_features.detach()
            elif detach =='agent':
                latent_agent = latent_agent.detach()
            elif detach =='both':
                ball_features = ball_features.detach()
                latent_agent = latent_agent.detach()

        return latent_agent, ball_features


class TransitionModel(nn.Module):
    """ Transition function MLP head for both w/o and w/ action"""
    def __init__(self, latent_dim, action_dim, scale=1, tanh=False, prediction_dim=2):
        super().__init__()

        self.input_dim = latent_dim + action_dim
        self.outputs = dict()

        self.prediction_dim = prediction_dim
        self.counter = 0
        self.tanh = tanh

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=int(8*scale)),
            nn.Tanh(),
            nn.Linear(in_features=int(8*scale), out_features=int(32*scale)),
            nn.Tanh(),
            nn.Linear(in_features=int(32*scale), out_features=int(32*scale)),
            nn.Tanh(),
            # nn.Linear(in_features=int(32 / scale), out_features=int(32 / scale)),
            # nn.Tanh(),
            nn.Linear(in_features=int(32*scale), out_features=int(8*scale)),
            nn.Tanh(),
            nn.Linear(in_features=int(8*scale), out_features=self.prediction_dim),
        )   # TODO test larger sizes of transitionmodels for perfect transitions in multimazes??


    def forward(self, z, detach=False):

        prediction = self.linear_layers(z)

        if detach:
            prediction = prediction.detach()

        return prediction


class DQNmodel(nn.Module):
    """ DQN Model"""
    def __init__(self, input_dim, scale=16, prediction_dim=4):
        super().__init__()

        self.counter = 0
        self.scale = int(scale)

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=int(8*scale)),
            nn.Tanh(),
            nn.Linear(in_features=int(8*scale), out_features=int(32*scale)),
            nn.Tanh(),
            nn.Linear(in_features=int(32*scale), out_features=int(32*scale)),
            nn.Tanh(),
            nn.Linear(in_features=int(32*scale), out_features=int(8*scale)),
            nn.Tanh(),
            nn.Linear(in_features=int(8*scale), out_features=prediction_dim),
        )

    def forward(self, controllable_latent, uncontrollable_features, detach=False):

        q_values = self.linear_layers(torch.cat((controllable_latent, uncontrollable_features.flatten(1)), dim=1))

        if detach:
            q_values = q_values.detach()

        return q_values


class InversePredictionModel(nn.Module):
    """ MLP for the inverse prediction"""

    def __init__(self, input_dim, num_actions=4, scale=1, activation='tanh', final_activation=True):
        super().__init__()

        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()

        self.final_activation = final_activation
        self.scale = int(scale)

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=int(8 * scale)),
            self.activation,
            nn.Linear(in_features=int(8 * scale), out_features=int(32 * scale)),
            self.activation,
            nn.Linear(in_features=int(32 * scale), out_features=int(32 * scale)),
            self.activation,
            nn.Linear(in_features=int(32 * scale), out_features=int(8 * scale)),
            self.activation,
            nn.Linear(in_features=int(8 * scale), out_features=num_actions),
        )

    def forward(self, z, detach=False):

        prediction = self.linear_layers(z)

        if detach:
            prediction = prediction.detach()

        return prediction
