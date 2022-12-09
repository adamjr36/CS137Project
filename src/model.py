import torch
import torch.nn as nn
import torch.nn.functional as F


class TeamModel(nn.Module):

    def __init__(self, feature_size):
        super(TeamModel, self).__init__()


class BaseModel(nn.Module):

    def __init__(self, feature_size, hidden_size, output_size=3):
        super(BaseModel, self).__init__()
        self.team_analyzer = TeamModel(feature_size)
        self.predictor = nn.Sequential(*[
            nn.Linear(feature_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax()
        ])

    def forward(self, x, y, z):
        x_features = self.team_analyzer(x, z)
        y_features = self.team_analyzer(y, z)
        features = torch.concat(x_features, y_features)
        outcome = self.predictor(features)
        return outcome




