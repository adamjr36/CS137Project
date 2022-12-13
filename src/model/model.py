import torch
import torch.nn as nn
import torch.nn.functional as F


class TeamModel(nn.Module):

    def __init__(self, input_size, hidden_size, feature_size):
        super(TeamModel, self).__init__()
        self.encoder = nn.Sequential(*[
            nn.Flatten(),
            nn.Linear(input_size*10, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, feature_size)
        ])
    
    def forward(self, x):
        return self.encoder(x)


class BaseModel(nn.Module):

    def __init__(self, input_size, feature_size, hidden_size1, hidden_size2, output_size=3):
        super(BaseModel, self).__init__()
        self.team_analyzer = TeamModel(input_size, hidden_size1, feature_size)
        self.predictor = nn.Sequential(*[
            nn.Linear(feature_size*2, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, output_size),
            nn.Softmax()
        ])

    def forward(self, x, y):
        x_features = self.team_analyzer(x)
        # print(x_features.shape)
        y_features = self.team_analyzer(y)
        # print(y_features.shape)
        features = torch.concat([x_features, y_features], axis=1)
        outcome = self.predictor(features)
        # print(outcome.shape)
        return outcome




