import sys
sys.path.append('..')
from utils import *

import torch.nn as nn

class Branch(nn.Module):
    def __init__(self, args):
        super(Branch, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(512*9, args.num_channels*2),
            nn.LayerNorm(args.num_channels*2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.num_channels*2, args.num_channels),
            nn.LayerNorm(args.num_channels),
            nn.ReLU(),
            nn.Dropout(args.dropout),
        )
        self.identity = nn.Linear(512*9, args.num_channels)
        self.pi = nn.Sequential(
            nn.Linear(args.num_channels, 24*24+1),
            nn.LogSoftmax(dim=1),
        )
        self.v = nn.Sequential(
            nn.Linear(args.num_channels, 1),
            nn.Tanh(),
        )

    def forward(self, s):
        """
        s: batch_size x 512*9
        """
        s = self.main(s) + self.identity(s)  # batch_size x num_channels
        pi = self.pi(s)  # batch_size x 24*24+1
        v = self.v(s)
        return pi, v


class SanmillNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.args = args

        super(SanmillNNet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, args.num_channels//4, 3, stride=1, padding=1),
            nn.InstanceNorm2d(args.num_channels//4),
            nn.ReLU(),
            nn.Conv2d(args.num_channels//4, args.num_channels//4, 3, stride=1, padding=1),
            nn.InstanceNorm2d(args.num_channels//4),
            nn.ReLU(),
            nn.Conv2d(args.num_channels//4, args.num_channels//2, 3, stride=1),
            nn.InstanceNorm2d(args.num_channels//2),
            nn.ReLU(),
            nn.Conv2d(args.num_channels//2, args.num_channels, 3, stride=1),
            nn.InstanceNorm2d(args.num_channels),
            nn.ReLU(),
        )
        self.branch = nn.ModuleList([Branch(args) for _ in range(4)])

    def forward(self, s, period):
        """
        s: batch_size x board_x x board_y
        period: 1
        """
        s = s.view(-1, 1, self.board_x, self.board_y)  # batch_size x 1 x (board_x-4) x (board_y-4)
        s = self.main(s)  # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, self.args.num_channels*(self.board_x-4)*(self.board_y-4))  # batch_size x num_channels*(board_x-4)*(board_y-4)
        pi, v = self.branch[period](s)
        return pi, v