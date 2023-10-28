import sys
sys.path.append('..')
from utils import *
import torch
import torch.nn as nn
from sanmill.SanmillLogic import Board
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Branch03(nn.Module):
    def __init__(self, args):
        super(Branch03, self).__init__()
        self.args = args
        self.main = nn.Sequential(
            nn.LayerNorm(512*9),
            nn.ReLU(),
            nn.Linear(512*9, args.num_channels*2),
            nn.Dropout(args.dropout),
            nn.LayerNorm(args.num_channels*2),
            nn.ReLU(),
            nn.Linear(args.num_channels*2, args.num_channels),
            nn.Dropout(args.dropout),
        )
        self.main_identity = nn.Linear(512*9, args.num_channels)
        self.pi = nn.Sequential(
            nn.ReLU(),
            nn.Linear(args.num_channels, 24),
            nn.LayerNorm(24),
            nn.LogSoftmax(dim=1),
        )
        self.v = nn.Sequential(
            nn.LayerNorm(args.num_channels),
            nn.ReLU(),
            nn.Linear(args.num_channels, 1),
            nn.Tanh(),
        )

    def forward(self, s):
        """
        s: batch_size x 512*9
        """
        s = self.main(s) + self.main_identity(s)  # batch_size x num_channels
        pi = torch.ones((s.size()[0], 24*24)).to(device) * -10
        pi[:,:24] = self.pi(s)  # batch_size x 24
        v = self.v(s)  # batch_size x 1
        return pi, v

class Branch14(nn.Module):
    def __init__(self, args):
        super(Branch14, self).__init__()
        # 4 mean the period that your opponent has 3 pieces on the board,
        # and you have at least 4 pieces on the board.
        self.args = args
        self.b = Board()
        self.cache_valids()
        self.main = nn.Sequential(
            nn.LayerNorm(512*9),
            nn.ReLU(),
            nn.Linear(512*9, args.num_channels*2),
            nn.Dropout(args.dropout),
            nn.LayerNorm(args.num_channels*2),
            nn.ReLU(),
            nn.Linear(args.num_channels*2, args.num_channels),
            nn.Dropout(args.dropout),
        )
        self.main_identity = nn.Linear(512*9, args.num_channels)
        self.pi = nn.Sequential(
            nn.ReLU(),
            nn.Linear(args.num_channels, 80),
            nn.LayerNorm(80),
            nn.LogSoftmax(dim=1),
        )
        self.v = nn.Sequential(
            nn.LayerNorm(args.num_channels),
            nn.ReLU(),
            nn.Linear(args.num_channels, 1),
            nn.Tanh(),
        )

    def cache_valids(self):
        self.valids = []
        legalMoves = self.b.get_valids_in_period1()
        for move in legalMoves:
            action = self.b.get_action_from_move(move)
            self.valids.append(action)

    def forward(self, s):
        """
        s: batch_size x 512*9
        """
        s = self.main(s) + self.main_identity(s)  # batch_size x num_channels
        pi = torch.ones((s.size()[0], 24*24),dtype=torch.float).to(device) * -10
        pi[:, self.valids] = self.pi(s)
        v = self.v(s)  # batch_size x 1
        return pi, v

class Branch2(nn.Module):
    def __init__(self, args):
        super(Branch2, self).__init__()
        self.main = nn.Sequential(
            nn.LayerNorm(512*9),
            nn.ReLU(),
            nn.Linear(512*9, args.num_channels*2),
            nn.Dropout(args.dropout),
            nn.LayerNorm(args.num_channels*2),
            nn.ReLU(),
            nn.Linear(args.num_channels*2, args.num_channels),
            nn.Dropout(args.dropout),
        )
        self.main_identity = nn.Linear(512*9, args.num_channels)
        self.pi = nn.Sequential(
            nn.ReLU(),
            nn.Linear(args.num_channels, 24*24),
            nn.LayerNorm(24*24),
            nn.LogSoftmax(dim=1),
        )
        self.v = nn.Sequential(
            nn.LayerNorm(args.num_channels),
            nn.ReLU(),
            nn.Linear(args.num_channels, 1),
            nn.Tanh(),
        )
    def forward(self, s):
        """
        s: batch_size x 512*9
        """
        s = self.main(s) + self.main_identity(s)  # batch_size x num_channels
        pi = self.pi(s)  # batch_size x 24*24
        v = self.v(s)  # batch_size x 1
        return pi, v

class SanmillNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.args = args

        super(SanmillNNet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, args.num_channels//4, 3, stride=1, padding=1),
            nn.BatchNorm2d(args.num_channels//4),
            nn.ReLU(),
            nn.Conv2d(args.num_channels//4, args.num_channels//2, 3, stride=1),
            nn.BatchNorm2d(args.num_channels//2),
            nn.ReLU(),
            nn.Conv2d(args.num_channels//2, args.num_channels, 3, stride=1),
            nn.BatchNorm2d(args.num_channels),
            nn.ReLU(),
        )
        self.branch = nn.ModuleList([Branch03(args), Branch14(args), Branch2(args), Branch03(args), Branch14(args)])
        self.attension = nn.MultiheadAttention(args.num_channels, 4, batch_first=True)

    def forward(self, s, period):
        """
        s: batch_size x board_x x board_y
        period: 1
        """
        s = s.view(-1, 1, self.board_x, self.board_y)  # batch_size x 1 x (board_x-4) x (board_y-4)
        s = self.main(s)  # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, self.args.num_channels, (self.board_x-4)*(self.board_y-4)).permute(0,2,1)  # batch_size x (board_x-4)*(board_y-4) x num_channels
        s, _ = self.attension(s, s, s)  # batch_size x (board_x-4)*(board_y-4) x num_channels
        s = s.reshape(-1, self.args.num_channels*(self.board_x-4)*(self.board_y-4))  # batch_size x num_channels*(board_x-4)*(board_y-4)
        pi, v = self.branch[period](s)
        return pi, v