import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim

from .SanmillNNet import SanmillNNet as snnet

class NNetWrapper(NeuralNet):
    def __init__(self, game, args):
        self.args = args
        self.nnet = snnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()

        if args.cuda:
            self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v, period)
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=self.args.lr)

        for epoch in range(self.args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / self.args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.choice(len(examples), size=self.args.batch_size, replace=False)
                boards, pis, vs, periods = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.tensor(boards, dtype=torch.float32)
                target_pis = torch.tensor(pis, dtype=torch.float32)
                target_vs = torch.tensor(vs, dtype=torch.float32)
                periods = torch.tensor(periods, dtype=torch.int8)

                # predict
                if self.args.cuda:
                    # boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()
                    boards = boards.contiguous().cuda()
                    target_pis = target_pis.contiguous().cuda()
                    target_vs = target_vs.contiguous().cuda()
                    periods = periods.contiguous().cuda()

                # compute output
                out_pi = torch.zeros(target_pis.size()).cuda()
                out_v = torch.zeros(target_vs.size()).cuda()
                for i in range(4):
                    pi, v = self.nnet(boards[periods==i], i)
                    out_pi[periods==i] = pi.view(-1, target_pis.size(1))
                    out_v[periods==i] = v.view(-1)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board, period):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = torch.tensor(board, dtype=torch.float)
        period = torch.tensor(period, dtype=torch.int8)
        if self.args.cuda: board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board, period)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]
        # alpha = 0.25

        # positive = (targets>1e-3).float()

        # alpha_w = alpha * positive + (1-alpha) * positive
        # p_t = positive * torch.exp(outputs) + (1-positive) * (1-torch.exp(outputs))
        # focal_loss = -torch.sum(alpha_w * targets * (1-p_t)**2 * torch.log(p_t)) / targets.size()[0]
        # return focal_loss

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            # raise ("No model in path {}".format(filepath))
            print("No model in path {}".format(filepath))
            exit(1)
        map_location = None if self.args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
