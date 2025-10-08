#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 15:38:42 2025

@author: adan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),#
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),#
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()#
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forwardConv(self, x):
        tmp1 = self.image_conv[0](x)
        tmp2 = self.image_conv[1](tmp1)
        tmp3 = self.image_conv[2](tmp2)
        tmp4 = self.image_conv[3](tmp3)
        tmp5 = self.image_conv[4](tmp4)
        tmp6 = self.image_conv[5](tmp5)
        tmp7 = self.image_conv[6](tmp6)

        return tmp7, [tmp1,tmp2,tmp3,tmp4,tmp5,tmp6, tmp7]

    def forwardActor(self, x):
        tmp1 = self.actor[0](x)
        tmp2 = self.actor[1](tmp1)
        tmp3 = self.actor[2](tmp2)
        return tmp3, [tmp1,tmp2,tmp3]


    def forward(self, obs, memory):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x, tmp = self.forwardConv(x)
        # x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)
            tmp.append(embedding)

        # x = self.actor(embedding)
        x, tmp2 = self.forwardActor(embedding)

        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory, [tmp, tmp2]

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]

class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, obs_space, action_space, model_dir,
                 argmax=False, num_envs=1, use_memory=False, use_text=False):
        obs_space, self.preprocess_obss = utils.get_obss_preprocessor(obs_space)
        self.acmodel = ACModel(obs_space, action_space, use_memory=use_memory, use_text=use_text)
        self.argmax = argmax
        self.num_envs = num_envs

        if self.acmodel.recurrent:
            self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size, device=device)
        self.hidden = []

        self.acmodel.load_state_dict(utils.get_model_state(model_dir))
        self.acmodel.to(device)
        self.acmodel.eval()
        if hasattr(self.preprocess_obss, "vocab"):
            self.preprocess_obss.vocab.load_vocab(utils.get_vocab(model_dir))

    def get_actions(self, obss):
        preprocessed_obss = self.preprocess_obss(obss, device=device)

        with torch.no_grad():
            if self.acmodel.recurrent:
                dist, _, self.memories, self.hidden = self.acmodel(preprocessed_obss, self.memories)
            else:
                dist, _ = self.acmodel(preprocessed_obss)

        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        return actions.cpu().numpy()

    def get_action(self, obs):
        return self.get_actions([obs])[0]

    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float, device=device).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])


import torch
import numpy as np

obs_space = {'image': (7, 7, 3), 'text': 100}

env = utils.make_env('MiniGrid-GoToDoor-5x5-v0', 123, render_mode="human")
env.reset()

agent = Agent(env.observation_space, env.action_space, './storage/GoToDoor3/',
              use_text=True)



start = [2,5,0]
start7 = np.block([start, start, start, start, start, start, start])
start5 = np.block([start, start, start, start, start])
side_door = np.vstack([start5, [4, 9, 1], start])
start4 = np.block([start, start, start, start])
grid = np.vstack([start4, [1, 0, 0], [1, 0, 0], [1, 0, 0]])
central_door = grid.copy()
central_door[3] = [4, 8, 1]
side_door2 = np.vstack([start5, [4, 7, 1], start])
rep = np.concatenate([[start7], [side_door], [grid], [central_door], [grid], [side_door2], [start7]])
rep_list = []
text_list = []
color = ['red', 'green', 'blue', 'purple', 'yellow', 'grey']
for left_door in [0,1,2,3,4,5]:
  for goal_door in [0,1,2,3,4,5]:
    for right_door in [0,1,2,3,4,5]:
      if(goal_door != left_door and goal_door != right_door and left_door != right_door):
        rep2 = np.where(rep == 9, left_door, rep)
        rep3 = np.where(rep2 == 8, goal_door, rep2)
        rep4 = np.where(rep2 == 7, right_door, rep3)
        text = 'go to the '+ color[goal_door] +' door'
        rep_list.append(rep4)
        text_list.append(text)

start = [2,5,0]
start7 = np.block([start, start, start, start, start, start, start])
start5 = np.block([start, start, start, start, start])
side_door = np.vstack([start5, [4, 9, 1], start])
start4 = np.block([start, start, start, start])
grid = np.vstack([start4, [1, 0, 0], [1, 0, 0], [1, 0, 0]])
central_door = grid.copy()
central_door[3] = [4, 8, 1]
side_door2 = np.vstack([start5, [4, 7, 1], start])
rep = np.concatenate([[start7], [side_door], [grid], [central_door], [grid], [side_door2], [start7]])
for i in range(len(rep)):
  x = rep[i]
  y = x[:6]
  z = np.vstack([start, y])
  rep[i] = z
rep_list = []
text_list = []
color = ['red', 'green', 'blue', 'purple', 'yellow', 'grey']
for left_door in [0,1,2,3,4,5]:
  for goal_door in [0,1,2,3,4,5]:
    for right_door in [0,1,2,3,4,5]:
      if(goal_door != left_door and goal_door != right_door and left_door != right_door):
        rep2 = np.where(rep == 9, left_door, rep)
        rep3 = np.where(rep2 == 8, goal_door, rep2)
        rep4 = np.where(rep2 == 7, right_door, rep3)
        text = 'go to the '+ color[left_door] +' door'
        rep_list.append(rep4)
        text_list.append(text)




hiddenLayerValues = []
actions = []
for idx, (image, text) in enumerate(zip(rep_list, text_list)):
    if idx > 24:
        break
    obs = {'image': image, 
           'direction': np.int64(1), 
           'mission': text}
    
    action = agent.get_action(obs)
    #hidden[0]: 0 - 6 convolutional, 7 textembeddings
    #hidden[1]: 0 - 2 actor network
    hidden = agent.hidden
    hiddenLayerValues.append(hidden)
    actions.append(torch.tensor(action))
    print(action)
    

conv1 = torch.stack([data[0][0].flatten() for data in hiddenLayerValues])
conv2 = torch.stack([data[0][1].flatten() for data in hiddenLayerValues])
conv3 = torch.stack([data[0][2].flatten() for data in hiddenLayerValues])
conv4 = torch.stack([data[0][3].flatten() for data in hiddenLayerValues])
conv5 = torch.stack([data[0][4].flatten() for data in hiddenLayerValues])
conv6 = torch.stack([data[0][5].flatten() for data in hiddenLayerValues])
conv7 = torch.stack([data[0][6].flatten() for data in hiddenLayerValues])
conv8 = torch.stack([data[0][7].flatten() for data in hiddenLayerValues])
action = torch.stack(actions)

l1 = torch.stack([data[1][0].flatten() for data in hiddenLayerValues])
l2 = torch.stack([data[1][1].flatten() for data in hiddenLayerValues])
l3 = torch.stack([data[1][2].flatten() for data in hiddenLayerValues])

conv1_mean = conv1.mean(1)
conv2_mean = conv2.mean(1)
conv3_mean = conv3.mean(1)
conv4_mean = conv4.mean(1)
conv5_mean = conv5.mean(1)
conv6_mean = conv6.mean(1)
conv7_mean = conv7.mean(1)
conv8_mean = conv8.mean(1)
action_mean = 0#action.mean(1)

l1_mean = conv5.mean(1)
l2_mean = conv6.mean(1)
l3_mean = conv7.mean(1)


conv1_tmp = conv1 - conv1_mean.unsqueeze(1)
conv2_tmp = conv2 - conv2_mean.unsqueeze(1)
conv3_tmp = conv3 - conv3_mean.unsqueeze(1)
conv4_tmp = conv4 - conv4_mean.unsqueeze(1)
conv5_tmp = conv5 - conv5_mean.unsqueeze(1)
conv6_tmp = conv6 - conv6_mean.unsqueeze(1)
conv7_tmp = conv7 - conv7_mean.unsqueeze(1)
conv8_tmp = conv8 - conv8_mean.unsqueeze(1)
action_tmp = action# - action_mean.unsqueeze(1)

l1_tmp = l1 - l1_mean.unsqueeze(1)
l2_tmp = l2 - l2_mean.unsqueeze(1)
l3_tmp = l3 - l3_mean.unsqueeze(1)

conv1_sq = torch.sqrt(1/conv1_mean.shape[0] * torch.sum((torch.sum(conv1_tmp**2))**2, 0))
conv2_sq = torch.sqrt(1/conv1_mean.shape[0] * torch.sum((torch.sum(conv2_tmp**2))**2, 0))
conv3_sq = torch.sqrt(1/conv1_mean.shape[0] * torch.sum((torch.sum(conv3_tmp**2))**2, 0))
conv4_sq = torch.sqrt(1/conv1_mean.shape[0] * torch.sum((torch.sum(conv4_tmp**2))**2, 0))
conv5_sq = torch.sqrt(1/conv1_mean.shape[0] * torch.sum((torch.sum(conv5_tmp**2))**2, 0))
conv6_sq = torch.sqrt(1/conv1_mean.shape[0] * torch.sum((torch.sum(conv6_tmp**2))**2, 0))
conv7_sq = torch.sqrt(1/conv1_mean.shape[0] * torch.sum((torch.sum(conv7_tmp**2))**2, 0))
conv8_sq = torch.sqrt(1/conv1_mean.shape[0] * torch.sum((torch.sum(conv8_tmp**2))**2, 0))
action_sq = torch.sqrt(1/conv1_mean.shape[0] * torch.sum((torch.sum(action_tmp**2))**2, 0))

l1_sq = torch.sqrt(1/l1_mean.shape[0] * torch.sum((torch.sum(l1_tmp**2))**2, 0))
l2_sq = torch.sqrt(1/l2_mean.shape[0] * torch.sum((torch.sum(l2_tmp**2))**2, 0))
l3_sq = torch.sqrt(1/l3_mean.shape[0] * torch.sum((torch.sum(l3_tmp**2))**2, 0))

# conv1: nn.Conv2d(3, 16, (2, 2)),
# conv2: nn.ReLU(),#
# conv3: nn.MaxPool2d((2, 2)),
# conv4: nn.Conv2d(16, 32, (2, 2)),
# conv5: nn.ReLU(),#
# conv6: nn.Conv2d(32, 64, (2, 2)),
# conv7: nn.ReLU()#

# conv8: Text embeddings

# l1: nn.Linear(self.embedding_size, 64),
# l2: nn.Tanh(),
# l3: nn.Linear(64, action_space.n)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(2)
plt.plot([#conv1_sq.item(),
          conv2_sq.item(),
          #conv3_sq.item(),
          # conv4_sq.item(),
          conv5_sq.item(),
          # conv6_sq.item(),
          conv7_sq.item(),
          # l1_sq.item(),
          l2_sq.item(),
          l3_sq.item(),
          action_sq.item(),
          ])
# plt.plot([#conv1_sq.item(),
#           conv2_sq.item(),
#           #conv3_sq.item(),
#           #conv4_sq.item(),
#           conv5_sq.item(),
#           #conv6_sq.item(),
#           conv7_sq.item(),
#           #l1_sq.item(),
#           l2_sq.item(),
#           l3_sq.item(),
#           action_sq.item(),
#           ])