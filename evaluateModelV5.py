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
        # print('inside network',len(tmp), len(tmp2))

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

env = utils.make_env('MiniGrid-GoToDoor-5x5-v0', 123)
env.reset()

agent = Agent(env.observation_space, env.action_space, './storage/PPO/GoToDoor4/', use_text=True)
agent = Agent(env.observation_space, env.action_space, './storage/A2C/GoToDoor2/', use_text=True)
# agent = Agent(env.observation_space, env.action_space, './storage/GoToDoor/', use_text=True)


goal_door = {"Left_Door": [[]], "Right_Door": [[]], "Front_Door": [[]], "Back_Door": [[]]}
color = ['red', 'green', 'blue', 'purple', 'yellow', 'grey']

start = [2,5,0]
start7 = np.block([start, start, start, start, start, start, start])
start5 = np.block([start, start, start, start, start])
side_door = np.vstack([start5, [4, 9, 1], start])
start4 = np.block([start, start, start, start])
grid = np.vstack([start4, [1, 0, 0], [1, 0, 0], [1, 0, 0]])
central_door = grid.copy()
central_door[3] = [4, 8, 1]
side_door2 = np.vstack([start5, [4, 7, 1], start])
exmp_state_center = np.concatenate([[start7], [side_door], [grid], [central_door], [grid], [side_door2], [start7]])
exmp_state_left = np.concatenate([[start7], [start7], [side_door], [grid], [central_door], [grid], [side_door2]])
exmp_state_right = np.concatenate([[side_door], [grid], [central_door], [grid], [side_door2], [start7], [start7]])
exmp_state_list = [exmp_state_center, exmp_state_left, exmp_state_right]

for shift_up in [0, 1, 2]:
    for exmp_state in exmp_state_list:
      base_state = exmp_state.copy()
      for shifts in range(shift_up):
          for i in range(len(base_state)):
            x = base_state[i]
            y = x[:6]
            z = np.vstack([start, y])
            base_state[i] = z

      for left_door in range(6):
          for right_door in range(6):
              for front_door in range(6):
                  for back_door in range(6):
                    x = [left_door, right_door, front_door, back_door]
                    if(len(x) <= len(set(x))):
                        state_w_leftd = np.where(base_state == 9, left_door, base_state)
                        state_w_frontd = np.where(state_w_leftd == 8, front_door, state_w_leftd)
                        final_state = np.where(state_w_frontd == 7, right_door, state_w_frontd)

                        left_door_goal = 'go to the '+ color[left_door] +' door'
                        goal_door['Left_Door'][-1].append({'image':final_state,'text':left_door_goal})
                        right_door_goal = 'go to the '+ color[right_door] +' door'
                        goal_door['Right_Door'][-1].append({'image':final_state,'text':right_door_goal})
                        front_door_goal = 'go to the '+ color[front_door] +' door'
                        goal_door['Front_Door'][-1].append({'image':final_state,'text':front_door_goal})
                        back_door_goal = 'go to the '+ color[back_door] +' door'
                        goal_door['Back_Door'][-1].append({'image':final_state,'text':back_door_goal})

      goal_door['Left_Door'].append([])   
      goal_door['Right_Door'].append([])
      goal_door['Front_Door'].append([])
      goal_door['Back_Door'].append([])    

hiddenLayerValues_dict = {}
convs1 = []
convs2 = []
convs3 = []
convs4 = []
convs5 = []
convs6 = []
convs7 = []
convs8 = []
actions = []
ls1 = []
ls2 = []
ls3 = []
for goal_key in goal_door.keys():
    goal = goal_door[goal_key]
    rep_list = []
    text_list = []
    for states in goal[:-1]:
        for state in states:
            rep_list.append(state['image'])
            text_list.append(state['text'])
    
    
    hiddenLayerValues = []
    # convs1 = []
    # convs2 = []
    # convs3 = []
    # convs4 = []
    # convs5 = []
    # convs6 = []
    # convs7 = []
    # convs8 = []
    # actions = []
    # ls1 = []
    # ls2 = []
    # ls3 = []
    for idx, (image, text) in enumerate(zip(rep_list, text_list)):
        # if idx > 24:
        #     break
        obs = {'image': image, 
               'direction': np.int64(1),
               'mission': text}
        
        action = agent.get_action(obs)
        #hidden[0]: 0 - 6 convolutional, 7 textembeddings
        #hidden[1]: 0 - 2 actor network
        hidden = agent.hidden
        # if not action==0:
        #     continue
        hiddenLayerValues.append(hidden)
        actions.append(torch.tensor(action))
        # print(action)
    
        conv1 = hidden[0][0].flatten()
        conv2 = hidden[0][1].flatten()
        conv3 = hidden[0][2].flatten()
        conv4 = hidden[0][3].flatten()
        conv5 = hidden[0][4].flatten()
        conv6 = hidden[0][5].flatten()
        conv7 = hidden[0][6].flatten()
        conv8 = hidden[0][7].flatten()
        action = action
    
        l1 = hidden[1][0].flatten()
        l2 = hidden[1][1].flatten()
        l3 = hidden[1][2].flatten()
        
        convs1.append(conv1)
        convs2.append(conv2)
        convs3.append(conv3)
        convs4.append(conv4)
        convs5.append(conv5)
        convs6.append(conv6)
        convs7.append(conv7)
        convs8.append(conv8)
        ls1.append(l1)
        ls2.append(l2)
        ls3.append(l3)
    
convs1 = torch.stack(convs1)
convs2 = torch.stack(convs2)
convs3 = torch.stack(convs3)
convs4 = torch.stack(convs4)
convs5 = torch.stack(convs5)
convs6 = torch.stack(convs6)
convs7 = torch.stack(convs7)
convs8 = torch.stack(convs8)
actions = torch.stack(actions)
ls1 = torch.stack(ls1)
ls2 = torch.stack(ls2)
ls3 = torch.stack(ls3)
    
    # hiddenLayerValues_dict[goal_key] = {"hidden":hiddenLayerValues, "action": action}
    # hiddenLayerValues_dict[goal_key] = hiddenLayerValues

# def get_dispersion(nconv):
#   print(nconv.shape)
#   nconv = nconv.view(nconv.shape[0],-1)
#   nconv_mean = nconv.mean(1)
#   conv1_tmp = nconv - nconv_mean.unsqueeze(1)
#   conv1_sq = torch.sqrt(1/nconv_mean.shape[0] * torch.sum((torch.sum(conv1_tmp**2))**2, 0))

#   return conv1_sq

def get_dispersion(nconv):
  # print(nconv.shape)
  nconv = nconv.view(nconv.shape[0],-1)
  nconv_mean = nconv.mean(1)
  conv1_tmp = nconv - nconv_mean.unsqueeze(1)
  dontknow = torch.sum(conv1_tmp**2)
  # print(dontknow)
  conv1_sq = torch.sqrt(1/nconv_mean.shape[0] * torch.sum(torch.sum(conv1_tmp**2), 0))

  return conv1_sq

network_activations = [convs1, convs2, convs3, convs4, convs5, convs6, convs7, convs8,
                       ls1, ls2, ls3,
                       actions.to(device).float()]

dispersions = []
dispersions_n = []
for relu_ in network_activations:
  mean_list = []
  mean = torch.zeros(relu_[0].flatten().shape).to(device)
  for i in range(relu_.shape[0]):
    mean += relu_[i].flatten()
    if i%360==1:
      mean_list.append(mean/360)
      mean = torch.zeros(relu_[0].flatten().shape).to(device)

  denominator = 0
  count = 0
  # for i in range(int(relu_.shape[0]/4)):
  #   for j in range(int(relu_.shape[0]/4)):
  for i in range(int(36)):
    for j in range(int(36)):
        if i != j:
          denominator += torch.sum((mean_list[i] - mean_list[j])**2)
          count+=1

  denominator = denominator/count

  numerator = 0
  for i in range(relu_.shape[0]):
    if i%(idx + 1)==0:
      numerator += get_dispersion(relu_[i:i+(idx + 1)])
  numerator = numerator/int(relu_.shape[0]/(idx + 1))

  dispersions.append((numerator/denominator).item())
  dispersions_n.append(numerator)
  # dispersions.append(numerator.item())

    
    
    
# 4 goals
# 9 states (10  with 1 empty)
# 360 sim states
# goal_door['Left_Door']
# print(torch.sum(torch.stack(actions) < 3)/len(actions))


# conv1 = torch.stack([data[0][0].flatten() for data in hiddenLayerValues])
# conv2 = torch.stack([data[0][1].flatten() for data in hiddenLayerValues])
# conv3 = torch.stack([data[0][2].flatten() for data in hiddenLayerValues])
# conv4 = torch.stack([data[0][3].flatten() for data in hiddenLayerValues])
# conv5 = torch.stack([data[0][4].flatten() for data in hiddenLayerValues])
# conv6 = torch.stack([data[0][5].flatten() for data in hiddenLayerValues])
# conv7 = torch.stack([data[0][6].flatten() for data in hiddenLayerValues])
# conv8 = torch.stack([data[0][7].flatten() for data in hiddenLayerValues])
# action = torch.stack(actions)

# l1 = torch.stack([data[1][0].flatten() for data in hiddenLayerValues])
# l2 = torch.stack([data[1][1].flatten() for data in hiddenLayerValues])
# l3 = torch.stack([data[1][2].flatten() for data in hiddenLayerValues])

# action_mean = 0#action.mean(1)
# action_tmp = action# - action_mean.unsqueeze(1)
# action_sq = torch.sqrt(1/action.shape[0] * torch.sum((torch.sum(action_tmp**2))**2, 0))

# conv1_sq = get_dispersion(conv1)
# conv2_sq = get_dispersion(conv2)
# conv3_sq = get_dispersion(conv3)
# conv4_sq = get_dispersion(conv4)
# conv5_sq = get_dispersion(conv5)
# conv6_sq = get_dispersion(conv6)
# conv7_sq = get_dispersion(conv7)
# conv8_sq = get_dispersion(conv8)

# l1_sq = get_dispersion(l1)
# l2_sq = get_dispersion(l2)
# l3_sq = get_dispersion(l3)

# 0, conv1: nn.Conv2d(3, 16, (2, 2)),
# 1, conv2: nn.ReLU(),#
# 2, conv3: nn.MaxPool2d((2, 2)),
# 3, conv4: nn.Conv2d(16, 32, (2, 2)),
# 4, conv5: nn.ReLU(),#
# 5, conv6: nn.Conv2d(32, 64, (2, 2)),
# 6, conv7: nn.ReLU()#

# 7, conv8: Text embeddings
# 8, self.embedding_size=92

# 9, l1: nn.Linear(self.embedding_size, 64),
# 10, l2: nn.Tanh(),
# 11, l3: nn.Linear(64, action_space.n)

import matplotlib.pyplot as plt
# plt.close('all')
plt.figure(1)
plt.plot([dispersions[0],
          dispersions[1],
          dispersions[2],
          dispersions[3],
          dispersions[4],
          dispersions[5],
          dispersions[6],
          dispersions[7],
          dispersions[8],
          dispersions[9],
          dispersions[10],
          dispersions[11],
          ],'b', linewidth=3)
# plt.plot([#conv1_sq.item(),
#           conv2_sq.item(),
#           #conv3_sq.item(),
#           # conv4_sq.item(),
#           conv5_sq.item(),
#           # conv6_sq.item(),
#           conv7_sq.item(),
#           # l1_sq.item(),
#           l2_sq.item(),
#           l3_sq.item(),
#           action_sq.item(),
#           ])
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