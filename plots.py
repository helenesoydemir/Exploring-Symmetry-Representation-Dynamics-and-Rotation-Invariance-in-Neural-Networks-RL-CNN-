#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 22:59:31 2025

@author: adan
"""

import numpy as np
import matplotlib.pyplot as plt 
tmp1 = np.loadtxt("/home/adan/project_layers/rl-starter-files/scripts/storage/A2C.csv",delimiter=',')
tmp2 = np.loadtxt("/home/adan/project_layers/rl-starter-files/scripts/storage/PPO.csv",delimiter=',')
plt.figure(1)
plt.plot(tmp1[:,1], tmp1[:,2],'b')
plt.plot(tmp1[:,1], tmp2[:,2],'r')

plt.figure(2)
plt.plot(tmp1[-100:,1], tmp1[-100:,2],'b')
plt.plot(tmp1[-100:,1], tmp2[-100:,2],'r')