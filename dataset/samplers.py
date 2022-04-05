#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import random
import time
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Sampler
from torch.utils.data.sampler import T_co


class CustomSampler(Sampler):
    def __init__(self, train_indices):
        Sampler.__init__(self, None)
        self.indices = train_indices
        self.random = True

    def update_indices(self, new_indices, is_random):
        self.indices = new_indices
        self.random = is_random

    def __iter__(self):
        if self.random:
            return (self.indices[i] for i in torch.randperm(len(self.indices)))
        else:
            return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

