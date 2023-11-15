import torch
import os
import numpy as np
import h5py
import copy

from FLAlgorithms.users.useravg import UserAVG
from FLAlgorithms.servers.serverbase import Server

class IndServer:
    def __init__(self, args):