#!/usr/bin/env python3
from confirmation_analysis import development_selection
import torch
if __name__=='__main__':
    torch.set_num_threads(1)
    development_selection()
