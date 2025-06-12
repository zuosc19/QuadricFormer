""" Cosine Scheduler

Cosine LR schedule with warmup, cycle/restarts, noise, k-decay.

Hacked together by / Copyright 2021 Ross Wightman
"""
import logging
import math
import numpy as np
import torch
from typing import List


class OffsetScheduler():

    def __init__(
            self,
            t_total: int,
            t_max_bound: int,
            offset_min: float = 1.,
            offset_max: float = 6.,

    ) -> None:

        self.t_total = t_total
        self.t_max_bound = t_max_bound
        self.offset_min = offset_min
        self.offset_max = offset_max

    def get_offset(self, t: int) -> float:
        if t < self.t_max_bound:
            offset = self.offset_min + 0.5 * (self.offset_max - self.offset_min) * (1 - math.cos(math.pi * t / self.t_max_bound))
        else:
            offset = self.offset_max

        return offset
