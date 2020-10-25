import unittest
import pytest

from models.utils.random_state import RandomState
from models.utils.inspect_model import count_parameters

import random
import numpy as np
import torch


@pytest.mark.dependency()
def test_set_random_state():
    random_state = RandomState(137)
    random_state.fix()

    assert np.random.get_state()[1][0] == 137
