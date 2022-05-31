import csv
import argparse
import random
from pathlib import Path
from typing import DefaultDict, Union, Optional

import torch
import transformers as hf
import datasets as hfd
from tqdm import tqdm

from utils import Silence, HANS_subcases

rte_dataset = hfd.load_dataset('super_glue', 'rte')
print(rte_dataset)
samples = rte_dataset['train'][:3]
print(samples)