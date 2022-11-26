import logging

name = 'gense'


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

from .gense import GenSE
from .synthesizer import Synthesizer
