import os
import sys


def import_models():
    sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/models/')
