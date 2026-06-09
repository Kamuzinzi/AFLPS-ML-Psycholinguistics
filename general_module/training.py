import sys
import types
import torch
import pandas as pd
import pickle
from IPython.display import display


def _ensure_numpy_core_alias():
    if "numpy._core" not in sys.modules:
        try:
            import numpy._core  # noqa: F401
        except ModuleNotFoundError:
            try:
                import numpy.core as np_core  # noqa: F401
            except ModuleNotFoundError:
                return False

            core_module = types.ModuleType("numpy._core")
            core_module.__path__ = getattr(np_core, "__path__", [])
            core_module.__package__ = "numpy._core"
            if hasattr(np_core, "multiarray"):
                core_module.multiarray = np_core.multiarray
            if hasattr(np_core, "_multiarray_umath"):
                core_module._multiarray_umath = np_core._multiarray_umath
            if hasattr(np_core, "umath"):
                core_module.umath = np_core.umath
            sys.modules["numpy._core"] = core_module

    if "numpy._core.multiarray" not in sys.modules:
        try:
            import numpy._core.multiarray  # noqa: F401
        except ModuleNotFoundError:
            try:
                import numpy.core.multiarray as np_multiarray  # noqa: F401
                sys.modules["numpy._core.multiarray"] = np_multiarray
            except ModuleNotFoundError:
                return False

    if "numpy._core._multiarray_umath" not in sys.modules:
        try:
            import numpy._core._multiarray_umath  # noqa: F401
        except ModuleNotFoundError:
            try:
                import numpy.core._multiarray_umath as np_multiarray_umath  # noqa: F401
                sys.modules["numpy._core._multiarray_umath"] = np_multiarray_umath
            except ModuleNotFoundError:
                pass

    return True


def extract(path):
    _ensure_numpy_core_alias()
    with open(path, 'rb') as file:
        return _NumpyCompatUnpickler(file).load()

def best_device(*args):
    assert len(args)!=0, "@Cannot be empty"

    if torch.cuda.is_available():
        device = torch.device("cuda")
        return [arg.to(device) for arg in args] if len(args) > 1 else args[0].to(device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        return [arg.to(device) for arg in args] if len(args) > 1 else args[0].to(device)

    return [arg for arg in args] if len(args) > 1 else args[0]

# create a class named model which initialise by a network
class CustomModel:
    def __init__(self, network):
        self.network = network
        self.epoch = 0
        self.ba = 0
        self.ra = 0

    def __str__(self):
        return f"Model: {self.network}, Epoch: {self.epoch}, BA: {self.ba}, RA: {self.ra}"

    def __repr__(self):
        return f"Model: {self.network}, Epoch: {self.epoch}, BA: {self.ba}, RA: {self.ra}"

    def update(self, network, epochs, ba, ra):
        assert epochs is not None, "@epochs cannot be None"
        assert ba is not None, "@ba cannot be None"
        assert ra is not None, "@ra cannot be None"
        assert network is not None, "@model cannot be None"

        if  ra > self.ra:
            self.network = network
            self.epoch = epochs
            self.ba = ba
            self.ra = ra

    def refresh(self, network, epochs, ba, ra):
        self.network = network
        self.epoch = epochs
        self.ba = ba
        self.ra = ra
    
    def override(self, ba, ra):
        self.ba = ba
        self.ra = ra

    def getOptEpoch(self):
        return self.epoch
    

def print_results(models):
    info = pd.DataFrame(columns=["Model", "Epoch", "BA", "RA"])
    for model_key in models.keys():
        # add a row to the dataframe
        info.loc[len(info)] = [model_key, models[model_key].epoch, models[model_key].ba, models[model_key].ra]
    # display info in iteractive table
    display(info)


class Checkpoint():

    def __init__(self):
        self.count=0
        self.model_state=[]
        pass

    def add(self, model_state, optimizer_state):
        self.model_state.append(model_state)
        self.count +=1

    def get(self, epoch):
        return self.model_state[epoch-1]
    
    def length(self):
        return self.count+1

# dummy data to test print_models
# models = {"O":CustomModel(None), "C":CustomModel(None), "E":CustomModel(None), "A":CustomModel(None), "N":CustomModel(None)}
# print_results(models)