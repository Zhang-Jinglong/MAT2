from ._settings import set_seed
from .train import BuildMAT2

print('Welcome to MAT2!')
set_seed(5)

__all__ = ["set_seed", "BuildMAT2"]
