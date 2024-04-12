"""
noisebase
---------

Datasets and benchmarks for neural Monte Carlo denoising.

This file serves as the main entry point for Noisebase. It registers all 
Noisebase config files in your Hydra path and, in addition, provides a 
`Noisebase` function should you wish to keep your use of Hydra to a minimum.
"""

__version__ = '1.1'

import os

from hydra.plugins.search_path_plugin import SearchPathPlugin
from hydra.core.plugins import Plugins

class NoisebaseSearchPathPlugin(SearchPathPlugin):
    """Adds Noisebase config files to the Hydra path.
    """
    def manipulate_search_path(self, search_path):
        """Computes search path.

        We want Noisebase to work in various installations. We compute the path
        of the current file (src/noisebase/__init__.py) and add the conf folder
        relatively to it to the Hydra path.
        """
        folder = os.path.normpath(os.path.join(os.path.dirname(__file__), './conf'))
        search_path.append(
            provider="noisebase", path=f'file://{folder}'
        )

Plugins.instance().register(NoisebaseSearchPathPlugin)



def Noisebase(config, options = {}):
    """Loads a Noisebase dataset using Hydra internally. (Use this if you don't
    want to use Hydra otherwise.)

    Args:
        config (str): name of the dataset definition file (e.g. 'sampleset_v1')
        options (dict): loader options (e.g. {'batch_size': 8}) (see wiki for available options)

    Returns:
        loader (object): For the selected framework 
            (e.g. torch.utils.data.DataLoader for Pytorch or pl.LightningDataModule for Lightning)
    """

    from hydra import initialize_config_dir, compose
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    
    try:
        with initialize_config_dir(config_dir=os.path.join(os.path.dirname(__file__), './conf/noisebase'), version_base='1.2'):
            cfg = compose(config_name=config)
            cfg = OmegaConf.merge(cfg, options)
            return instantiate(cfg)
    except ValueError as hydra_error:
        if hydra_error.args[0] == 'GlobalHydra is already initialized, call GlobalHydra.instance().clear() if you want to re-initialize':
            raise ValueError('Are you running your project with Hydra? Don\'t call `noisebase.Noisebase`; instead, put Noisebase configs in your Defaults List.') from hydra_error
        else:
            raise hydra_error