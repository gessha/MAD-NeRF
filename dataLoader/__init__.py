from .llff import LLFFDataset
from .blender import BlenderDataset
from .nsvf import NSVF
from .tankstemple import TanksTempleDataset, TanksTempleDataset_AUDIO
from .your_own_data import YourOwnDataset



dataset_dict = {'blender': BlenderDataset,
               'llff':LLFFDataset,
               'tankstemple':TanksTempleDataset,
               'ad_nerf': TanksTempleDataset_AUDIO,
               'nsvf':NSVF,
                'own_data':YourOwnDataset}