import os
from typing import Dict, Tuple, List
from typing import Optional, Callable, Any

import torch
from torchvision.datasets.folder import default_loader, ImageFolder, make_dataset
import torchvision.transforms.functional as F

from utils import is_rank_zero, print_once


import numpy as np
from PIL import Image


def test_loader(path):
    with open(path, 'rb') as f:
        img = Image.fromarray(
            np.stack(
                (np.uint8(np.asarray(Image.open(f)) * 255 // 65535),) * 3, axis=-
                1))
    return img

class ImageFolderWithFilenames(ImageFolder):
    def __init__(self, root: str, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 loader: Callable[[str], Any] = test_loader,
                 is_valid_file: Optional[Callable[[str], bool]] = None):
        super().__init__(root=root, transform=transform, target_transform=target_transform,
                         loader=loader, is_valid_file=is_valid_file)

    @staticmethod
    def make_dataset(
            directory: str,
            class_to_idx: Dict[str, int],
            extensions: Optional[Tuple[str, ...]] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError(
                "The class_to_idx parameter cannot be None."
            )
        cache_path = os.path.join(directory, 'cache.pt')
        try:
            dataset = torch.load(cache_path, map_location='cpu')
            print_once(f'Loading dataset from cache in {directory}')
        except FileNotFoundError:
            print_once(f'Creating dataset and saving to cache in {directory}')
            dataset = make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)
            if is_rank_zero():
                torch.save(dataset, cache_path)
        except EOFError:
            print_once(f'Error loading cache from {directory},'
                       f' likely because dataset is small and read/write were attempted concurrently. '
                       f'Proceeding by remaking dataset in-memory.')
            dataset = make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)
        print_once(f'{len(dataset)} images in dataset')
        return dataset

    def __getitem__(self, i):
        x, y = super().__getitem__(i)
        return x, {'labels': y, 'filenames': self.imgs[i][0]}

class PadToSquare:
	"""Add padding to the image until it is a square"""

	def __init__(self, fill, padding_mode):
		self.fill = fill
		self.padding_mode = padding_mode

	def __call__(self, img):
		if img.size[0] > img.size[1]:
			return F.pad(img, [0, img.size[0] - img.size[1], 0, 0])
		else:
			return F.pad(img, [img.size[1] - img.size[0], 0, 0, 0])