

from pathlib import Path
from typing import Union, TypeAlias
import numpy as np
import torch
import torch.nn.functional as F



def main():

    this_file = Path(__file__)
    this_dir = this_file.parent
    output_dir = this_dir / f'out_{this_file.stem}'
    output_dir.mkdir(exist_ok=True, parents=True)

    n_classes = 5
    labelmap = np.asarray([[0, 0, 1], [0, 2, 0]], dtype=np.int64)
    print(labelmap.shape) # (2, 3)
    one_hot = np.eye(n_classes)[labelmap]
    print(one_hot.shape)
    print((labelmap > 0).all())

    labelmap = torch.from_numpy(labelmap)
    one_hot = F.one_hot(labelmap, num_classes=n_classes)
    print(one_hot.shape)

    print((labelmap > 0).all())


if __name__ == '__main__':
    main()