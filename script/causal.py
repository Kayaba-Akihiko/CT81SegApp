

from pathlib import Path
import numpy as np
import math


def main():

    this_file = Path(__file__)
    this_dir = this_file.parent
    output_dir = this_dir / f'out_{this_file.stem}'
    output_dir.mkdir(exist_ok=True, parents=True)

    x = np.asarray([1, 2, 3, None], dtype=np.float64)
    x = float(x[-1])
    print(math.isnan(x), np.isnan(x))


if __name__ == '__main__':
    main()