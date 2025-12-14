

from pathlib import Path
import numpy as np




def main():

    this_file = Path(__file__)
    this_dir = this_file.parent
    output_dir = this_dir / f'out_{this_file.stem}'
    output_dir.mkdir(exist_ok=True, parents=True)


    x = np.array([True, True, False])
    y = x.sum()
    print(y, y.dtype)



if __name__ == '__main__':
    main()