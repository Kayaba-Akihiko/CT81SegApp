

from pathlib import Path
from typing import Union, TypeAlias, Literal




def main():

    this_file = Path(__file__)
    this_dir = this_file.parent
    output_dir = this_dir / f'out_{this_file.stem}'
    output_dir.mkdir(exist_ok=True, parents=True)



if __name__ == '__main__':
    main()