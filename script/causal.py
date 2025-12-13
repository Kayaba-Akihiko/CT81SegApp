

from pathlib import Path
from typing import Union, TypeAlias

TypeTest: TypeAlias = Union[int, str]


def main():

    this_file = Path(__file__)
    this_dir = this_file.parent
    output_dir = this_dir / f'out_{this_file.stem}'
    output_dir.mkdir(exist_ok=True, parents=True)

    x = 1
    print(isinstance(x, TypeTest))
    x = 'ss'
    print(isinstance(x, TypeTest))
    x = 3.5
    print(isinstance(x, TypeTest))



if __name__ == '__main__':
    main()