#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

import tqdm as tq
from xmodules.xutils import dist_utils

class tqdm(tq.tqdm):

    def __init__(
            self, iterable=None, desc=None, total=None, leave=True, file=None,
            ncols=None, mininterval=0.1, maxinterval=10.0, miniters=None,
            ascii=None, disable=False, unit='it', unit_scale=False,
            dynamic_ncols=False, smoothing=0.3, bar_format=None, initial=0,
            position=None, postfix=None, unit_divisor=1000, write_bytes=False,
            lock_args=None, nrows=None, colour=None, delay=0, gui=False,
            **kwargs):
        desc = desc or ''
        if dist_utils.is_initialized():
            desc = f"[RANK {dist_utils.get_global_rank()}] {desc}"
        super().__init__(
            iterable=iterable, desc=desc, total=total, leave=leave, file=file,
            ncols=ncols, mininterval=mininterval, maxinterval=maxinterval,
            miniters=miniters, ascii=ascii, disable=disable, unit=unit,
            unit_scale=unit_scale, dynamic_ncols=dynamic_ncols,
            smoothing=smoothing, bar_format=bar_format, initial=initial,
            position=position, postfix=postfix, unit_divisor=unit_divisor,
            write_bytes=write_bytes,lock_args=lock_args, nrows=nrows,
            colour=colour, delay=delay, gui=gui,
            **kwargs)