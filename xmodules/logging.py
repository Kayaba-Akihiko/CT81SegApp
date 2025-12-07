#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

from functools import lru_cache
import io
from logging import getLoggerClass

from .xutils import dist_utils

class Logger(getLoggerClass()):
    def _log(
            self, level, msg, args,
            exc_info=None, extra=None, stack_info=False, stacklevel=1):
        if dist_utils.is_initialized():
            msg = f"[RANK{dist_utils.get_global_rank()}] {msg}"
        super(Logger, self)._log(
            level=level, msg=msg, args=args,exc_info=exc_info,
            extra=extra, stack_info=stack_info, stacklevel=stacklevel)

    @lru_cache(1)
    def info_once(self, msg):
        self.info(msg)


class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ''
    def __init__(self, logger: Logger, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logger.level
    def write(self,buf):
        self.buf = buf.strip('\r\n\t ')
    def flush(self):
        self.logger.log(self.level, self.buf)