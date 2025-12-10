#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


from pathlib import Path
from typing import Union, Dict, TypeAlias, Literal, Optional
import io

import numpy as np
import numpy.typing as npt
from pptx.presentation import (
    Presentation as PPTXPresentation,
    Slides as PPTXSlides,
)
from pptx.shapes.picture import Picture as PPTXPicture
from pptx.shapes.base import BaseShape as PPTXBaseShape
from pptx.slide import (
    Slide as PPTXSlide,
    SlideShapes as PPTXSlideShapes
)
from pptx import Presentation as create_presentation
import imageio.v3 as iio

from xmodules.xutils import os_utils
from xmodules.typing import TypePathLike

TypeFitMode: TypeAlias = Literal[
    'contain', 'cover', 'match_height', 'match_width']


class ReportPPT:
    def __init__(
            self,
            template: Union[PPTXPresentation, Path],
            fit_mode: Union[TypeFitMode, Dict[str, TypeFitMode]] = 'contain',
    ):
        if isinstance(fit_mode, str):
            assert fit_mode in {'contain', 'cover', 'match_height', 'match_width'}
        elif isinstance(fit_mode, dict):
            for k, v in fit_mode.items():
                assert v in {'contain', 'cover', 'match_height', 'match_width'}
        else:
            raise ValueError(f'Invalid fit mode: {fit_mode}')
        self.fit_mode = fit_mode
        if isinstance(template, PPTXPresentation):
            self.presentation = template
        else:
            self.presentation = create_presentation(str(template))

    @classmethod
    def _iter_shapes(cls, shapes: PPTXSlideShapes):
        for sh in shapes:
            yield sh
            child = getattr(sh, 'shapes', None)
            if child is not None:
                yield from cls._iter_shapes(child)

    @staticmethod
    def _send_to_back(shape: PPTXBaseShape):
        el = shape._element
        parent = el.getparent()
        parent.remove(el)
        parent.insert(2, el)

    @classmethod
    def _fit_image_into_slide(
            cls,
            slide: PPTXSlide,
            image: npt.NDArray[np.uint8],
            bbox: tuple[float, float, float, float],
            fit_mode: TypeFitMode,
    ):
        left, top, w_box, h_box = bbox
        ih, iw = image.shape[:2]

        if iw == 0 or ih == 0:
            raise ValueError(f'Invalid image size: {image.shape}')

        scale_w, scale_h = 1, 1
        if fit_mode == 'contain':
            scale_w = min(w_box / iw, h_box / ih)
            scale_h = scale_w
        elif fit_mode == 'cover':
            scale_w = max(w_box / iw, h_box / ih)
            scale_h = scale_w
        elif fit_mode == 'match_height':
            scale_w = h_box / ih
        elif fit_mode == 'match_width':
            scale_h = w_box / iw
        else:
            raise ValueError(f'Invalid fit mode: {fit_mode}')

        width_emu, height_emu = w_box * scale_w, h_box * scale_h
        x = left + (w_box - int(width_emu)) // 2
        y = top + (h_box - int(height_emu)) // 2
        pic = slide.shapes.add_picture(cls._stream_image(image), x, y)
        pic.width = int(width_emu)
        pic.height = int(height_emu)
        pic.left, pic.top = x, y
        return pic


    @staticmethod
    def _stream_image(
            image: npt.NDArray[np.uint8], extension: Optional[str] = None
    ) -> io.BytesIO:
        image_stream = io.BytesIO()
        iio.imwrite(image_stream, image, extension=extension)
        image_stream.seek(0)
        return image_stream