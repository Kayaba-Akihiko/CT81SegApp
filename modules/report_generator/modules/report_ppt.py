#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


from pathlib import Path
from typing import Union, Dict, TypeAlias, Literal, Optional, Tuple, Self
import io
from dataclasses import dataclass
import logging

import numpy as np
import numpy.typing as npt
from pptx.presentation import (
    Presentation as PPTXPresentation,
    Slides as PPTXSlides,
)
from pptx.shapes.picture import Picture as PPTXPicture
from pptx.util import Length as PPTXLength
from pptx.shapes.base import BaseShape as PPTXBaseShape
from pptx.text.text import _Run as PPTXRun
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

_logger = logging.getLogger(__name__)

@dataclass
class FillImageData:
    name: Optional[str] = None
    image: Optional[npt.NDArray[np.uint8]] = None
    fill_mode: TypeFitMode = 'contain'
    send_to_back: bool = False


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
            self.presentation = self.clone_presentation(template)
        else:
            self.presentation = create_presentation(str(template))

    def copy(self) -> Self:
        return ReportPPT(self.presentation, fit_mode=self.fit_mode)

    def collect_shape_sizes(
            self, dpi: int = 96
    ) -> Dict[str, Tuple[int, int]]:
        sizes: Dict[str, Tuple[int, int]] = {}
        for slide in self.presentation.slides:
            for shape in self._iter_shapes(slide.shapes):
                name = getattr(shape, 'name', '') or ''
                if 'IMG:' not in name:
                    continue

                key = name.split('IMG:', 1)[1].strip()
                if not key:
                    continue
                w_px = self._emu_to_px(shape.width, dpi)
                h_px = self._emu_to_px(shape.height, dpi)
                sizes[key] = (w_px, h_px)

        return sizes

    def fill_images(
            self,
            images: Dict[str, FillImageData],
    ) -> Dict[str, PPTXPicture]:
        res: Dict[str, PPTXPicture] = {}
        finished_key = set()
        for slide in self.presentation.slides:
            for shape in self._iter_shapes(slide.shapes):
                name = getattr(shape, 'name', '') or ''
                if 'IMG:' not in name:
                    continue

                key = name.split('IMG:', 1)[1].strip()
                if not key or key not in images:
                    continue
                fill_image_data = images[key]
                if fill_image_data.image is None:
                    raise ValueError(f'Image not found: {key}')
                pic = self._fit_image_into_slide(
                    slide, fill_image_data.image,
                    (shape.left, shape.top, shape.width, shape.height),
                    fill_image_data.fill_mode
                )

                if fill_image_data.send_to_back:
                    el = pic._element
                    parent = el.getparent()
                    parent.remove(el)
                    parent.insert(2, el)

                res[key] = pic
                finished_key.add(key)
        unknown_filling_keys = set(images.keys()) - finished_key
        if unknown_filling_keys:
            _logger.warning(
                f'Unknown filling keys: {unknown_filling_keys}')
        return res

    def fill_texts(self, mapping: dict[str, str]) -> Dict[str, PPTXRun]:
        tokens = {f'{{{{{k}}}}}': str(v) for k, v in mapping.items()}

        finished_key = set()
        res: Dict[str, PPTXRun] = {}
        for slide in self.presentation.slides:
            for shape in self._iter_shapes(slide.shapes):
                if getattr(shape, 'has_text_frame', False) and shape.has_text_frame:
                    for p in shape.text_frame.paragraphs:
                        for r in p.runs:
                            text = r.text
                            for tk, val in tokens.items():
                                if tk in text:
                                    text = text.replace(tk, val)
                                    res[tk] = r
                                    finished_key.add(tk)
                            if text != r.text:
                                r.text = text

                if getattr(shape, 'has_table', False) and shape.has_table:
                    tbl = shape.table
                    for r_i in range(len(tbl.rows)):
                        for c_i in range(len(tbl.columns)):
                            cell = tbl.cell(r_i, c_i)
                            tf = getattr(cell, 'text_frame', None)
                            if tf is None:
                                continue
                            for p in tf.paragraphs:
                                for r in p.runs:
                                    text = r.text
                                    for tk, val in tokens.items():
                                        if tk in text:
                                            text = text.replace(tk, val)
                                            res[tk] = r
                                            finished_key.add(tk)
                                    if text != r.text:
                                        r.text = text
        unknown_filling_keys = set(mapping.keys()) - finished_key
        if unknown_filling_keys:
            _logger.warning(
                f'Unknown filling keys: {unknown_filling_keys}')
        return res

    @classmethod
    def _iter_shapes(cls, shapes: PPTXSlideShapes):
        for sh in shapes:
            yield sh
            child = getattr(sh, 'shapes', None)
            if child is not None:
                yield from cls._iter_shapes(child)

    def save(self, save_path: Path) -> None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.presentation.save(str(save_path))

    @classmethod
    def _fit_image_into_slide(
            cls,
            slide: PPTXSlide,
            image: npt.NDArray[np.uint8],
            bbox: tuple[PPTXLength, PPTXLength, PPTXLength, PPTXLength],
            fit_mode: TypeFitMode,
    ):
        left, top, w_box, h_box = bbox
        ih, iw = image.shape[:2]

        if iw == 0 or ih == 0:
            raise ValueError(f'Invalid image size: {image.shape}')

        if fit_mode == 'contain':
            scale = min(w_box / iw, h_box / ih)
            width_emu, height_emu = iw * scale, ih * scale
        elif fit_mode == 'cover':
            scale = max(w_box / iw, h_box / ih)
            width_emu, height_emu = iw * scale, ih * scale
        elif fit_mode == 'match_height':
            scale = h_box / ih
            width_emu, height_emu = iw * scale, h_box
        elif fit_mode == 'match_width':
            scale = w_box / iw
            width_emu, height_emu = w_box, ih * scale
        else:
            raise ValueError(f'Invalid fit mode: {fit_mode}')

        x = left + (w_box - int(width_emu)) // 2
        y = top + (h_box - int(height_emu)) // 2
        pic = slide.shapes.add_picture(cls._stream_image(image, extension='.png'), x, y)
        pic.width = int(width_emu)
        pic.height = int(height_emu)
        pic.left, pic.top = x, y
        return pic

    @staticmethod
    def clone_presentation(presentation: PPTXPresentation) -> PPTXPresentation:
        buffer = io.BytesIO()
        presentation.save(buffer)
        buffer.seek(0)
        return create_presentation(buffer)


    @staticmethod
    def _emu_to_px(emu: int, dpi: int = 96) -> int:
        # EMU_PER_INCH = 914400
        return int(round(emu / 914400 * dpi))

    @staticmethod
    def _stream_image(
            image: npt.NDArray[np.uint8], extension: Optional[str] = None
    ) -> io.BytesIO:
        image_stream = io.BytesIO()
        iio.imwrite(image_stream, image, extension=extension)
        image_stream.seek(0)
        return image_stream