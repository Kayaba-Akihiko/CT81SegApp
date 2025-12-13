#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


from pathlib import Path
from typing import Union, Dict, TypeAlias, Literal, Optional, Tuple, Self, IO
import io
from dataclasses import dataclass
import logging
import tempfile
import subprocess
import shutil

import numpy as np
import numpy.typing as npt
from pptx.presentation import (
    Presentation as PPTXPresentation,
)
from pptx.shapes.picture import Picture as PPTXPicture
from pptx.util import Length as PPTXLength
from pptx.text.text import _Run as PPTXRun
from pptx.slide import (
    Slide as PPTXSlide,
    SlideShapes as PPTXSlideShapes
)
from pptx import Presentation as create_presentation
import imageio.v3 as iio

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
                            for (tk, val), check_key in zip(tokens.items(), mapping):
                                if tk in text:
                                    text = text.replace(tk, val)
                                    res[check_key] = r
                                    finished_key.add(check_key)
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
                                    for (tk, val), check_key in zip(tokens.items(), mapping):
                                        if tk in text:
                                            text = text.replace(tk, val)
                                            res[check_key] = r
                                            finished_key.add(check_key)
                                    if text != r.text:
                                        r.text = text
        unknown_filling_keys = set(mapping.keys()) - finished_key
        if unknown_filling_keys:
            _logger.warning(
                f'Unknown filling keys: {unknown_filling_keys}')
        return res

    def save(
            self,
            pptx_save_path: Optional[Union[Path, IO]] = None,
            pdf_save_path: Optional[Path] = None,
            image_save_path: Optional[Union[Path, IO]] = None,
            dpi=200,  # Used for when saving as image
    ):
        if not pptx_save_path and not pdf_save_path and not image_save_path:
            raise ValueError('One path must be specified.')

        if pptx_save_path is not None:
            # If caller passes an IO, python-pptx can save to it directly.
            if isinstance(pptx_save_path, Path):
                pptx_save_path.parent.mkdir(parents=True, exist_ok=True)
                self.presentation.save(str(pptx_save_path))
            else:
                self.presentation.save(pptx_save_path)

        if pdf_save_path or image_save_path is not None:
            with tempfile.TemporaryDirectory(prefix="lo_") as temp_dir_name:
                temp_dir = Path(temp_dir_name)
                if pptx_save_path is not None:
                    temp_pptx_path = pptx_save_path
                else:
                    temp_pptx_path = temp_dir / "temp_ppt.pptx"
                    self.presentation.save(str(temp_pptx_path))

                # Use an isolated LO profile to avoid lock/hang issues
                profile_dir = temp_dir / "profile"
                profile_dir.mkdir(parents=True, exist_ok=True)

                # Convert PPTX -> PDF
                cmd = [
                    "soffice",  # prefer soffice over libreoffice
                    "--headless",
                    "--nologo",
                    "--nolockcheck",
                    "--nodefault",
                    "--norestore",
                    f"-env:UserInstallation=file://{profile_dir}",
                    "--convert-to", "pdf:impress_pdf_Export",
                    "--outdir", str(temp_dir),
                    str(temp_pptx_path),  # ✅ INPUT IS PPTX
                ]
                r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if r.returncode != 0:
                    raise RuntimeError(f"LibreOffice failed (code {r.returncode}).\nSTDERR:\n{r.stderr}\nSTDOUT:\n{r.stdout}")

                temp_pdf_path = temp_dir / f"{temp_pptx_path.stem}.pdf"
                if not temp_pdf_path.exists():
                    raise RuntimeError(f"LibreOffice reported success but PDF not found: {temp_pdf_path}")

                if pdf_save_path is not None:
                    pdf_save_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(temp_pdf_path, pdf_save_path)

                if image_save_path is not None:
                    # Requires poppler installed for pdf2image on Linux:
                    # sudo apt-get install -y poppler-utils
                    from pdf2image import convert_from_path

                    pages = convert_from_path(str(temp_pdf_path), dpi=dpi)
                    # If image_save_path is a Path, save to it; if it's a file-like object, PIL can write to it too.
                    pages[0].save(str(image_save_path) if isinstance(image_save_path, Path) else image_save_path)

    @classmethod
    def _iter_shapes(cls, shapes: PPTXSlideShapes):
        for sh in shapes:
            yield sh
            child = getattr(sh, 'shapes', None)
            if child is not None:
                yield from cls._iter_shapes(child)

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