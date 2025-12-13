#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


from typing import Sequence, Any, Optional, Union, Tuple, List
from pathlib import Path
import logging

import pydicom


import numpy as np
import numpy.typing as npt

from . import os_utils, multiprocessing_utils
from ..protocol import TypePathLike

_logger = logging.getLogger(__name__)


class DicomUtils:

    @staticmethod
    def read_dicom_file(
            read_path: TypePathLike, required_tag: Optional[str | Sequence[str]] = None,
    ) -> tuple[
             npt.NDArray,
             npt.NDArray[np.double],
             Union[npt.NDArray[np.double], None],
         ] | tuple[
             npt.NDArray,
             npt.NDArray[np.double],
             Union[npt.NDArray[np.double], None],
             Any,
         ]:
        dcfile = pydicom.dcmread(read_path)
        spacing = np.asarray(dcfile.PixelSpacing, dtype=np.float64)

        position = getattr(dcfile, "ImagePositionPatient", None)
        if position is not None:
            position = np.asarray(position, dtype=np.float64)

        img2d = pydicom.pixels.apply_modality_lut(dcfile.pixel_array, dcfile)

        ret = (img2d, spacing, position)

        if required_tag is None:
            return ret

        if isinstance(required_tag, str):
            required_tag = [required_tag]

        if len(required_tag) == 0:
            return ret

        tag_res = {}
        for tag in required_tag:
            if hasattr(dcfile, tag):
                tag_res[tag] = getattr(dcfile, tag)
            else:
                tag_res[tag] = None
        if len(tag_res) == 1:
            tag_res = tag_res[required_tag[0]]
        return ret[0], ret[1], ret[2], tag_res

    @classmethod
    def read_dicom_folder(
            cls,
            read_path: TypePathLike,
            name_regex='.*\\.dcm$',
            required_tag: Optional[str | Sequence[str]] = None,
            n_workers: int = 0,
            progress_bar=True,
            progress_desc='',
            progress_mininterval=1.0,
            progress_maxinterval=10.0,
    ) -> tuple[
             npt.NDArray,
             npt.NDArray[np.double],
             npt.NDArray[np.double],
         ] | tuple[
             npt.NDArray,
             npt.NDArray[np.double],
             npt.NDArray[np.double],
             Any,
         ]:
        "https://pydicom.github.io/pydicom/stable/auto_examples/image_processing/reslice.html"

        files, file_paths = cls._dcmread_folder(
            read_path=read_path,
            name_regex=name_regex,
            n_workers=n_workers,
            progress_bar=progress_bar,
            progress_desc=progress_desc,
            progress_mininterval=progress_mininterval,
            progress_maxinterval=progress_maxinterval,
        )
        if len(files) == 0:
            raise ValueError(
                f"No dicom files found with name pattern {name_regex} in {read_path} .")
        n_total = len(files)
        # skip files with no SliceLocation (eg scout views)
        slices = []
        skipcount = 0
        for f, path in zip(files, file_paths):
            if hasattr(f, "SliceLocation") and f.SliceLocation is not None:
                slices.append(f)
            else:
                _logger.info(f'Skipp slice without slice location {path}.')
                skipcount = skipcount + 1
        if skipcount > 0:
            _logger.info(
                f'Skipped {skipcount} from total {n_total} dicom files in {read_path}.'
            )
        if len(slices) == 0:
            raise RuntimeError(
                f'No valid slices found from {len(files)} files with name pattern {name_regex} in {read_path} .')

        # ensure they are in the correct order
        slices = sorted(slices, key=lambda s: s.SliceLocation)

        ps, z_pos = zip(*[(s.PixelSpacing, s.ImagePositionPatient[2]) for s in slices])
        ps = np.asarray(ps, dtype=np.float64)  # (N, 2)
        z_pos = np.asarray(z_pos, dtype=np.float64)  # (N, )
        if not np.allclose(ps[0], ps[1:]):
            raise ValueError("Slices have different PixelSpacing")
        ps = ps[0]
        z_diff = np.abs(z_pos[1:] - z_pos[:-1])
        if not np.allclose(z_diff[0], z_diff[1:]):
            raise ValueError("Slices have different z position")
        ss = z_diff[0]

        # create 3D array
        img_shape = list(slices[0].pixel_array.shape)
        img3d = np.empty((len(slices), *img_shape))

        # (z, y, x)
        # fill 3D array with the images from the files
        for i, s in enumerate(slices):
            img2d = pydicom.pixels.apply_modality_lut(s.pixel_array, s)
            # img2d = s.pixel_array
            img3d[i, :, :] = img2d
        # (x, y, z)
        spacing = np.asarray(
            (ps[0], ps[1], ss), dtype=np.float64)
        # (x, y ,z)
        position = np.asarray(
            slices[0].ImagePositionPatient, dtype=np.float64)
        # (x, y ,z) -> (z, y ,x)
        spacing = spacing[::-1].copy()
        # (x, y ,z) -> (z, y ,x)
        position = position[::-1].copy()


        ret = (
            img3d, spacing, position
        )

        if required_tag is None:
            return ret

        if isinstance(required_tag, str):
            required_tag = [required_tag]

        if len(required_tag) == 0:
            return ret

        tag_res = {}
        for tag in required_tag:
            if hasattr(slices[0], tag):
                tag_res[tag] = getattr(slices[0], tag).value
            else:
                tag_res[tag] = None
        if len(tag_res) == 1:
            tag_res = tag_res[required_tag[0]]

        return ret[0], ret[1], ret[2], tag_res

    @classmethod
    def _dcmread_folder(
            cls,
            read_path: TypePathLike,
            name_regex='.*\\.dcm$',
            n_workers: int = 0,
            progress_bar: bool = True,
            progress_desc='',
            progress_mininterval=1.,
            progress_maxinterval=10.0,
    ) -> Tuple[List[pydicom.FileDataset], List[TypePathLike]]:

        n_workers = min(n_workers, os_utils.get_max_n_worker())

        slice_entries = []
        for slice_entry in os_utils.scan_dirs_for_file(
                read_path, name_regex=name_regex):
            slice_entries.append(slice_entry)

        args = [(slice_entry.path, True) for slice_entry in slice_entries]

        iterator = multiprocessing_utils.run_jobs(
            args=args,
            func=cls._dcmread,
            n_workers=n_workers,
            progress_bar=progress_bar,
            progress_desc=progress_desc,
            progress_mininterval=progress_mininterval,
            progress_maxinterval=progress_maxinterval,
            total=len(args),
        )
        files = []
        file_paths = []
        n_total, n_invalid = 0, 0
        for file, (file_path, _) in zip(iterator, args):
            n_total += 1
            if file is None:
                n_invalid += 1
                continue
            files.append(file)
            file_paths.append(file_path)
        if n_invalid > 0:
            _logger.warning(
                f'Found {n_invalid} invalid from total {n_total} files with name pattern {name_regex} in {read_path}')
        return files, file_paths

    @staticmethod
    def _dcmread(
            _read_path: TypePathLike,
            allow_invalid: bool = False,
    ) -> Union[pydicom.FileDataset, None]:
        try:
            _file = pydicom.dcmread(_read_path)
        except pydicom.errors.InvalidDicomError as e:
            if not allow_invalid:
                raise e
            _logger.warning(f'Error reading {_read_path}: {e}')
            _file = None
        return _file


read_dicom_file = DicomUtils.read_dicom_file
read_dicom_folder = DicomUtils.read_dicom_folder