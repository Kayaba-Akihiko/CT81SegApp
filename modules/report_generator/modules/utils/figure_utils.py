#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

from typing import Union, List, Tuple, Optional, Dict
from dataclasses import dataclass
import io

import numpy as np
import numpy.typing as npt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import imageio.v3 as iio

@dataclass
class BoxDrawingData:
    name: Optional[float] = None
    target: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    color: Optional[Tuple[float, float, float]] = None
    visible: bool = True

class FigureUtils:

    @staticmethod
    def compute_lim(
            array: Union[
                Tuple[Union[float, int]],
                List[Union[float, int]],
                npt.NDArray[Union[np.floating, np.integer]],
            ],
            pad_ratio: float = 0
    ):
        if pad_ratio < 0:
            raise ValueError(f'pad_ratio must be non-negative: {pad_ratio}')
        array = np.asarray(array).flatten()
        valid = np.isfinite(array)
        if not valid.any():
            return None, None
        valid_array = array[valid]
        min_val, max_val = valid_array.min(), valid_array.max()
        if pad_ratio == 0:
            return min_val, max_val
        span = max_val - min_val
        pad = span * float(pad_ratio)
        return min_val - pad, max_val + pad


    @classmethod
    def draw_hu_boxes(
            cls,
            boxes: List[BoxDrawingData],
            canvas_size_px: Optional[Tuple[int, int]]  = None,
            xlim: Optional[Tuple[Union[float, None], Union[float, None]]] = None,
            s_star: float = 400.0,
            box_height_px: int = 14,
            row_gap_px: int = 8,
            dpi=96
    ):
        if canvas_size_px is None:
            y_pos = np.arange(len(boxes))[::-1]
            fig_h = max(3.0, 1.1 * len(boxes) + 0.8)
            fig, ax = plt.subplots(figsize=(9, fig_h), dpi=150)
            pixel_mode = False
        else:
            w_px, h_px = int(canvas_size_px[0]), int(canvas_size_px[1])
            fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
            ax = fig.add_axes([0, 0, 1, 1])
            pixel_mode = True

        # ----------------------------
        # Helpers
        # ----------------------------
        def _is_valid_number(val) -> bool:
            try:
                return val is not None and np.isfinite(float(val))
            except Exception:
                return False


        if pixel_mode:
            pad_px = 6.0
            n = len(boxes)
            avail = max(1.0, h_px - 2 * pad_px)
            total_needed = n * (box_height_px + row_gap_px) - row_gap_px

            if total_needed > avail:
                scale = avail / max(1.0, total_needed)
                box_h = max(6.0, box_height_px * scale)
                gap_h = row_gap_px * scale
            else:
                box_h = float(box_height_px)
                gap_h = float(row_gap_px)

            ax.set_ylim(0, h_px)
            ax.invert_yaxis()

            n = len(boxes)
            box_h = max(6, int(round(box_h)))
            gap_h = int(round(gap_h))

            avail_int = h_px - 2 * int(pad_px)
            used_int = n * box_h + (n - 1) * gap_h
            top_margin = int(pad_px) + max(0, (avail_int - used_int) // 2)
            y_centers = [top_margin + i * (box_h + gap_h) + box_h // 2 for i in range(n)]

            ax.set_ylim(-1, h_px + 1)
            ax.invert_yaxis()
        else:
            y_pos = np.arange(len(boxes))[::-1]
            box_h = 0.7
            y_centers = None

        # ---- X limits (data-driven, includes boxes + stars) ----

        if xlim is None:
            xmin, xmax = None, None
        else:
            xmin, xmax = xlim
        if xmin is None or xmax is None:
            xs = []
            for b in boxes:
                if not b.visible:
                    continue
                xs.extend([b.mean, b.mean + b.std, b.mean - b.std, b.target])
            if xs:
                _xmin, _xmax = cls.compute_lim(xs, pad_ratio=0.05)
                if xmin is None:
                    xmin = _xmin
                if xmax is None:
                    xmax = _xmax
                del _xmin, _xmax
            del xs
        if xmin is not None and xmax is not None:
            ax.set_xlim(xmin, xmax)
            ax.set_xticks(np.linspace(xmin, xmax, 6))
        elif xmin is not None:
            ax.set_xlim(left=xmin)
        elif xmax is not None:
            ax.set_xlim(right=xmax)

        ax.set_xticklabels([])

        # ----------------------------
        # Draw boxes / stars
        # ----------------------------
        for i, box_data in enumerate(boxes):
            if not box_data.visible:
                continue
            color = box_data.color
            target = box_data.target
            mean, std = box_data.mean, box_data.std

            draw_box = _is_valid_number(mean) and _is_valid_number(std) and float(std) > 0.0
            draw_star = _is_valid_number(target)

            if draw_box:
                x0 = mean - std
                width = 2 * std
                if pixel_mode:
                    y_c = y_centers[i]
                    y0 = int(y_c - box_h // 2)
                    rect = Rectangle(
                        (x0, y0), width, int(box_h),
                        facecolor=color,
                        edgecolor='black', linewidth=0.1, antialiased=False)

                    ax.add_patch(rect)
                    # half = int(round(box_h * 0.48))
                    half = max(1, int(round(box_h * 0.48)))
                    ax.plot(
                        [mean, mean],
                        [y_c - half, y_c + half],
                        color='black',
                        linewidth=1.2, antialiased=False)
                else:
                    y = y_pos[i]
                    rect = Rectangle(
                        (x0, y - 0.35), width, 0.7,
                        facecolor=color, edgecolor="black",
                        alpha=0.8, linewidth=0.1
                    )
                    ax.add_patch(rect)
                    ax.plot(
                        [mean, mean],
                        [y - 0.38, y + 0.38],
                        color="black", linewidth=1.2
                    )

            if draw_star:
                if pixel_mode:
                    y_c = y_centers[i]
                    ax.scatter(
                        [target], [y_c],
                        marker='*', s=s_star,
                        facecolor='black', edgecolors='white',
                        linewidths=0.7, zorder=5, clip_on=False)
                else:
                    ax.scatter(
                        [target], [y_pos[i]],
                        marker='*', s=s_star,
                        facecolor='black', edgecolors='white',
                        linewidths=0.7, zorder=5)

        # ----------------------------
        # Draw "invisible" rows as horizontal lines across current x-range
        # ----------------------------
        xmin_line, xmax_line = ax.get_xlim()
        for i, box_data in enumerate(boxes):
            if box_data.visible:
                continue
            if pixel_mode:
                y_c = y_centers[i]
                ax.hlines(
                    int(y_c), xmin_line, xmax_line,
                    colors='black',
                    linewidth=1.0, zorder=6, antialiased=False)
            else:
                ax.hlines(
                    y_pos[i], xmin_line, xmax_line,
                    colors='black',
                    linewidth=1.0, zorder=6)

        # ----------------------------
        # Styling + export
        # ----------------------------
        ax.set_yticks([])
        ax.grid(True)
        for spine in ax.spines.values():
            spine.set_visible(False)

        buf = io.BytesIO()
        if pixel_mode:
            # fig.savefig(
            #     buf, dpi=dpi, bbox_inches=None, pad_inches=0.0)
            fig.savefig(buf, dpi=dpi, bbox_inches=None, pad_inches=1.0 / dpi)
        else:
            fig.tight_layout(pad=0.1)
            fig.savefig(
                buf, dpi=160, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return iio.imread(buf)

    @staticmethod
    def star_size_points2_from_box_height(
            box_h_px: int, dpi: int, scale: float = 0.5) -> float:
        h_pt = box_h_px * 72.0 / dpi
        return (h_pt * scale) ** 2

compute_lim = FigureUtils.compute_lim
draw_hu_boxes = FigureUtils.draw_hu_boxes
star_size_points2_from_box_height = FigureUtils.star_size_points2_from_box_height