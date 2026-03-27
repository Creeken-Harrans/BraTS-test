#!/usr/bin/env python3
"""
Dependencies:
    nibabel
    numpy
    matplotlib
    pandas

This script scans the BraTS archive recursively, finds the first BraTS-style
patient directory in sorted-path order, loads the 3D NIfTI volumes, and writes
text/JSON/CSV summaries plus multiple high-resolution visualizations.

BraTS data is a 3D medical image volume instead of a 2D JPG/PNG:
    - volume shape describes the size of the 3D array
    - axial/coronal/sagittal refer to three orthogonal slice directions
    - voxel spacing comes from the NIfTI header
    - seg stores voxel-wise tumor labels in the same 3D space
    - bounding box and overlay are defined in 3D and then projected to slices
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Protocol, cast

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle

matplotlib.use("Agg")


DATA_ROOT = "/home/creeken/Desktop/deep-learning/archive"
OUTPUT_DIR = "/home/creeken/Desktop/deep-learning/BraTS-test/visualize/result"

MODALITY_ORDER = ("t1", "t1ce", "t2", "flair")
PLANE_ORDER = ("axial", "coronal", "sagittal")
FIG_DPI = 220
MONTAGE_SLICES = 16
SEG_MONTAGE_SLICES = 12
DISPLAY_PERCENTILES = (1.0, 99.0)

KNOWN_LABEL_MEANINGS = {
    0: "Background",
    1: "Necrotic / non-enhancing tumor core (NCR/NET)",
    2: "Peritumoral edema (ED)",
    4: "Enhancing tumor (ET)",
}

SEG_COLORS = {
    0: "#000000",
    1: "#e63946",
    2: "#ffb703",
    3: "#8d99ae",
    4: "#00b4d8",
}

MODALITY_COLORS = {
    "t1": "#355070",
    "t1ce": "#6d597a",
    "t2": "#b56576",
    "flair": "#e56b6f",
}


class NiftiImageLike(Protocol):
    shape: tuple[int, ...]
    dataobj: Any
    affine: Any
    header: Any

    def get_fdata(self, dtype: Any = np.float32) -> np.ndarray:
        ...


def log(message: str) -> None:
    print(f"[BraTS visualize] {message}")


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def pretty_modality_name(role: str) -> str:
    return {
        "t1": "T1",
        "t1ce": "T1CE",
        "t2": "T2",
        "flair": "FLAIR",
        "seg": "SEG",
    }.get(role, role.upper())


def is_nifti_file(path: Path) -> bool:
    lowered = path.name.lower()
    return path.is_file() and (lowered.endswith(".nii") or lowered.endswith(".nii.gz"))


def strip_nifti_suffix(path: Path) -> str:
    lowered = path.name.lower()
    if lowered.endswith(".nii.gz"):
        return path.name[:-7]
    if lowered.endswith(".nii"):
        return path.name[:-4]
    return path.stem


def detect_series_role(path: Path) -> str | None:
    stem = strip_nifti_suffix(path).lower()
    tokens = [token for token in re.split(r"[^a-z0-9]+", stem) if token]
    compact = "".join(tokens)
    token_set = set(tokens)

    if "seg" in token_set or "segmentation" in token_set or compact.endswith("seg"):
        return "seg"
    if "flair" in token_set or compact.endswith("flair"):
        return "flair"
    if "t1ce" in token_set or "t1gd" in token_set or compact.endswith("t1ce") or compact.endswith("t1gd"):
        return "t1ce"
    if "t2" in token_set or compact.endswith("t2"):
        return "t2"
    if ("t1" in token_set or compact.endswith("t1")) and not (
        compact.endswith("t1ce") or compact.endswith("t1gd")
    ):
        return "t1"
    return None


def find_case_candidate_dirs(data_root: Path) -> list[Path]:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {data_root}")

    grouped: dict[Path, list[tuple[str, Path]]] = {}
    for path in data_root.rglob("*"):
        if not is_nifti_file(path):
            continue
        role = detect_series_role(path)
        if role is None:
            continue
        grouped.setdefault(path.parent, []).append((role, path))

    candidates: list[Path] = []
    for directory, role_files in grouped.items():
        distinct_roles = {role for role, _ in role_files}
        if len(role_files) >= 4 and len(distinct_roles) >= 3:
            candidates.append(directory)

    return sorted(candidates)


def select_first_case_dir(data_root: Path) -> Path:
    candidates = find_case_candidate_dirs(data_root)
    if not candidates:
        raise FileNotFoundError(
            f"No BraTS-style patient directory was found under {data_root}. "
            "Expected a directory containing multiple recognized NIfTI files."
        )
    return candidates[0]


def discover_case_files(case_dir: Path) -> dict[str, Path]:
    nifti_files = sorted(path for path in case_dir.iterdir() if is_nifti_file(path))
    if not nifti_files:
        raise FileNotFoundError(f"Selected case directory contains no NIfTI files: {case_dir}")

    discovered: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    unrecognized: list[Path] = []

    for path in nifti_files:
        role = detect_series_role(path)
        if role is None:
            unrecognized.append(path)
            continue
        if role in discovered:
            duplicates.setdefault(role, [discovered[role]]).append(path)
        else:
            discovered[role] = path

    if duplicates:
        duplicate_lines = []
        for role, files in sorted(duplicates.items()):
            duplicate_lines.append(
                f"{role}: {', '.join(str(path) for path in sorted(files))}"
            )
        raise ValueError(
            "Duplicate files matched the same role in the selected case directory:\n"
            + "\n".join(duplicate_lines)
        )

    missing = [role for role in (*MODALITY_ORDER, "seg") if role not in discovered]
    if missing:
        found_roles = ", ".join(sorted(discovered))
        found_files = "\n".join(f"  - {path.name}" for path in nifti_files)
        extra_text = ""
        if unrecognized:
            extra_text = "\nUnrecognized NIfTI files:\n" + "\n".join(
                f"  - {path.name}" for path in unrecognized
            )
        raise FileNotFoundError(
            f"The first patient directory in sorted order is incomplete: {case_dir}\n"
            f"Missing required roles: {', '.join(missing)}\n"
            f"Recognized roles: {found_roles or 'none'}\n"
            f"Files in directory:\n{found_files}{extra_text}"
        )

    return discovered


def load_volume(path: Path, role: str) -> dict[str, Any]:
    image = cast(NiftiImageLike, cast(object, nib.load(str(path))))

    if len(image.shape) != 3:
        raise ValueError(
            f"{role} file must be a 3D volume, but got shape {image.shape} from {path}"
        )

    if role == "seg":
        raw_data = np.asanyarray(image.dataobj)
        if not np.allclose(raw_data, np.rint(raw_data)):
            raise ValueError(
                f"Segmentation labels must be integer-like, but found non-integer values in {path}"
            )
        data = np.rint(raw_data).astype(np.int16)
    else:
        data = image.get_fdata(dtype=np.float32)
        data = np.asarray(data, dtype=np.float32)

    return {
        "role": role,
        "path": path,
        "image": image,
        "data": data,
        "shape": tuple(int(v) for v in image.shape),
        "dtype": str(image.header.get_data_dtype()),
        "spacing": tuple(float(v) for v in image.header.get_zooms()[:3]),
        "affine": np.asarray(image.affine, dtype=float),
    }


def validate_volume_alignment(records: dict[str, dict[str, Any]]) -> None:
    names = list(records)
    reference = records[names[0]]
    ref_shape = reference["shape"]
    ref_spacing = np.asarray(reference["spacing"], dtype=float)
    ref_affine = np.asarray(reference["affine"], dtype=float)

    mismatches: list[str] = []
    for role, record in records.items():
        if record["shape"] != ref_shape:
            mismatches.append(
                f"{role} shape {record['shape']} does not match reference shape {ref_shape}"
            )
        if not np.allclose(record["spacing"], ref_spacing, atol=1e-5):
            mismatches.append(
                f"{role} voxel spacing {record['spacing']} does not match reference spacing {tuple(ref_spacing)}"
            )
        if not np.allclose(record["affine"], ref_affine, atol=1e-4):
            mismatches.append(f"{role} affine does not match the reference affine")

    if mismatches:
        raise ValueError(
            "Volumes are not aligned in the same 3D space, so overlays would be unreliable:\n"
            + "\n".join(mismatches)
        )


def compute_seg_label_stats(seg_data: np.ndarray) -> list[dict[str, Any]]:
    labels, counts = np.unique(seg_data, return_counts=True)
    total_voxels = int(seg_data.size)
    stats: list[dict[str, Any]] = []
    for label, count in zip(labels, counts):
        label_int = int(label)
        voxel_count = int(count)
        stats.append(
            {
                "label": label_int,
                "name": KNOWN_LABEL_MEANINGS.get(label_int, f"Unknown label {label_int}"),
                "voxel_count": voxel_count,
                "fraction_of_volume": voxel_count / total_voxels,
            }
        )
    return stats


def serializable_affine(affine: np.ndarray) -> list[list[float]]:
    return [[round(float(value), 6) for value in row] for row in affine.tolist()]


def build_case_summary(
    case_dir: Path,
    case_files: dict[str, Path],
    records: dict[str, dict[str, Any]],
    seg_label_stats: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "case_dir": str(case_dir),
        "files": {role: str(case_files[role]) for role in (*MODALITY_ORDER, "seg")},
        "volumes": {},
        "seg_labels": seg_label_stats,
    }

    for role in (*MODALITY_ORDER, "seg"):
        record = records[role]
        summary["volumes"][role] = {
            "path": str(record["path"]),
            "shape": [int(v) for v in record["shape"]],
            "dtype": record["dtype"],
            "voxel_spacing": [float(v) for v in record["spacing"]],
            "affine": serializable_affine(record["affine"]),
        }

    return summary


def write_case_summary_text(summary: dict[str, Any], output_path: Path) -> None:
    lines: list[str] = []
    lines.append("BraTS First Case Summary")
    lines.append("=" * 80)
    lines.append(f"Case directory: {summary['case_dir']}")
    lines.append("")
    lines.append("Files")
    lines.append("-" * 80)

    for role in (*MODALITY_ORDER, "seg"):
        lines.append(f"{role}: {summary['files'][role]}")

    lines.append("")
    lines.append("Volume Metadata")
    lines.append("-" * 80)

    for role in (*MODALITY_ORDER, "seg"):
        info = summary["volumes"][role]
        lines.append(f"[{role}]")
        lines.append(f"  path: {info['path']}")
        lines.append(f"  shape: {tuple(info['shape'])}")
        lines.append(f"  dtype: {info['dtype']}")
        lines.append(f"  voxel spacing: {tuple(info['voxel_spacing'])}")
        lines.append("  affine:")
        for row in info["affine"]:
            lines.append("    " + " ".join(f"{value:10.6f}" for value in row))
        lines.append("")

    lines.append("Segmentation Labels")
    lines.append("-" * 80)
    for item in summary["seg_labels"]:
        lines.append(
            f"label {item['label']}: {item['name']} | "
            f"voxel_count={item['voxel_count']} | "
            f"fraction_of_volume={item['fraction_of_volume']:.6f}"
        )

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def compute_basic_stats(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"min": None, "max": None, "mean": None, "std": None}
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def compute_intensity_stats(records: dict[str, dict[str, Any]]) -> Any:
    rows: list[dict[str, Any]] = []
    for role in MODALITY_ORDER:
        volume = np.asarray(records[role]["data"], dtype=np.float32)
        all_values = volume.reshape(-1)
        nonzero_values = all_values[all_values != 0]

        all_stats = compute_basic_stats(all_values)
        nonzero_stats = compute_basic_stats(nonzero_values)

        rows.append(
            {
                "modality": role,
                "path": str(records[role]["path"]),
                "shape": "x".join(str(v) for v in records[role]["shape"]),
                "dtype": records[role]["dtype"],
                "nonzero_voxel_ratio": float(np.count_nonzero(volume) / volume.size),
                "all_min": all_stats["min"],
                "all_max": all_stats["max"],
                "all_mean": all_stats["mean"],
                "all_std": all_stats["std"],
                "nonzero_min": nonzero_stats["min"],
                "nonzero_max": nonzero_stats["max"],
                "nonzero_mean": nonzero_stats["mean"],
                "nonzero_std": nonzero_stats["std"],
            }
        )
    return pd.DataFrame(rows)


def write_seg_labels_summary(seg_label_stats: list[dict[str, Any]], output_path: Path) -> None:
    total_voxels = sum(item["voxel_count"] for item in seg_label_stats)
    tumor_voxels = sum(item["voxel_count"] for item in seg_label_stats if item["label"] != 0)

    lines = [
        "BraTS Segmentation Labels Summary",
        "=" * 80,
        "BraTS segmentation is a 3D label volume aligned with the MRI volumes.",
        f"Total voxels in the volume: {total_voxels}",
        f"Total non-background tumor voxels: {tumor_voxels}",
        "",
        "Labels",
        "-" * 80,
    ]

    for item in seg_label_stats:
        lines.append(
            f"label {item['label']}: {item['name']}\n"
            f"  voxel count: {item['voxel_count']}\n"
            f"  fraction of full volume: {item['fraction_of_volume']:.6f}"
        )

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def compute_display_range(volume: np.ndarray) -> tuple[float, float]:
    nonzero = volume[volume != 0]
    values = nonzero if nonzero.size else volume.reshape(-1)

    if values.size == 0:
        return 0.0, 1.0

    vmin, vmax = np.percentile(values, DISPLAY_PERCENTILES)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin = float(np.min(values))
        vmax = float(np.max(values))
    if math.isclose(float(vmin), float(vmax)):
        vmax = float(vmin) + 1.0
    return float(vmin), float(vmax)


def get_plane_length(volume: np.ndarray, plane: str) -> int:
    if plane == "axial":
        return int(volume.shape[2])
    if plane == "coronal":
        return int(volume.shape[1])
    if plane == "sagittal":
        return int(volume.shape[0])
    raise ValueError(f"Unsupported plane: {plane}")


def get_display_slice(volume: np.ndarray, plane: str, index: int) -> np.ndarray:
    if plane == "axial":
        slice_2d = volume[:, :, index]
    elif plane == "coronal":
        slice_2d = volume[:, index, :]
    elif plane == "sagittal":
        slice_2d = volume[index, :, :]
    else:
        raise ValueError(f"Unsupported plane: {plane}")

    # Transpose so that x/y are displayed in a consistent medical-image view.
    return np.asarray(slice_2d).T


def count_nonzero_per_slice(mask: np.ndarray, plane: str) -> np.ndarray:
    if plane == "axial":
        return np.count_nonzero(mask, axis=(0, 1))
    if plane == "coronal":
        return np.count_nonzero(mask, axis=(0, 2))
    if plane == "sagittal":
        return np.count_nonzero(mask, axis=(1, 2))
    raise ValueError(f"Unsupported plane: {plane}")


def finalize_indices(preferred: np.ndarray, total_length: int, target_count: int) -> list[int]:
    if total_length <= 0:
        return []

    seen: set[int] = set()
    result: list[int] = []

    for raw_index in np.round(preferred).astype(int):
        index = int(max(0, min(total_length - 1, raw_index)))
        if index not in seen:
            seen.add(index)
            result.append(index)

    if len(result) < min(target_count, total_length):
        fallback = np.round(
            np.linspace(0, total_length - 1, max(target_count * 4, total_length))
        ).astype(int)
        for index in fallback:
            int_index = int(index)
            if int_index not in seen:
                seen.add(int_index)
                result.append(int_index)
            if len(result) >= min(target_count, total_length):
                break

    return result[: min(target_count, total_length)]


def select_montage_indices(volume: np.ndarray, plane: str, num_slices: int) -> list[int]:
    total_length = get_plane_length(volume, plane)
    counts = count_nonzero_per_slice(volume != 0, plane)
    informative = np.where(counts > 0)[0]

    if informative.size >= num_slices:
        preferred = informative[
            np.round(np.linspace(0, informative.size - 1, num_slices)).astype(int)
        ]
    elif informative.size > 0:
        preferred = np.linspace(int(informative[0]), int(informative[-1]), num_slices)
    else:
        preferred = np.linspace(0, total_length - 1, num_slices)

    return finalize_indices(preferred, total_length, num_slices)


def select_best_seg_slice(seg_data: np.ndarray, plane: str) -> int:
    counts = count_nonzero_per_slice(seg_data > 0, plane)
    max_count = int(np.max(counts)) if counts.size else 0
    if max_count <= 0:
        raise ValueError(
            "Segmentation contains only background label 0, so tumor-focused visualizations cannot be created."
        )
    return int(np.argmax(counts))


def build_seg_colormap(seg_label_stats: list[dict[str, Any]]) -> tuple[Any, Any]:
    labels = [item["label"] for item in seg_label_stats]
    max_label = max(max(labels), max(SEG_COLORS))
    colors = [SEG_COLORS.get(index, "#cccccc") for index in range(max_label + 1)]
    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, max_label + 1.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def build_label_handles(seg_label_stats: list[dict[str, Any]], include_background: bool = False) -> list[Any]:
    handles: list[Any] = []
    for item in seg_label_stats:
        label = item["label"]
        if label == 0 and not include_background:
            continue
        handles.append(
            Patch(
                facecolor=SEG_COLORS.get(label, "#cccccc"),
                edgecolor="black",
                label=f"{label}: {item['name']}",
            )
        )
    return handles


def save_figure(fig: Any, output_path: Path) -> None:
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_modalities_mid_slices(records: dict[str, dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(
        nrows=len(MODALITY_ORDER),
        ncols=len(PLANE_ORDER),
        figsize=(14, 18),
        dpi=FIG_DPI,
    )

    for row, role in enumerate(MODALITY_ORDER):
        volume = records[role]["data"]
        vmin, vmax = compute_display_range(volume)
        for col, plane in enumerate(PLANE_ORDER):
            index = get_plane_length(volume, plane) // 2
            ax = axes[row, col]
            ax.imshow(
                get_display_slice(volume, plane, index),
                cmap="gray",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(
                f"{pretty_modality_name(role)} | {plane} | slice {index}",
                fontsize=10,
            )
            ax.axis("off")

    fig.suptitle(
        "BraTS 3D volume middle slices across axial / coronal / sagittal planes",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, output_path)


def plot_modality_montage(role: str, record: dict[str, Any], output_path: Path) -> None:
    volume = record["data"]
    indices = select_montage_indices(volume, plane="axial", num_slices=MONTAGE_SLICES)
    columns = 4
    rows = math.ceil(len(indices) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(16, rows * 4), dpi=FIG_DPI)
    axes = np.atleast_1d(axes).ravel()
    vmin, vmax = compute_display_range(volume)

    for ax, index in zip(axes, indices):
        ax.imshow(
            get_display_slice(volume, "axial", index),
            cmap="gray",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"axial slice {index}", fontsize=10)
        ax.axis("off")

    for ax in axes[len(indices) :]:
        ax.axis("off")

    fig.suptitle(
        f"{pretty_modality_name(role)} axial montage "
        "(3D volume sampled across multiple slices, avoiding all-black regions)",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, output_path)


def plot_segmentation_montage(
    seg_record: dict[str, Any],
    seg_label_stats: list[dict[str, Any]],
    output_path: Path,
) -> None:
    seg_data = seg_record["data"]
    indices = select_montage_indices(seg_data, plane="axial", num_slices=SEG_MONTAGE_SLICES)
    columns = 4
    rows = math.ceil(len(indices) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(16, rows * 4), dpi=FIG_DPI)
    axes = np.atleast_1d(axes).ravel()
    cmap, norm = build_seg_colormap(seg_label_stats)

    for ax, index in zip(axes, indices):
        display_slice = get_display_slice(seg_data, "axial", index)
        tumor_voxels = int(np.count_nonzero(display_slice))
        ax.imshow(
            display_slice,
            cmap=cmap,
            norm=norm,
            origin="lower",
            interpolation="nearest",
        )
        ax.set_facecolor("black")
        ax.set_title(f"axial slice {index} | labeled voxels {tumor_voxels}", fontsize=10)
        ax.axis("off")

    for ax in axes[len(indices) :]:
        ax.axis("off")

    handles = build_label_handles(seg_label_stats, include_background=False)
    fig.suptitle(
        "Segmentation axial montage with discrete tumor labels", fontsize=15
    )
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=max(1, len(handles)),
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, output_path)


def plot_overlay_best_slices(
    base_record: dict[str, Any],
    seg_record: dict[str, Any],
    seg_label_stats: list[dict[str, Any]],
    output_path: Path,
) -> None:
    base_volume = base_record["data"]
    seg_data = seg_record["data"]
    vmin, vmax = compute_display_range(base_volume)
    cmap, norm = build_seg_colormap(seg_label_stats)
    fig, axes = plt.subplots(1, len(PLANE_ORDER), figsize=(18, 6), dpi=FIG_DPI)

    for ax, plane in zip(axes, PLANE_ORDER):
        index = select_best_seg_slice(seg_data, plane)
        base_slice = get_display_slice(base_volume, plane, index)
        seg_slice = get_display_slice(seg_data, plane, index)
        tumor_voxels = int(np.count_nonzero(seg_slice))

        ax.imshow(base_slice, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        ax.imshow(
            np.ma.masked_where(seg_slice == 0, seg_slice),
            cmap=cmap,
            norm=norm,
            origin="lower",
            alpha=0.45,
            interpolation="nearest",
        )
        ax.set_title(f"{plane} | slice {index} | tumor voxels {tumor_voxels}", fontsize=11)
        ax.axis("off")

    handles = build_label_handles(seg_label_stats, include_background=False)
    fig.suptitle(
        f"{pretty_modality_name(base_record['role'])} + segmentation overlay on tumor-dominant slices",
        fontsize=16,
    )
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=max(1, len(handles)),
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, output_path)


def compute_3d_bounding_box(mask: np.ndarray, margin: int = 5) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise ValueError("Cannot compute a tumor bounding box because segmentation has no non-zero labels.")

    mins = coordinates.min(axis=0)
    maxs = coordinates.max(axis=0) + 1
    mins = np.maximum(mins - margin, 0)
    maxs = np.minimum(maxs + margin, np.asarray(mask.shape))
    return mins.astype(int), maxs.astype(int)


def project_bbox_to_plane(mins: np.ndarray, maxs: np.ndarray, plane: str) -> tuple[int, int, int, int]:
    if plane == "axial":
        return (
            int(mins[0]),
            int(mins[1]),
            int(maxs[0] - mins[0]),
            int(maxs[1] - mins[1]),
        )
    if plane == "coronal":
        return (
            int(mins[0]),
            int(mins[2]),
            int(maxs[0] - mins[0]),
            int(maxs[2] - mins[2]),
        )
    if plane == "sagittal":
        return (
            int(mins[1]),
            int(mins[2]),
            int(maxs[1] - mins[1]),
            int(maxs[2] - mins[2]),
        )
    raise ValueError(f"Unsupported plane: {plane}")


def crop_display_slice(
    display_slice: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    plane: str,
) -> np.ndarray:
    x0, y0, width, height = project_bbox_to_plane(mins, maxs, plane)
    return display_slice[y0 : y0 + height, x0 : x0 + width]


def plot_tumor_bbox_views(
    base_record: dict[str, Any],
    seg_record: dict[str, Any],
    seg_label_stats: list[dict[str, Any]],
    output_path: Path,
) -> None:
    base_volume = base_record["data"]
    seg_data = seg_record["data"]
    mins, maxs = compute_3d_bounding_box(seg_data > 0, margin=5)
    vmin, vmax = compute_display_range(base_volume)
    cmap, norm = build_seg_colormap(seg_label_stats)

    fig, axes = plt.subplots(2, len(PLANE_ORDER), figsize=(18, 10), dpi=FIG_DPI)

    for col, plane in enumerate(PLANE_ORDER):
        index = select_best_seg_slice(seg_data, plane)
        base_slice = get_display_slice(base_volume, plane, index)
        seg_slice = get_display_slice(seg_data, plane, index)
        masked_seg = np.ma.masked_where(seg_slice == 0, seg_slice)

        ax_full = axes[0, col]
        ax_crop = axes[1, col]

        ax_full.imshow(base_slice, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        ax_full.imshow(
            masked_seg,
            cmap=cmap,
            norm=norm,
            origin="lower",
            alpha=0.35,
            interpolation="nearest",
        )
        x0, y0, width, height = project_bbox_to_plane(mins, maxs, plane)
        ax_full.add_patch(
            Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor="white",
                linewidth=2.2,
                linestyle="--",
            )
        )
        ax_full.set_title(f"{plane} full slice {index} with tumor bbox", fontsize=11)
        ax_full.axis("off")

        crop_base = crop_display_slice(base_slice, mins, maxs, plane)
        crop_seg = crop_display_slice(seg_slice, mins, maxs, plane)
        ax_crop.imshow(crop_base, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        ax_crop.imshow(
            np.ma.masked_where(crop_seg == 0, crop_seg),
            cmap=cmap,
            norm=norm,
            origin="lower",
            alpha=0.45,
            interpolation="nearest",
        )
        ax_crop.set_title(f"{plane} bbox crop", fontsize=11)
        ax_crop.axis("off")

    handles = build_label_handles(seg_label_stats, include_background=False)
    fig.suptitle(
        f"Tumor 3D bounding-box views on {pretty_modality_name(base_record['role'])}",
        fontsize=16,
    )
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=max(1, len(handles)),
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, output_path)


def plot_intensity_histograms(records: dict[str, dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(
        len(MODALITY_ORDER),
        2,
        figsize=(16, 18),
        dpi=FIG_DPI,
    )

    for row, role in enumerate(MODALITY_ORDER):
        volume = np.asarray(records[role]["data"], dtype=np.float32).reshape(-1)
        nonzero = volume[volume != 0]
        color = MODALITY_COLORS[role]

        ax_all = axes[row, 0]
        ax_nonzero = axes[row, 1]

        ax_all.hist(volume, bins=120, color=color, alpha=0.85)
        ax_all.set_title(f"{pretty_modality_name(role)} | all voxels", fontsize=11)
        ax_all.set_xlabel("Intensity")
        ax_all.set_ylabel("Voxel count")

        if nonzero.size:
            ax_nonzero.hist(nonzero, bins=120, color=color, alpha=0.85)
        else:
            ax_nonzero.text(
                0.5,
                0.5,
                "No non-zero voxels",
                ha="center",
                va="center",
                transform=ax_nonzero.transAxes,
            )
        ax_nonzero.set_title(f"{pretty_modality_name(role)} | non-zero voxels only", fontsize=11)
        ax_nonzero.set_xlabel("Intensity")
        ax_nonzero.set_ylabel("Voxel count")

    fig.suptitle("Intensity distributions for BraTS modalities", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, output_path)


def add_bar_labels(ax: Any, bars: Any, percentages: list[float]) -> None:
    for bar, percentage in zip(bars, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{percentage:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_seg_label_distribution(seg_label_stats: list[dict[str, Any]], output_path: Path) -> None:
    labels = [item["label"] for item in seg_label_stats]
    counts = [item["voxel_count"] for item in seg_label_stats]
    total = float(sum(counts))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=FIG_DPI)

    full_percentages = [count / total * 100.0 for count in counts]
    full_colors = [SEG_COLORS.get(label, "#cccccc") for label in labels]
    bars_all = axes[0].bar([str(label) for label in labels], full_percentages, color=full_colors)
    add_bar_labels(axes[0], bars_all, full_percentages)
    axes[0].set_title("Label proportion in the full 3D volume", fontsize=12)
    axes[0].set_xlabel("Label value")
    axes[0].set_ylabel("Percentage of all voxels")

    tumor_items = [item for item in seg_label_stats if item["label"] != 0 and item["voxel_count"] > 0]
    if tumor_items:
        tumor_total = float(sum(item["voxel_count"] for item in tumor_items))
        tumor_labels = [item["label"] for item in tumor_items]
        tumor_percentages = [item["voxel_count"] / tumor_total * 100.0 for item in tumor_items]
        tumor_colors = [SEG_COLORS.get(label, "#cccccc") for label in tumor_labels]
        bars_tumor = axes[1].bar(
            [str(label) for label in tumor_labels],
            tumor_percentages,
            color=tumor_colors,
        )
        add_bar_labels(axes[1], bars_tumor, tumor_percentages)
        axes[1].set_title("Non-background label proportion inside tumor voxels", fontsize=12)
        axes[1].set_xlabel("Label value")
        axes[1].set_ylabel("Percentage of tumor voxels")
    else:
        axes[1].text(
            0.5,
            0.5,
            "No non-background labels found",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_title("Non-background label proportion inside tumor voxels", fontsize=12)
        axes[1].set_axis_off()

    fig.suptitle("BraTS segmentation label distribution", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, output_path)


def write_readme(output_dir: Path, case_dir: Path) -> None:
    content = f"""# BraTS 第一个病例可视化结果

本目录由 `visualize_first_brats_case.py` 自动生成。

- 数据根目录：`{DATA_ROOT}`
- 当前自动选中的病例目录：`{case_dir}`
- 输出目录：`{output_dir}`

## 为什么 BraTS 数据不是 jpg/png，而是 3D NIfTI

BraTS 使用的是医学影像常见的 **NIfTI** 格式，因为它保存的是完整的三维 MRI 体数据，而不是单张二维图片。

- 一份 NIfTI 文件通常对应一个 3D volume，例如 `(240, 240, 155)` 这样的体素网格。
- 每个体素不仅有强度值，还带有空间信息，例如 **voxel spacing** 和 **affine**。
- 这样我们才能在 **axial / coronal / sagittal** 三个方向上切片观察，而不是只看一张静态图。
- 分割标签 `seg` 也是一个与 MRI 完全对齐的 3D 体数据，所以可以做逐体素 overlay、3D bounding box 和标签统计。

## 模态说明

- `t1`：T1 加权 MRI，适合看基础解剖结构。
- `t1ce`：T1 对比增强 MRI，也常写作 `t1gd`，增强区通常更明显。
- `t2`：T2 加权 MRI，液体相关信号更亮。
- `flair`：抑制脑脊液后的 T2 风格图像，常更利于观察水肿区域。
- `seg`：分割标签体数据，不是普通图片，而是每个体素一个类别值。

## 输出文件说明

- `case_summary.txt`：病例路径、文件路径、shape、dtype、voxel spacing、affine、seg 标签统计的可读文本版摘要。
- `case_summary.json`：与文本摘要对应的结构化 JSON，便于程序继续处理。
- `intensity_stats.csv`：四个模态的强度统计，包括全体素和非零区域统计。
- `modalities_mid_slices.png`：四个模态在 axial / coronal / sagittal 三个方向的中间切片总览。
- `t1_montage.png`：T1 的 axial 多切片 montage，用来观察 3D 体数据沿 z 方向的变化。
- `t1ce_montage.png`：T1CE 的 axial 多切片 montage。
- `t2_montage.png`：T2 的 axial 多切片 montage。
- `flair_montage.png`：FLAIR 的 axial 多切片 montage。
- `seg_labels_summary.txt`：seg 标签值解释及体素数量统计。
- `seg_montage.png`：seg 标签的多切片可视化，使用离散颜色区分标签。
- `flair_overlay_best_slices.png`：在 FLAIR 上叠加 seg，自动选择最能体现肿瘤的三视图切片。
- `t1ce_overlay_best_slices.png`：在 T1CE 上叠加 seg，观察增强区域与标签的对应关系。
- `tumor_bbox_views.png`：根据 seg 计算 3D tumor bounding box，并展示整图视角与局部放大视角。
- `intensity_histograms.png`：四个模态的强度分布直方图，分别展示全体素和非零体素。
- `seg_label_distribution.png`：seg 标签在全 volume 中的占比，以及在非背景 tumor voxel 中的占比。

## 学习建议

- 先看 `case_summary.txt`，建立对 shape、spacing、affine 的基本认识。
- 再看 `modalities_mid_slices.png` 和各个 montage，理解同一病例在不同模态、不同切片中的表现差异。
- 最后看 overlay 和 bounding box 图，体会分割标签是如何与 3D MRI 体数据对齐的。
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    data_root = Path(DATA_ROOT)
    output_dir = Path(OUTPUT_DIR)
    ensure_output_dir(output_dir)

    log(f"Scanning data root: {data_root}")
    case_dir = select_first_case_dir(data_root)
    log(f"Selected first case directory: {case_dir}")

    case_files = discover_case_files(case_dir)
    log("Loading 3D NIfTI volumes and headers")

    records: dict[str, dict[str, Any]] = {}
    for role in (*MODALITY_ORDER, "seg"):
        records[role] = load_volume(case_files[role], role)

    validate_volume_alignment(records)

    seg_label_stats = compute_seg_label_stats(records["seg"]["data"])

    summary = build_case_summary(case_dir, case_files, records, seg_label_stats)
    log("Writing summary text, JSON, and CSV files")
    write_case_summary_text(summary, output_dir / "case_summary.txt")
    (output_dir / "case_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    compute_intensity_stats(records).to_csv(
        output_dir / "intensity_stats.csv",
        index=False,
    )
    write_seg_labels_summary(seg_label_stats, output_dir / "seg_labels_summary.txt")

    log("Creating multi-view and tumor-focused figures")
    plot_modalities_mid_slices(records, output_dir / "modalities_mid_slices.png")
    for role in MODALITY_ORDER:
        plot_modality_montage(role, records[role], output_dir / f"{role}_montage.png")
    plot_segmentation_montage(records["seg"], seg_label_stats, output_dir / "seg_montage.png")
    plot_overlay_best_slices(
        records["flair"],
        records["seg"],
        seg_label_stats,
        output_dir / "flair_overlay_best_slices.png",
    )
    plot_overlay_best_slices(
        records["t1ce"],
        records["seg"],
        seg_label_stats,
        output_dir / "t1ce_overlay_best_slices.png",
    )
    plot_tumor_bbox_views(
        records["flair"],
        records["seg"],
        seg_label_stats,
        output_dir / "tumor_bbox_views.png",
    )
    plot_intensity_histograms(records, output_dir / "intensity_histograms.png")
    plot_seg_label_distribution(seg_label_stats, output_dir / "seg_label_distribution.png")

    log("Writing README")
    write_readme(output_dir, case_dir)
    log(f"All results were written to: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc
