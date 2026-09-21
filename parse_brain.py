from __future__ import annotations

import asyncio
import gc
import io
import json
import logging
import math
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import numpy as np
import pyarrow.ipc as pa_ipc
import polars as pl
from aiohttp.resolver import ThreadedResolver
from aiogram import Bot, Dispatcher, F
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.types import Message
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix
import uvicorn

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


LOG = logging.getLogger("flywire")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

CONNECTOME_PATH = os.getenv("CONNECTOME_PATH", "proofread_connections_783.feather")
ANNOTATIONS_PATH = os.getenv("ANNOTATIONS_PATH", "annotations.tsv")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
SIM_STEPS = max(25, min(30, int(os.getenv("SIM_STEPS", "30"))))
MAX_SNAPSHOT_NODES = 250
DIRECTION_DEADZONE = 0.07


@dataclass(slots=True)
class Graph:
    node_ids: np.ndarray
    W: csr_matrix
    x: np.ndarray
    y: np.ndarray
    source_sign: np.ndarray
    source_known: np.ndarray
    in_degree: np.ndarray
    out_degree: np.ndarray
    optic: np.ndarray
    optic_left: np.ndarray
    optic_right: np.ndarray
    descending: np.ndarray
    dnp01: np.ndarray
    steering: np.ndarray
    steering_left: np.ndarray
    steering_right: np.ndarray
    food: np.ndarray
    central_complex: np.ndarray
    gustatory: np.ndarray
    feeding_motor: np.ndarray
    sez: np.ndarray
    grn: np.ndarray
    fbn: np.ndarray
    retina_on: np.ndarray
    retina_off: np.ndarray
    retina_type: np.ndarray
    coordinate_origin: tuple[float, float] = (0.0, 0.0)
    coordinate_span: tuple[float, float] = (1.0, 1.0)
    retina_synaptic_mass: np.ndarray | None = None
    retina_raw_synaptic_mass: np.ndarray | None = None

    @property
    def n(self) -> int:
        return int(self.node_ids.size)

    @property
    def groups(self) -> dict[str, np.ndarray]:
        return {
            name: getattr(self, name)
            for name in (
                "optic", "optic_left", "optic_right", "descending", "dnp01",
                "steering", "steering_left", "steering_right", "food",
                "central_complex", "gustatory", "feeding_motor", "sez", "grn",
                "fbn", "retina_on", "retina_off",
            )
        }


def load_graph(path: str) -> Graph:
    # Validate annotations before allocating the large connectome arrays. These
    # are the actual column names in the supplied FlyWire annotation release.
    annotations_path = ANNOTATIONS_PATH
    if not os.path.isfile(annotations_path):
        raise FileNotFoundError(
            f"FlyWire annotations are required: {annotations_path}. "
            "Provide annotations.tsv or set ANNOTATIONS_PATH."
        )
    annotation_columns = (
        "root_id", "super_class", "cell_class", "cell_sub_class", "cell_type",
        "side", "top_nt", "soma_x", "soma_y", "pos_x", "pos_y",
    )
    schema = pl.scan_csv(
        annotations_path, separator="\t", infer_schema=False,
    ).collect_schema()
    missing_columns = sorted(set(annotation_columns) - set(schema.names()))
    if missing_columns:
        raise ValueError(
            f"FlyWire annotations {annotations_path} lack required columns: "
            + ", ".join(missing_columns)
        )
    del schema

    reader = pa_ipc.open_file(path)
    names = reader.schema.names
    required = ("pre_pt_root_id", "post_pt_root_id", "syn_count")
    if any(name not in names for name in required):
        raise ValueError(f"Connectome lacks required columns: {required}")
    pre_col, post_col, weight_col = (names.index(name) for name in required)

    nodes: set[int] = set()
    edge_count = 0
    for batch_index in range(reader.num_record_batches):
        batch = reader.get_batch(batch_index)
        pre = np.asarray(batch.column(pre_col))
        post = np.asarray(batch.column(post_col))
        nodes.update(int(value) for value in np.unique(np.concatenate((pre, post))))
        edge_count += batch.num_rows
    if edge_count == 0 or not nodes:
        raise ValueError("Connectome contains no edges")

    node_ids = np.asarray(sorted(nodes), dtype=np.int64)
    del nodes
    pre_index = np.empty(edge_count, dtype=np.int32)
    post_index = np.empty(edge_count, dtype=np.int32)
    weights = np.empty(edge_count, dtype=np.float32)
    offset = 0
    for batch_index in range(reader.num_record_batches):
        batch = reader.get_batch(batch_index)
        end = offset + batch.num_rows
        pre = np.asarray(batch.column(pre_col))
        post = np.asarray(batch.column(post_col))
        weight = np.nan_to_num(
            np.asarray(batch.column(weight_col), dtype=np.float32),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        pre_index[offset:end] = np.searchsorted(node_ids, pre)
        post_index[offset:end] = np.searchsorted(node_ids, post)
        weights[offset:end] = np.maximum(weight, np.float32(0.0))
        offset = end

    W = csr_matrix(
        (weights, (post_index, pre_index)),
        shape=(node_ids.size, node_ids.size), dtype=np.float32,
    )
    W.sum_duplicates()
    W.eliminate_zeros()
    W.sort_indices()
    # Release the Feather batches and COO arrays before loading annotations.
    del reader, batch, pre, post, weight, weights, pre_index, post_index
    gc.collect()

    ann = pl.read_csv(
        annotations_path, separator="\t", columns=list(annotation_columns),
        infer_schema=False,
        schema_overrides={
            "root_id": pl.Int64,
            **{name: pl.Float32 for name in ("soma_x", "soma_y", "pos_x", "pos_y")},
        },
        null_values=["", "NA", "NaN", "null"],
    )
    if ann.get_column("root_id").null_count():
        raise ValueError("FlyWire annotations contain missing root_id values")
    if ann.get_column("root_id").n_unique() != ann.height:
        raise ValueError("FlyWire annotations contain duplicate root_id values")
    root = ann.get_column("root_id").to_numpy()
    ann_index = np.searchsorted(node_ids, root)
    matched = ann_index < node_ids.size
    matched &= node_ids[np.minimum(ann_index, node_ids.size - 1)] == root
    dst = ann_index[matched]
    LOG.info(
        "FlyWire root_id matched: %d/%d graph nodes, %d annotation rows; unmatched=%d",
        dst.size, node_ids.size, ann.height, node_ids.size - dst.size,
    )
    if not dst.size:
        raise ValueError("No annotation root_id values match the connectome")

    def text_column(name: str) -> np.ndarray:
        return np.asarray(
            ann.get_column(name).fill_null("").str.strip_chars().to_numpy(), dtype=str,
        )

    superclass = text_column("super_class")
    cell_class = text_column("cell_class")
    cell_sub_class = text_column("cell_sub_class")
    cell_type = text_column("cell_type")
    side = text_column("side")
    transmitter = np.char.lower(text_column("top_nt"))
    for name, values in (
        ("super_class", superclass), ("cell_class", cell_class),
        ("cell_sub_class", cell_sub_class), ("side", side), ("top_nt", transmitter),
    ):
        labels, counts = np.unique(values, return_counts=True)
        LOG.info("FlyWire %s values: %s", name, dict(zip(labels.tolist(), counts.tolist())))
    for pattern in ("DNa", "DNp01", "FBn", "GRN"):
        selected = np.char.find(cell_type, pattern) >= 0
        labels, counts = np.unique(cell_type[selected], return_counts=True)
        LOG.info("FlyWire cell_type containing %s: %s", pattern, dict(zip(labels.tolist(), counts.tolist())))

    x = np.full(node_ids.size, np.nan, dtype=np.float32)
    y = np.full(node_ids.size, np.nan, dtype=np.float32)
    coordinate_origin: list[float] = []
    coordinate_span: list[float] = []
    for axis, output in (("x", x), ("y", y)):
        soma = ann.get_column(f"soma_{axis}").to_numpy()
        position = ann.get_column(f"pos_{axis}").to_numpy()
        raw = np.where(np.isfinite(soma), soma, position)[matched]
        finite = np.isfinite(raw)
        if not np.any(finite):
            raise ValueError(f"FlyWire annotations contain no finite {axis} coordinates")
        lo, hi = float(raw[finite].min()), float(raw[finite].max())
        coordinate_origin.append(lo)
        coordinate_span.append(hi - lo)
        if hi > lo:
            output[dst[finite]] = np.clip((raw[finite] - lo) / (hi - lo), 0, 1)
        else:
            output[dst[finite]] = np.float32(0.5)
    LOG.info(
        "FlyWire coordinates: soma_x/pos_x and soma_y/pos_y; %d nodes lack coordinates",
        int(np.count_nonzero(~np.isfinite(x) | ~np.isfinite(y))),
    )

    def node_mask(annotation_mask: np.ndarray) -> np.ndarray:
        mask = np.zeros(node_ids.size, dtype=bool)
        mask[dst] = annotation_mask[matched]
        return mask

    optic = node_mask(
        np.isin(superclass, ("optic", "visual_projection", "visual_centrifugal"))
        | np.isin(cell_class, ("visual", "optic_lobes"))
    )
    left = node_mask(side == "left")
    right = node_mask(side == "right")
    descending = node_mask(superclass == "descending")
    steering = descending & node_mask(np.char.startswith(cell_type, "DNa"))
    gustatory = node_mask(cell_class == "gustatory")
    feeding_motor = node_mask(np.isin(cell_sub_class, (
        "ingestion_motor_neuron", "proboscis_motor_neuron", "crop_motor_neuron",
        "salivary_motor_neuron", "haustellum_motor_neuron",
    )))
    sez = node_mask(cell_sub_class == "SEZ-NSC")
    grn = node_mask(np.char.find(cell_type, "GRN") >= 0)
    # This release has no FBn labels. Keep its explicit empty diagnostic group;
    # do not confuse fan-shaped-body FB* cells with feeding FBn neurons.
    fbn = node_mask(np.char.startswith(cell_type, "FBn"))
    masks = {
        "optic": optic,
        "optic_left": optic & left,
        "optic_right": optic & right,
        "descending": descending,
        "dnp01": descending & node_mask(cell_type == "DNp01"),
        "steering": steering,
        "steering_left": steering & left,
        "steering_right": steering & right,
        "food": gustatory | feeding_motor | sez | grn | fbn,
        "central_complex": node_mask(cell_class == "CX"),
        "gustatory": gustatory,
        "feeding_motor": feeding_motor,
        "sez": sez,
        "grn": grn,
        "fbn": fbn,
        "retina_on": optic & node_mask(np.isin(cell_type, ("L1", "Mi1", "Tm3"))),
        "retina_off": optic & node_mask(np.isin(cell_type, ("L2", "Tm1", "Tm2"))),
    }
    retina_type = np.zeros(node_ids.size, dtype=np.int8)
    for code, label in enumerate(("L1", "Mi1", "Tm3", "L2", "Tm1", "Tm2"), start=1):
        retina_type[optic & node_mask(cell_type == label)] = code
    for name, mask in masks.items():
        LOG.info("FlyWire functional group %s: %d", name, int(mask.sum()))
    mandatory = (
        "optic", "optic_left", "optic_right", "descending", "dnp01", "steering",
        "steering_left", "steering_right", "food", "central_complex",
        "gustatory", "feeding_motor", "sez", "grn", "retina_on", "retina_off",
    )
    absent = [name for name in mandatory if not np.any(masks[name])]
    if absent:
        raise ValueError(
            "Required FlyWire functional groups are absent after root_id matching: "
            + ", ".join(absent)
        )
    for name in ("retina_on", "retina_off"):
        for side_name, side_mask in (("left", left), ("right", right)):
            if not np.any(masks[name] & side_mask & np.isfinite(x) & np.isfinite(y)):
                raise ValueError(f"No annotated {name}/{side_name} cells with coordinates")
    if not np.any(fbn):
        LOG.info("FlyWire FBn is absent in this annotation release (optional group=0)")

    inhibitory_names = ("gaba", "glutamate", "histamine")
    known_names = (*inhibitory_names, "acetylcholine", "dopamine", "serotonin", "octopamine")
    source_known = node_mask(np.isin(transmitter, known_names))
    source_sign = np.ones(node_ids.size, dtype=np.float32)
    source_sign[node_mask(np.isin(transmitter, inhibitory_names))] = np.float32(-1.0)
    # A blank/unrecognized transmitter is not assigned invented biology. Its
    # +1 storage placeholder is inert: remove all outgoing edges explicitly.
    unknown_edges = ~source_known[W.indices]
    removed_edges = int(unknown_edges.sum())
    W.data[unknown_edges] = np.float32(0.0)
    del unknown_edges
    W.eliminate_zeros()
    LOG.info(
        "FlyWire sources: excitatory=%d inhibitory=%d unknown/blocked=%d; removed outgoing edges=%d",
        int(np.count_nonzero(source_known & (source_sign > 0))),
        int(np.count_nonzero(source_known & (source_sign < 0))),
        int(np.count_nonzero(~source_known)), removed_edges,
    )
    row_sum = np.asarray(W.sum(axis=1)).ravel().astype(np.float32, copy=False)
    row_sum[row_sum <= 0] = np.float32(1.0)
    retinal_sources = masks["retina_on"] | masks["retina_off"]
    retina_mass = np.zeros(node_ids.size, dtype=np.float64)
    retina_raw_mass = np.zeros(node_ids.size, dtype=np.float64)
    # Bound normalization scratch space to a block rather than nnz float32s.
    for first_row in range(0, node_ids.size, 4096):
        last_row = min(first_row + 4096, node_ids.size)
        start, end = int(W.indptr[first_row]), int(W.indptr[last_row])
        columns = W.indices[start:end]
        retinal_edges = retinal_sources[columns] & np.repeat(
            optic[first_row:last_row], np.diff(W.indptr[first_row:last_row + 1]),
        )
        retina_raw_mass += np.bincount(
            columns[retinal_edges], weights=W.data[start:end][retinal_edges], minlength=node_ids.size,
        )
        W.data[start:end] *= np.repeat(
            np.float32(0.90) / row_sum[first_row:last_row],
            np.diff(W.indptr[first_row:last_row + 1]),
        )
        # W stores nonnegative magnitudes; transmitter signs stay separate.
        # Use effective retinal -> optic conductance in the actual simulator,
        # not raw syn_count whose scale was removed by row normalization.
        retina_mass += np.bincount(
            columns[retinal_edges], weights=W.data[start:end][retinal_edges], minlength=node_ids.size,
        )
    in_degree = np.diff(W.indptr).astype(np.int32, copy=False)
    out_degree = np.bincount(W.indices, minlength=node_ids.size).astype(np.int32)
    LOG.info("Connectome: %d nodes, %d float32 CSR connections", node_ids.size, W.nnz)
    del ann, root, ann_index, matched, dst, row_sum, columns, retinal_edges, retinal_sources
    gc.collect()
    return Graph(
        node_ids=node_ids, W=W, x=x, y=y, source_sign=source_sign,
        source_known=source_known, in_degree=in_degree, out_degree=out_degree,
        retina_type=retina_type, coordinate_origin=tuple(coordinate_origin),
        coordinate_span=tuple(coordinate_span), **masks,
        retina_synaptic_mass=retina_mass.astype(np.float32),
        retina_raw_synaptic_mass=retina_raw_mass.astype(np.float32),
    )


@dataclass(slots=True)
class Retina:
    current: np.ndarray
    left_energy: float
    right_energy: float
    contrast: float
    edge_density: float
    brightness: float
    saturation: float
    colorfulness: float
    warmth: float
    feature_hash: str
    edge_orientation: str
    edge_strength: float
    edge_coherence: float
    directional_signal: bool = True
    mirror_contrast: float = 1.0
    darkened_fraction: float = 0.0
    escape_eligible: bool = False
    frame_interval_ms: float | None = None


class RetinaEncoder:

    # Input-event criteria are model parameters, not measured fly physiology.
    darkening_area_threshold = 0.65
    darkening_absolute_drop = np.float32(0.25)
    darkening_relative_drop = np.float32(0.50)
    max_frame_interval_ms = 100.0

    def __init__(self, graph: Graph, size: int = 32, gain: float = 0.90) -> None:
        self.graph = graph
        self.height = self.width = int(size)
        self.gain = np.float32(gain)
        if self.width < 4 or self.width % 2:
            raise ValueError("retina size must be even and at least 4")
        if np.any(graph.optic_left & graph.optic_right):
            raise ValueError("Left/right optic annotations must be disjoint")
        self.maps = self._make_maps()
        if graph.retina_synaptic_mass is None:
            raise ValueError("Retinal calibration requires synaptic masses from load_graph")
        self.pool_scales: dict[str, np.float32] = {}
        self.pool_masses = {
            name: float(graph.retina_synaptic_mass[ids].sum(dtype=np.float64))
            for name, (ids, _) in self.maps.items()
        }
        target_sizes = {"left": int(graph.optic_left.sum()), "right": int(graph.optic_right.sum())}
        self.channel_masses = {
            channel: sum(mass for name, mass in self.pool_masses.items() if name.startswith(channel + "_"))
            for channel in ("left_on", "right_on", "left_off", "right_off")
        }
        for name, (ids, _) in self.maps.items():
            side, polarity, cell_type = name.split("_")
            opposite = f"{'right' if side == 'left' else 'left'}_{polarity}_{cell_type}"
            if opposite not in self.maps:
                raise ValueError(f"Retina channel {name} has no matched opposite pool")
            mass, other_mass = self.pool_masses[name], self.pool_masses[opposite]
            if not (math.isfinite(mass) and math.isfinite(other_mass) and min(mass, other_mass) > 0):
                raise ValueError(f"Retina channel {name} has no finite positive synaptic mass")
            other_side = opposite.split("_")[0]
            mass_per_cell = self.channel_masses[f"{side}_{polarity}"] / target_sizes[side]
            other_per_cell = self.channel_masses[f"{other_side}_{polarity}"] / target_sizes[other_side]
            # Match weighted input budgets per optic cell at every luminance.
            # Use one gain for the entire ON or OFF channel: scaling inhibitory
            # L1 independently of excitatory Mi1/Tm3 changes their timing and
            # E/I balance. This includes pool-size correction once. Choosing the
            # smaller mass as reference only attenuates the stronger pool;
            # no side gets an artificial boost toward the firing ceiling.
            self.pool_scales[name] = np.float32(min(mass_per_cell, other_per_cell) / mass_per_cell)
        self.mass_calibration = {
            name: {"effective_mass": self.pool_masses[name],
                   "optic_cells": target_sizes[name.split("_")[0]],
                   "raw_mass": float(graph.retina_raw_synaptic_mass[ids].sum(dtype=np.float64))
                   if graph.retina_raw_synaptic_mass is not None else None,
                   "gain": float(self.pool_scales[name])}
            for name, (ids, _) in self.maps.items()
        }
        self.channel_calibration = {
            channel: {"effective_mass": mass,
                      "optic_cells": target_sizes[channel.split("_")[0]],
                      "gain": float(next(self.pool_scales[name] for name in self.maps
                                         if name.startswith(channel + "_")))}
            for channel, mass in self.channel_masses.items()
        }
        LOG.info("Retina synaptic-mass normalization (retinal -> optic): %s", self.mass_calibration)
        LOG.info("Retina ON/OFF channel budgets: %s", self.channel_calibration)

    @staticmethod
    def _local_coordinates(values: np.ndarray) -> np.ndarray:
        # Preserve relative anatomical positions within a cell type and side.
        # Soma/position coordinates provide a topographic approximation, not
        # measured visual receptive fields or an optic-column registration.
        if not np.all(np.isfinite(values)):
            raise ValueError("Retina cells need finite soma/position coordinates")
        lo, hi = float(values.min()), float(values.max())
        if hi <= lo:
            raise ValueError("Retina cell coordinates have no spatial extent")
        return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    def _make_maps(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        graph = self.graph
        maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        # Equal-width, disjoint image halves; paired population gains are
        # applied after sampling, independently for each annotated cell type.
        half = self.width // 2
        for side, side_mask, offset in (
            ("left", graph.optic_left, 0),
            ("right", graph.optic_right, half),
        ):
            for polarity, channel_mask in (("on", graph.retina_on), ("off", graph.retina_off)):
                selected = side_mask & channel_mask
                if not np.any(selected):
                    raise ValueError(f"Annotations have no {side} {polarity.upper()} retina cells")
                for cell_type in np.unique(graph.retina_type[selected]):
                    ids = np.flatnonzero(selected & (graph.retina_type == cell_type)).astype(np.int32)
                    local_x = self._local_coordinates(graph.x[ids])
                    local_y = self._local_coordinates(graph.y[ids])
                    sample_x = np.float32(offset) + local_x * np.float32(half - 1)
                    sample_y = local_y * np.float32(self.height - 1)
                    coordinates = np.stack((sample_y, sample_x)).astype(np.float32)
                    maps[f"{side}_{polarity}_{int(cell_type)}"] = (ids, coordinates)
        LOG.info("Retina projections: %s", {name: len(ids) for name, (ids, _) in maps.items()})
        return maps

    def _image(self, image: Any) -> tuple[np.ndarray, np.ndarray]:
        from PIL import Image

        if isinstance(image, (bytes, bytearray, memoryview)):
            image = Image.open(io.BytesIO(bytes(image))).convert("RGB")
        if hasattr(image, "convert") and hasattr(image, "resize"):
            image = image.convert("RGB").resize((self.width, self.height))
            rgb = np.asarray(image, dtype=np.float32)
        else:
            original = np.asarray(image)
            normalized_input = np.issubdtype(original.dtype, np.floating) or original.dtype == np.bool_
            array = np.array(original, dtype=np.float32, copy=True)
            if array.ndim not in (2, 3):
                raise ValueError("image must be grayscale/RGB array or encoded bytes")
            if not array.size:
                raise ValueError("image cannot be empty")
            array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
            if normalized_input and float(array.max(initial=0.0)) <= 1.0:
                array *= 255.0
            if array.ndim == 2:
                image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).convert("RGB")
            else:
                if array.shape[2] == 1:
                    array = np.repeat(array, 3, axis=2)
                if array.shape[2] < 3:
                    raise ValueError("RGB image must have three channels")
                image = Image.fromarray(np.clip(array[..., :3], 0, 255).astype(np.uint8)).convert("RGB")
            rgb = np.asarray(image.resize((self.width, self.height)), dtype=np.float32)
        rgb = np.nan_to_num(rgb / np.float32(255.0), nan=0.0, posinf=1.0, neginf=0.0)
        # Camera RGB supplies a luminance surrogate, not measured fly spectral responses.
        fly_signal = np.clip(
            np.float32(0.05) * rgb[..., 0]
            + np.float32(0.55) * rgb[..., 1]
            + np.float32(0.40) * rgb[..., 2],
            0.0,
            1.0,
        )
        return fly_signal.astype(np.float32, copy=False), rgb.astype(np.float32, copy=False)

    def encode(self, image: Any, *, previous_image: Any = None, frame_interval_ms: float = 33.0) -> Retina:
        import hashlib
        from scipy.ndimage import map_coordinates

        image, rgb = self._image(image)
        dog = gaussian_filter(image, 0.9, mode="reflect") - gaussian_filter(image, 2.6, mode="reflect")
        # A static frame is presented from a dark baseline: its luminance
        # onset drives ON cells, while spatial negative DoG drives OFF cells.
        # Smooth compression retains contrast differences between boundaries.
        on = (
            np.float32(0.55) * image
            + np.float32(0.85) * np.tanh(np.float32(4.0) * np.maximum(dog, 0.0))
        ).astype(np.float32)
        off = (np.float32(0.85) * np.tanh(np.float32(4.0) * np.maximum(-dog, 0.0))).astype(np.float32)
        darkened_fraction = 0.0
        escape_eligible = False
        interval = None
        if previous_image is not None:
            interval = float(frame_interval_ms)
            if not math.isfinite(interval) or interval <= 0.0:
                raise ValueError("frame_interval_ms must be finite and positive")
            prior, _ = self._image(previous_image)
            drop = np.maximum(prior - image, np.float32(0.0))
            darkened = ((drop >= self.darkening_absolute_drop)
                        & (drop >= self.darkening_relative_drop * prior))
            darkened_fraction = float(darkened.mean())
            adjacent = interval <= self.max_frame_interval_ms
            escape_eligible = adjacent and darkened_fraction > self.darkening_area_threshold
            if adjacent:
                # A real luminance decrement supplies OFF current through the
                # annotated retinal cells, never directly to DNp01.
                off += np.float32(0.85) * np.tanh(np.float32(4.0) * drop)
        mid = self.width // 2
        # Reject common visual input before reading anatomical handedness as a
        # direction. A calibrated final response cannot balance every transient:
        # almost identical ON/OFF fields need the same 7% tolerance as steering.
        mirror_difference = sum(
            float(np.abs(channel[:, :mid] - channel[:, mid:][:, ::-1]).sum(dtype=np.float64))
            for channel in (on, off)
        )
        mirror_total = sum(float(channel.sum(dtype=np.float64)) for channel in (on, off))
        mirror_contrast = mirror_difference / max(1e-6, mirror_total)
        directional_signal = mirror_contrast >= DIRECTION_DEADZONE
        current = np.zeros(self.graph.n, dtype=np.float32)
        for name, (ids, coordinates) in self.maps.items():
            signal = on if "_on_" in name else off
            sampled = map_coordinates(signal, coordinates, order=1, mode="nearest", prefilter=False)
            # The same smooth bounded transfer precedes mass normalization on
            # both sides. No brightness-specific knots, gain changes or clips.
            current[ids] = np.tanh(sampled) * self.gain * self.pool_scales[name]
        gradient_y, gradient_x = np.gradient(image)
        horizontal_change = float(np.mean(np.abs(gradient_x)))
        vertical_change = float(np.mean(np.abs(gradient_y)))
        edge_strength = horizontal_change + vertical_change
        edge_coherence = abs(horizontal_change - vertical_change) / (edge_strength + 1e-8)
        edge_orientation = (
            "none" if edge_strength < 0.002 else
            "mixed" if edge_coherence < 0.25 else
            "vertical" if horizontal_change > vertical_change else "horizontal"
        )
        return Retina(
            current=current,
            left_energy=float(np.abs(dog[:, :mid]).mean()),
            right_energy=float(np.abs(dog[:, mid:]).mean()),
            contrast=float(np.std(image)),
            edge_density=float(np.mean(np.abs(dog) > np.float32(0.035))),
            brightness=float(np.mean(image)),
            saturation=float(np.mean(np.max(rgb, axis=2) - np.min(rgb, axis=2))),
            colorfulness=float(np.clip(np.mean(np.std(rgb, axis=(0, 1))) * 2.2, 0.0, 1.0)),
            warmth=float(np.clip(0.5 + 0.5 * (float(rgb[..., 0].mean()) - float(rgb[..., 2].mean())), 0.0, 1.0)),
            feature_hash=hashlib.sha256(np.rint(rgb * np.float32(255.0)).astype(np.uint8).tobytes()).hexdigest(),
            edge_orientation=edge_orientation,
            edge_strength=edge_strength,
            edge_coherence=edge_coherence,
            directional_signal=directional_signal,
            mirror_contrast=mirror_contrast,
            darkened_fraction=darkened_fraction,
            escape_eligible=escape_eligible,
            frame_interval_ms=interval,
        )


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.dtype == np.bool_ and b.dtype == np.bool_:
        denom = math.sqrt(int(np.count_nonzero(a)) * int(np.count_nonzero(b)))
        return float(np.count_nonzero(a & b)) / denom if denom else 0.0
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b)) / denom if denom else 0.0


class CriticSimulator:

    stimulus_steps = 6
    steering_deadzone = DIRECTION_DEADZONE
    propagation_gain = np.float32(3.75)
    membrane_decay = np.float32(math.exp(-1.0 / 8.0))
    threshold = np.float32(0.38)
    dnp01_threshold = np.float32(0.50)
    adaptation_decay = np.float32(0.951)
    adaptation_spike = np.float32(0.16)
    resource_recovery = np.float32(0.10)
    depression_factor = np.float32(0.35)

    def __init__(self, graph: Graph, max_snapshot_nodes: int = 250) -> None:
        self.graph = graph
        self.retina = RetinaEncoder(graph)
        self.max_snapshot_nodes = max(1, min(250, int(max_snapshot_nodes)))
        # Reconstruct original coordinate scales for display only; the retinal
        # coordinates remain bit-for-bit unchanged. The sagittal reference is
        # halfway between optic population medians, with one physical scale
        # for both sides and axes. +Y up is a display convention, not an inferred
        # anterior/dorsal orientation absent from the supplied annotations.
        view_x = graph.x.astype(np.float64) * graph.coordinate_span[0] + graph.coordinate_origin[0]
        view_y = graph.y.astype(np.float64) * graph.coordinate_span[1] + graph.coordinate_origin[1]
        self._view_valid = np.isfinite(view_x) & np.isfinite(view_y)
        if not np.any(self._view_valid):
            raise ValueError("Brain projection requires finite soma/position coordinates")
        left = self._view_valid & graph.optic_left
        right = self._view_valid & graph.optic_right
        if not np.any(left) or not np.any(right):
            raise ValueError("Brain projection requires coordinates in both optic populations")
        median_left = float(np.median(view_x[left]))
        median_right = float(np.median(view_x[right]))
        midline_x = 0.5 * (median_left + median_right)
        center_y = float(np.median(view_y[self._view_valid]))
        extent_x = float(np.max(np.abs(view_x[self._view_valid] - midline_x)))
        extent_y = float(np.max(np.abs(view_y[self._view_valid] - center_y)))
        pixels_per_unit = min(
            0.45 * 960.0 / extent_x if extent_x > 0.0 else math.inf,
            0.45 * 600.0 / extent_y if extent_y > 0.0 else math.inf,
        )
        if not math.isfinite(pixels_per_unit):
            pixels_per_unit = 1.0
        self._view_x = (0.5 + (view_x - midline_x) * pixels_per_unit / 960.0).astype(np.float32)
        self._view_y = (0.5 - (view_y - center_y) * pixels_per_unit / 600.0).astype(np.float32)
        self._view_x[~self._view_valid] = np.nan
        self._view_y[~self._view_valid] = np.nan
        LOG.info(
            "Brain display projection: %d pairs; origin=%s span=%s; sagittal X=%.3f, center Y=%.3f; "
            "optic X medians=(%.3f, %.3f); common pixels/unit=%.6f; +X right, +Y up (display convention)",
            int(np.count_nonzero(self._view_valid)), graph.coordinate_origin, graph.coordinate_span,
            midline_x, center_y, median_left, median_right, pixels_per_unit,
        )

    def _brain_snapshot(self, active: np.ndarray, *, step: int | None = None) -> dict[str, Any]:
        """Sample actual active cells in proportion to their anatomical pools.

        Spikes are binary in this model. Ranking membrane overshoot and then
        using it as brightness hid much of the opposite optic lobe. Keep the
        same point budget, use spatially distributed representatives, and
        report the complete population counts separately from the sample.
        """
        g = self.graph
        counts = {
            "optic_left": int(np.count_nonzero(active & g.optic_left)),
            "optic_right": int(np.count_nonzero(active & g.optic_right)),
            "other": int(np.count_nonzero(active & ~(g.optic_left | g.optic_right))),
            "unmapped": int(np.count_nonzero(active & ~self._view_valid)),
            "total": int(np.count_nonzero(active)),
        }
        ids = np.flatnonzero(active & self._view_valid)
        pools = (
            ids[g.optic_left[ids]],
            ids[g.optic_right[ids]],
            ids[~(g.optic_left[ids] | g.optic_right[ids])],
        )
        sizes = np.asarray([pool.size for pool in pools], dtype=np.int64)
        budget = min(self.max_snapshot_nodes, ids.size)
        quotas = sizes.copy()
        if ids.size > budget:
            exact = sizes * (budget / ids.size)
            quotas = np.floor(exact).astype(np.int64)
            remaining = budget - int(quotas.sum())
            order = np.argsort(-(exact - quotas), kind="stable")
            quotas[order[:remaining]] += 1
        # Step-dependent positions within each spatial interval prevent the
        # same root IDs from monopolizing the view. The final overview uses
        # interval midpoints and covers the entire recruited population.
        phase = 0.5 if step is None else ((step % 7) + 0.5) / 7.0
        nodes: list[dict[str, Any]] = []
        shown: dict[str, int] = {}
        for name, pool, quota in zip(("optic_left", "optic_right", "other"), pools, quotas):
            quota = int(quota)
            shown[name] = quota
            if not quota:
                continue
            order = np.lexsort((pool, self._view_y[pool], self._view_x[pool]))
            positions = np.floor((np.arange(quota) + phase) * pool.size / quota).astype(np.int64)
            selected = pool[order[positions]]
            for i in selected:
                nodes.append({
                    "id": int(i), "x": float(self._view_x[i]), "y": float(self._view_y[i]),
                    "a": 1.0, "group": name, "weight": float(pool.size / quota),
                })
        return {
            "nodes": nodes, "counts": counts, "sampled_counts": shown,
            "mode": "spikes" if step is not None else "recruited",
        }

    def _decode(
        self,
        history: np.ndarray,
        counts: np.ndarray,
        retina: Retina,
        currents: np.ndarray,
        memory_trace: np.ndarray,
    ) -> dict[str, Any]:
        """Read annotated circuits; image descriptors never choose a drive/state.

        Drive scales are model heuristics, not behavioral probabilities. Escape
        needs an explicitly observed rapid darkening and actual circuit firing;
        a still image alone cannot establish looming or recognize food.
        """
        g = self.graph
        steps, n = history.shape
        recruited_mask = np.any(history, axis=0)
        recruited = int(recruited_mask.sum())
        raw_memory = int(np.count_nonzero(memory_trace > np.float32(0.20)))
        group_activity: dict[str, dict[str, Any]] = {}
        strength: dict[str, float] = {}
        for name, mask in g.groups.items():
            size = int(mask.sum())
            if size:
                wave = history[:, mask]
                coverage = float(recruited_mask[mask].mean())
                rate = float(wave.mean())
                trace = float(memory_trace[mask].mean())
                # Smooth, size-normalized readout: recruitment, firing and memory.
                activation = float(-math.expm1(-(4.0 * coverage + 6.0 * rate + 2.0 * trace)))
                group_activity[name] = {
                    "size": size,
                    "recruited": int(recruited_mask[mask].sum()),
                    "spikes": int(wave.sum()),
                    "coverage": round(coverage, 6),
                    "spike_rate": round(rate, 6),
                    "peak_rate": round(float(wave.mean(axis=1).max()), 6),
                    "late_rate": round(float(wave[-6:].mean()), 6),
                    "memory_trace": round(trace, 6),
                    "activation": round(activation * 100.0, 3),
                }
            else:
                activation = 0.0
                group_activity[name] = dict(size=0, recruited=0, spikes=0, coverage=0.0,
                                            spike_rate=0.0, peak_rate=0.0, late_rate=0.0,
                                            memory_trace=0.0, activation=0.0)
            strength[name] = activation

        # Compare lags beyond the absolute refractory period. bool dot is not a
        # spike intersection; _cosine explicitly counts overlapping active cells.
        resonance_by_lag = [
            float(np.mean([_cosine(history[t - lag], history[t])
                           for t in range(max(lag, steps - 8), steps)]))
            for lag in (2, 3, 4, 5)
        ]
        resonance = max(resonance_by_lag, default=0.0)
        active_steps = np.flatnonzero(counts)
        active_span = int(active_steps[-1] - active_steps[0] + 1) if active_steps.size else 0
        span_ratio = active_span / steps
        peak_count = int(counts.max(initial=0))
        persistence = float(np.clip(float(counts[-6:].mean()) / max(1, peak_count) * 4.0, 0, 1))
        post_input_fraction = float(np.mean(counts[self.stimulus_steps:] > 0))
        recurrence = 0.50 * persistence + 0.25 * resonance + 0.25 * post_input_fraction
        nonvisual = ~g.optic
        nonvisual_coverage = float(recruited_mask[nonvisual].mean()) if nonvisual.any() else 0.0
        propagation = float(-math.expm1(-nonvisual_coverage / 0.01))
        optic_wave = history[:, g.optic].mean(axis=1)
        visual_rise = float(np.maximum(np.diff(optic_wave, prepend=0.0), 0.0).max())
        visual_surge = float(-math.expm1(-visual_rise / 0.04))
        early_decay = 1.0 - 0.5 * span_ratio - 0.5 * persistence

        # Temporal visual evidence opens the circuit; it does not manufacture
        # DNp01 spikes or replace the measured neural contribution to the drive.
        drive_escape = (100.0 * min(
            1.0, 1.2 * strength["dnp01"] * strength["descending"] * visual_surge,
        ) if retina.escape_eligible else 0.0)
        drive_panic = drive_escape  # Keep existing Telegram/WS statistics compatible.
        dnp01_spike = float(history[:, g.dnp01].mean(axis=1).max(initial=0.0))
        left, right = strength["steering_left"], strength["steering_right"]
        raw_dna_asymmetry = (right - left) / max(1e-8, left + right)
        # Use spike density, not total pool size or the nonlinear activation
        # score. Rates per step give the same ratio as these spikes per cell.
        activity_l = group_activity["optic_left"]["spikes"] / group_activity["optic_left"]["size"]
        activity_r = group_activity["optic_right"]["spikes"] / group_activity["optic_right"]["size"]
        raw_visual_asymmetry = (activity_r - activity_l) / max(1e-6, activity_l + activity_r)
        steering_enabled = (retina.directional_signal
                            and abs(raw_visual_asymmetry) >= self.steering_deadzone)
        visual_asymmetry = raw_visual_asymmetry if steering_enabled else 0.0
        signed_asymmetry = raw_dna_asymmetry if steering_enabled else 0.0
        dna_turn_gain = (100.0 * min(1.0, 1.5 * abs(right - left) * math.sqrt(max(left, right)))
                         if steering_enabled else 0.0)
        # Read optic *activity*, not pixel energy. Ignore asymmetries below 20%
        # and fade sensory support out once DNa alone reaches the motor threshold.
        visual_excess = float(np.clip((abs(visual_asymmetry) - 0.20) / 0.80, 0.0, 1.0))
        visual_turn_gain = (100.0 * visual_excess * strength["optic"]
                            * max(0.0, 1.0 - dna_turn_gain / 25.0))
        signed_turn = float(np.clip(
            math.copysign(dna_turn_gain, right - left)
            + math.copysign(visual_turn_gain, visual_asymmetry), -100.0, 100.0,
        ))
        turn_gain = abs(signed_turn)
        drive_left, drive_right = max(0.0, -signed_turn), max(0.0, signed_turn)
        turn_source = ("mixed" if dna_turn_gain > 0.0 and visual_turn_gain > 0.0 else
                       "visual" if visual_turn_gain > 0.0 else
                       "dna" if dna_turn_gain > 0.0 else "none")
        food_activity = 0.55 * strength["food"] + 0.45 * strength["feeding_motor"]
        food_group = group_activity["food"]
        food_tail = min(1.0, 4.0 * food_group["late_rate"] / max(1e-8, food_group["peak_rate"]))
        food_recent = float(np.mean(memory_trace[g.food] > np.float32(0.20)))
        food_retention = min(1.0, food_recent / max(1e-8, food_group["coverage"]))
        food_continuation = 0.75 * food_tail + 0.25 * food_retention
        drive_food = 100.0 * food_activity * recurrence * food_continuation
        drive_apathy = 100.0 * (0.60 * (1.0 - propagation) + 0.40 * early_decay) * (
            1.0 - 0.30 * strength["optic"]
        )
        motor = {"panic": drive_panic, "turn_left": drive_left,
                 "turn_right": drive_right, "food_interest": drive_food}
        # Subthreshold escape must not crowd out an eligible turn/food response.
        eligible_motor = {**motor, "panic": drive_escape if drive_escape >= 65.0 else 0.0}
        motor_winner = max(eligible_motor, key=eligible_motor.get)
        motor_peak = eligible_motor[motor_winner]
        drive_passive = 100.0 * (0.25 + 0.75 * strength["optic"]) * (
            1.0 - motor_peak / 100.0
        ) * (1.0 - drive_apathy / 100.0)
        drives = {**motor, "apathy": drive_apathy, "passive_observation": drive_passive}
        # An optic-only directional tendency is diagnostic, not proof of motor
        # firing. Active vision with silent behavioral readouts stays observation.
        readouts_silent = all(group_activity[name]["spikes"] == 0
                              for name in ("dnp01", "steering", "food"))
        if strength["optic"] > 0.20 and readouts_silent:
            dominant = "passive_observation"
        elif motor_peak >= 25.0 and motor_peak > drive_apathy:
            dominant = motor_winner
        elif drive_apathy >= 75.0:
            dominant = "apathy"
        else:
            dominant = "passive_observation"
        turn_direction = "none"
        if turn_gain >= 25.0:
            turn_direction = "right" if signed_turn > 0.0 else "left"
        # This is response strength, not aesthetic quality or object semantics.
        engagement = (0.50 * strength["optic"] + 0.25 * strength["central_complex"]
                      + 0.15 * propagation + 0.10 * recurrence)
        score = 1.0 + 9.0 * engagement
        peak_step = int(np.argmax(counts)) + 1 if peak_count else 0
        stats = {
            "score": round(score, 1), "dominant": dominant, "dominant_state": dominant,
            "behavior": dominant, "total_nodes": n, "total_recruited": recruited,
            "memory_active": raw_memory, "resonance_rate": round(resonance * 100.0, 2),
            "turn_direction": turn_direction,
            # Directional readout: raw DNa handedness is suppressed on neutral input.
            "steering_yaw": round(signed_asymmetry, 5),
            "dnp01_spike": dnp01_spike,
            "escape_event": retina.escape_eligible and dnp01_spike > 0.0,
            "escape_eligible": retina.escape_eligible,
            "darkened_fraction": retina.darkened_fraction,
            "frame_interval_ms": retina.frame_interval_ms,
            "dnp01_threshold": float(self.dnp01_threshold),
            "turn_source": turn_source,
            "dna_turn_drive": round(dna_turn_gain, 2),
            "visual_turn_drive": round(visual_turn_gain, 2),
            "hemisphere_asymmetry": round(abs(signed_asymmetry) * 100.0, 2),
            "steering_asymmetry": round(signed_asymmetry, 5),
            "dna_asymmetry": round(signed_asymmetry, 5),
            "visual_asymmetry": round(visual_asymmetry, 5),
            "directional_input": retina.directional_signal,
            "retina_mirror_contrast": retina.mirror_contrast,
            "steering_enabled": steering_enabled,
            "steering_deadzone": self.steering_deadzone,
            "activity_l": activity_l,
            "activity_r": activity_r,
            "raw_dna_asymmetry": round(raw_dna_asymmetry, 5),
            "raw_visual_asymmetry": round(raw_visual_asymmetry, 5),
            "left_activity": group_activity["optic_left"]["spikes"],
            "right_activity": group_activity["optic_right"]["spikes"],
            "left_retina_energy": round(retina.left_energy, 6),
            "right_retina_energy": round(retina.right_energy, 6),
            "retina_current_left": float(retina.current[g.optic_left].sum(dtype=np.float64)),
            "retina_current_right": float(retina.current[g.optic_right].sum(dtype=np.float64)),
            # Legacy scalar adjustments are identities; actual gains now come
            # exclusively from the per-pool synaptic-mass normalization.
            "retina_right_compensation": 1.0,
            "retina_right_midgray_compensation": 1.0,
            "retina_normalization": {"method": "synaptic_mass", "channels": self.retina.channel_calibration,
                                     "pools": self.retina.mass_calibration},
            "active_span": active_span, "span_ratio": round(span_ratio, 5),
            "peak_step": peak_step, "spike_counts": counts.tolist(),
            "recurrent_drive": round(recurrence, 5), "persistence": round(persistence, 5),
            "propagation": round(propagation, 5), "visual_surge": round(visual_surge, 5),
            "food_continuation": round(food_continuation, 5),
            "drive_panic": round(drive_panic, 2), "drive_apathy": round(drive_apathy, 2),
            "drive_escape": round(drive_escape, 2),
            "drive_turn": round(turn_gain, 2), "drive_food": round(drive_food, 2),
            "peak_energy": round(float(currents.max(initial=0.0)), 6),
            "peak_rate": round(peak_count / n, 6), "final_rate": round(int(counts[-1]) / n, 6),
            "retina_contrast": round(retina.contrast, 5), "edge_density": round(retina.edge_density, 5),
            "brightness": round(retina.brightness, 5), "saturation": round(retina.saturation, 5),
            "colorfulness": round(retina.colorfulness, 5), "warmth": round(retina.warmth, 5),
            "edge_orientation": retina.edge_orientation, "edge_coherence": round(retina.edge_coherence, 5),
            "feature_hash": retina.feature_hash,
            "drives": {key: round(float(value), 2) for key, value in drives.items()},
            "group_activity": group_activity,
        }
        stats["thought"] = self._thought(stats)
        stats["brain_snapshot"] = self._brain_snapshot(recruited_mask)
        LOG.info("Response %s: drives=%s; recruited=%d; peak step=%d", dominant,
                 stats["drives"], recruited, peak_step)
        return stats

    @staticmethod
    def _thought(stats: dict[str, Any]) -> str:
        import hashlib

        phrases = {
            "panic": (
                "Зрительный всплеск дошёл до контура ухода — хочется отступить.",
                "Контур DNp01 отозвался на зрительную волну: возникает импульс к уходу.",
                "Резкий зрительный отклик включил реакцию ухода.",
                "Волна затронула клетки ухода; первым возникает порыв отдалиться.",
                "После зрительного пика выделился контур ухода.",
                "Сейчас сильнее всего сигнал к уходу, поддержанный нисходящими клетками.",
            ),
            "apathy": (
                "Волна слаба: выраженного движения не возникает.",
                "Отклик быстро теряет силу, продолжать движение незачем.",
                "Сигнал почти не распространяется за зрительные контуры.",
                "Устойчивого продолжения у этой волны мало.",
                "Зрительный вход оставляет слабый след в сети.",
                "Моторные контуры молчат или отвечают слишком слабо.",
            ),
            "turn_left": (
                "Левые DNa-клетки отвечают сильнее: возникает доворот влево.",
                "Волна даёт левый моторный перевес.",
                "Слева steering-контур активнее; тянет повернуться туда.",
                "Нисходящий ответ смещён влево — намечается левый поворот.",
                "Левый контур поворота выделился на фоне остальных.",
                "В ответе DNa устойчивее левая сторона: движение склоняется влево.",
            ),
            "turn_right": (
                "Правые DNa-клетки отвечают сильнее: возникает доворот вправо.",
                "Волна даёт правый моторный перевес.",
                "Справа steering-контур активнее; тянет повернуться туда.",
                "Нисходящий ответ смещён вправо — намечается правый поворот.",
                "Правый контур поворота выделился на фоне остальных.",
                "В ответе DNa устойчивее правая сторона: движение склоняется вправо.",
            ),
            "food_interest": (
                "Устойчивый отклик затронул пищевые контуры — возникает исследовательский импульс.",
                "Пищевые группы удерживают часть волны после исчезновения входа.",
                "Рекуррентный ответ поддержал клетки пищевого контура.",
                "Волна продолжилась в группах, связанных с питанием.",
                "Пищевой контур заметно отвечает и сохраняет след.",
                "Среди откликнувшихся групп выделяются клетки питания с продолженным ответом.",
            ),
            "passive_observation": (
                "Зрительная волна идёт, а явного моторного победителя нет.",
                "Отклик распределён по сети; выраженного поворота или ухода не возникает.",
                "Зрительные контуры работают без сильной команды движению.",
                "Волна оставляет след, но пока не выделяет моторное направление.",
                "Наблюдение продолжается: ни один моторный контур не получил перевеса.",
                "Сеть отвечает на структуру изображения без выраженной команды к действию.",
            ),
        }
        selection = json.dumps({"image": stats["feature_hash"], "state": stats["dominant"],
                                "drives": stats["drives"], "peak": stats["peak_step"]},
                               sort_keys=True, separators=(",", ":")).encode()
        choice = int.from_bytes(hashlib.sha256(selection).digest()[:8], "big")
        base = phrases[stats["dominant"]][choice % len(phrases[stats["dominant"]])]
        if stats["dominant"] in ("turn_left", "turn_right") and stats["visual_turn_drive"] > 0.0:
            direction = "влево" if stats["dominant"] == "turn_left" else "вправо"
            sensory_phrases = (
                "Зрительная асимметрия поддерживает тенденцию повернуться {direction}.",
                "Разница зрительных ответов усиливает направленный отклик {direction}.",
                "Зрительный вклад смещает общий драйв поворота {direction}.",
                "С учётом зрительной активности сильнее тенденция {direction}.",
                "Неравномерный зрительный ответ добавляет импульс к повороту {direction}.",
                "Общий направленный ответ склоняется {direction} при поддержке зрительного контура.",
            )
            base = sensory_phrases[choice % len(sensory_phrases)].format(direction=direction)
        if not stats["total_recruited"]:
            wave = "Спайковой волны нет"
        elif stats["persistence"] > 0.35:
            wave = f"Волна сохраняется до конца, пик на такте {stats['peak_step']}"
        elif stats["active_span"] < 12:
            wave = f"Короткая волна: {stats['active_span']} тактов"
        else:
            wave = f"Волна затухает после пика на такте {stats['peak_step']}"
        if stats["resonance_rate"] > 20:
            resonance = "с заметным повторением активных ансамблей"
        elif stats["resonance_rate"] > 0:
            resonance = "со слабым повторением активных ансамблей"
        else:
            resonance = "без позднего повторения ансамблей"
        side = stats["steering_asymmetry"]
        laterality = ("DNa смещены вправо" if side > 0.10 else
                      "DNa смещены влево" if side < -0.10 else "DNa без выраженного крена")
        light = "тёмный" if stats["brightness"] < 0.20 else "светлый" if stats["brightness"] > 0.70 else "средней яркости"
        color = "насыщенный цветом" if stats["saturation"] > 0.35 else "сдержанный по цвету"
        edges = ("с плотной сетью границ" if stats["edge_density"] > 0.30 else
                 "с отдельными границами" if stats["edge_density"] > 0.03 else "с плавной структурой")
        visual_side = stats["visual_asymmetry"]
        visual_laterality = ("зрительный ответ сильнее справа" if visual_side > 0.10 else
                             "зрительный ответ сильнее слева" if visual_side < -0.10 else
                             "зрительный ответ почти симметричен")
        orientation = {"vertical": ", преобладает вертикальная структура",
                       "horizontal": ", преобладает горизонтальная структура"}.get(stats["edge_orientation"], "")
        return (f"{base} {wave}, {resonance}; {visual_laterality}, {laterality}. "
                f"Стимул {light}, {color}, {edges}{orientation}.")

    async def evaluate(self, image: Any, callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None, steps: int = 30, *, previous_image: Any = None, frame_interval_ms: float = 33.0) -> dict[str, Any]:
        """Evaluate one trial; previous_image must be an explicitly adjacent frame.

        Trials share no image history. Separate Telegram photos therefore cannot
        produce an invented motion/darkening event. Timing describes the supplied
        frame pair, not a conversion of simulation ticks to biological time.
        """
        steps = max(25, min(30, int(steps)))
        retina = self.retina.encode(image, previous_image=previous_image, frame_interval_ms=frame_interval_ms)
        n = self.graph.n
        v = np.zeros(n, dtype=np.float32)
        adaptation = np.zeros(n, dtype=np.float32)
        refractory = np.zeros(n, dtype=np.uint8)
        relative_refractory = np.zeros(n, dtype=np.float32)
        resource = np.ones(n, dtype=np.float32)
        streak = np.zeros(n, dtype=np.uint8)
        silence = np.zeros(n, dtype=np.uint8)
        previous = np.zeros(n, dtype=np.float32)
        memory_trace = np.zeros(n, dtype=np.float32)
        history = np.zeros((steps, n), dtype=bool)
        counts = np.zeros(steps, dtype=np.int32)
        currents = np.zeros(steps, dtype=np.float32)
        optic_left_size = int(self.graph.optic_left.sum())
        optic_right_size = int(self.graph.optic_right.sum())
        visual_spikes_left = visual_spikes_right = 0
        thresholds = np.full(n, self.threshold, dtype=np.float32)
        # Sensory gating is applied before reset/history/recurrent propagation,
        # not by hiding already-fired motor neurons in the final report.
        thresholds[self.graph.dnp01] = self.dnp01_threshold if retina.escape_eligible else np.float32(np.inf)

        for t in range(steps):
            current = self.propagation_gain * self.graph.W.dot(
                previous * resource * self.graph.source_sign
            )
            if t < self.stimulus_steps:
                current += retina.current
            currents[t] = np.mean(np.abs(current))
            v = self.membrane_decay * v + current - adaptation
            adaptation *= self.adaptation_decay
            relative_refractory *= np.float32(math.exp(-1.0 / 4.0))
            refractory = np.maximum(refractory.astype(np.int16) - 1, 0).astype(np.uint8)
            spikes = (v >= thresholds + adaptation + np.float32(0.18) * relative_refractory) & (refractory == 0)
            v[spikes] = np.float32(0.08)
            adaptation[spikes] += self.adaptation_spike
            relative_refractory[spikes] += np.float32(1.0)
            refractory[spikes] = 2
            silence = np.where(spikes, 0, np.minimum(silence + 1, 3)).astype(np.uint8)
            streak = np.where(spikes, np.minimum(streak.astype(np.uint16) + 1, 255), streak).astype(np.uint8)
            streak[silence >= 3] = 0
            resource += self.resource_recovery * (1.0 - resource)
            resource[spikes & (streak >= 3)] *= self.depression_factor
            previous = spikes.astype(np.float32)
            memory_trace = np.maximum(np.float32(0.88) * memory_trace, previous)
            history[t] = spikes
            counts[t] = int(spikes.sum())
            if callback is not None:
                snapshot = self._brain_snapshot(spikes, step=t)
                dna_l = float(spikes[self.graph.steering_left].mean())
                dna_r = float(spikes[self.graph.steering_right].mean())
                spikes_l = int(np.count_nonzero(spikes[self.graph.optic_left]))
                spikes_r = int(np.count_nonzero(spikes[self.graph.optic_right]))
                optic_l = spikes_l / optic_left_size
                optic_r = spikes_r / optic_right_size
                visual_spikes_left += spikes_l
                visual_spikes_right += spikes_r
                # Accumulated visual evidence survives the delay to motor
                # neurons; one late optic spike cannot reopen a neutral trial.
                activity_l = visual_spikes_left / optic_left_size
                activity_r = visual_spikes_right / optic_right_size
                raw_dna_asymmetry = (dna_r - dna_l) / max(1e-8, dna_l + dna_r)
                raw_visual_asymmetry = (activity_r - activity_l) / max(1e-6, activity_l + activity_r)
                steering_enabled = (retina.directional_signal
                                    and abs(raw_visual_asymmetry) >= self.steering_deadzone)
                dna_asymmetry = raw_dna_asymmetry if steering_enabled else 0.0
                visual_asymmetry = raw_visual_asymmetry if steering_enabled else 0.0
                await callback({
                    "type": "step", "step": t + 1, "total_steps": steps,
                    "energy": round(float(counts[t] / n), 6),
                    "dna_l": dna_l,
                    "dna_r": dna_r,
                    "dnp01": float(spikes[self.graph.dnp01].mean()),
                    "escape_event": retina.escape_eligible and bool(np.any(spikes[self.graph.dnp01])),
                    "escape_eligible": retina.escape_eligible,
                    "darkened_fraction": retina.darkened_fraction,
                    "frame_interval_ms": retina.frame_interval_ms,
                    "optic_l": optic_l,
                    "optic_r": optic_r,
                    "steering_yaw": dna_asymmetry * min(1.0, (dna_l + dna_r) / 0.10),
                    "dna_asymmetry": dna_asymmetry,
                    "visual_asymmetry": visual_asymmetry,
                    "directional_input": retina.directional_signal,
                    "steering_enabled": steering_enabled,
                    "steering_deadzone": self.steering_deadzone,
                    "activity_l": activity_l,
                    "activity_r": activity_r,
                    "raw_dna_asymmetry": raw_dna_asymmetry,
                    "raw_visual_asymmetry": raw_visual_asymmetry,
                    "nodes": snapshot["nodes"],
                    "brain_counts": snapshot["counts"],
                })
            await asyncio.sleep(0.01 if callback is not None else 0)

        return self._decode(history, counts, retina, currents, memory_trace)

    async def evaluate_image(self, image: Any, callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None, steps: int = 30, *, previous_image: Any = None, frame_interval_ms: float = 33.0) -> dict[str, Any]:
        return await self.evaluate(image, callback, steps, previous_image=previous_image, frame_interval_ms=frame_interval_ms)


FlyBrain = CriticSimulator


class Hub:

    def __init__(self) -> None:
        self.clients: dict[WebSocket, asyncio.Queue[str]] = {}
        self.tasks: dict[WebSocket, asyncio.Task[None]] = {}
        self.lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=2)
        async with self.lock:
            self.clients[websocket] = queue
            self.tasks[websocket] = asyncio.create_task(self._sender(websocket, queue))

    async def _sender(self, websocket: WebSocket, queue: asyncio.Queue[str]) -> None:
        try:
            while True:
                await websocket.send_text(await queue.get())
        except Exception:
            await self.disconnect(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self.lock:
            self.clients.pop(websocket, None)
            task = self.tasks.pop(websocket, None)
        if task and task is not asyncio.current_task():
            task.cancel()

    async def send(self, payload: dict[str, Any]) -> None:
        text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        for websocket, queue in tuple(self.clients.items()):
            try:
                queue.put_nowait(text)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(text)
                except asyncio.QueueFull:
                    pass

    async def close(self) -> None:
        for websocket in tuple(self.clients):
            await self.disconnect(websocket)


dp = Dispatcher()
hub = Hub()
simulator: CriticSimulator | None = None
load_lock: asyncio.Lock | None = None
run_lock = asyncio.Lock()


async def get_simulator() -> CriticSimulator:
    global simulator, load_lock
    if simulator is not None:
        return simulator
    if load_lock is None:
        load_lock = asyncio.Lock()
    async with load_lock:
        if simulator is None:
            LOG.info("Loading %s", CONNECTOME_PATH)
            graph = await asyncio.to_thread(load_graph, CONNECTOME_PATH)
            simulator = CriticSimulator(graph, MAX_SNAPSHOT_NODES)
            LOG.info("Ready: %d nodes, %d CSR synapses", graph.n, graph.W.nnz)
    return simulator


def telegram_text(stats: dict[str, Any]) -> str:
    labels = {
        "panic": "УХОД / ESCAPE-КОНТУР",
        "apathy": "СЛАБЫЙ ОТКЛИК / АПАТИЯ",
        "turn_left": "ДОВОРОТ ВЛЕВО",
        "turn_right": "ДОВОРОТ ВПРАВО",
        "food_interest": "ПИЩЕВОЙ КОНТУР / РЕЗОНАНС",
        "passive_observation": "ПАССИВНОЕ НАБЛЮДЕНИЕ",
    }
    drives = stats["drives"]
    groups = stats["group_activity"]
    group_labels = {
        "optic": "Optic/visual", "optic_left": "Visual L", "optic_right": "Visual R",
        "descending": "Descending", "dnp01": "DNp01",
        "steering_left": "DNa L", "steering_right": "DNa R",
        "central_complex": "CX", "food": "Пищевые группы",
        "gustatory": "Gustatory", "feeding_motor": "Пищевая моторика",
        "sez": "SEZ-NSC", "grn": "GRN", "fbn": "FBn",
    }
    group_lines = []
    for name, label in group_labels.items():
        group = groups[name]
        if not group["size"]:
            group_lines.append(f"• {label}: нет в аннотациях")
        else:
            group_lines.append(
                f"• {label}: {group['activation']:.1f}/100; "
                f"{group['recruited']}/{group['size']} клеток; "
                f"след {group['memory_trace']:.3f}"
            )
    return (
        f"🦟 Сила отклика: {stats['score']:.1f} / 10\n"
        f"Реакция: {labels[stats['dominant']]}\n\n"
        f"{stats['thought']}\n\n"
        "Драйвы (0–100, не вероятности):\n"
        f"• Уход: {drives['panic']:.2f} | апатия: {drives['apathy']:.2f}\n"
        f"• Поворот L/R: {drives['turn_left']:.2f}/{drives['turn_right']:.2f}\n"
        f"  Вклад DNa: {stats['dna_turn_drive']:.2f}; зрительный: {stats['visual_turn_drive']:.2f}\n"
        f"• Пищевой: {drives['food_interest']:.2f} | наблюдение: {drives['passive_observation']:.2f}\n\n"
        "Группы (активация; рекрутирование; средний след):\n"
        + "\n".join(group_lines)
        + f"\n\nВсего: {stats['total_recruited']}/{stats['total_nodes']} клеток; "
        f"финальная память: {stats['memory_active']}.\n"
        f"Резонанс: {stats['resonance_rate']:.2f}%; "
        f"волна: {stats['active_span']} тактов, пик: {stats['peak_step']}; "
        f"направленная асимметрия DNa: {100 * stats['steering_asymmetry']:+.1f}%; "
        f"зрительная: {100 * stats['visual_asymmetry']:+.1f}%.\n"
        f"Сырая асимметрия графа DNa / Visual: "
        f"{100 * stats['raw_dna_asymmetry']:+.1f}% / {100 * stats['raw_visual_asymmetry']:+.1f}%.\n"
        f"Visual L/R, спайков на клетку: {stats['activity_l']:.4f} / {stats['activity_r']:.4f}; "
        f"мёртвая зона руля: {100 * stats['steering_deadzone']:.0f}%.\n"
        "Токи сетчатки нормализованы по синаптической массе ON/OFF-пулов.\n"
        + ("Временное затемнение не измерено: один статичный кадр.\n"
           if stats["frame_interval_ms"] is None else
           f"Контрастное затемнение: {100 * stats['darkened_fraction']:.1f}% поля за "
           f"{stats['frame_interval_ms']:.1f} мс; разрешение ухода: "
           f"{'да' if stats['escape_eligible'] else 'нет'}.\n")
        + "Это модель зрительного и моторного отклика, без распознавания объектов."
    )


HTML_PAGE = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FlyWire — зрительный и моторный отклик</title>
<style>
:root{color-scheme:dark;--bg:#080d13;--panel:#0e1721;--line:#233344;--ink:#e2ecf2;--muted:#859aab;--cyan:#65e0ec;--gold:#d8b779}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:13px ui-monospace,SFMono-Regular,Consolas,monospace}main{max-width:1250px;margin:auto;padding:24px}header{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:0 0 20px}.brand{font-size:16px;letter-spacing:.09em}.brand small{display:block;font-size:10px;color:var(--muted);letter-spacing:.15em;margin-top:6px}.live{display:flex;align-items:center;gap:14px;flex-wrap:wrap;color:var(--muted)}#status{color:var(--cyan);font-size:11px;letter-spacing:.09em}.dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:currentColor;margin-right:7px}button{border:1px solid var(--line);background:var(--panel);color:var(--ink);font:inherit;padding:8px 12px;border-radius:5px;cursor:pointer}button:hover{border-color:var(--cyan)}button:focus-visible{outline:2px solid var(--cyan);outline-offset:3px}.visuals{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(330px,1fr);gap:14px}.panel{border:1px solid var(--line);border-radius:9px;overflow:hidden;background:var(--panel)}.panel-title{display:flex;justify-content:space-between;align-items:center;gap:8px;padding:14px 16px;border-bottom:1px solid var(--line);font-size:11px;letter-spacing:.08em}.panel-title span{color:var(--muted);font-size:10px;letter-spacing:0}canvas{display:block;width:100%;height:auto}.brain-area{position:relative;background:#060b12;min-height:300px;height:calc(100% - 44px);display:flex;align-items:center}#field{aspect-ratio:960/600}.brain-caption{position:absolute;bottom:12px;left:16px;right:16px;display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:8px 12px;font-size:10px;color:var(--muted);pointer-events:none}.fly-wrap{background:radial-gradient(ellipse at 50% 48%,#132330,#0a121b 70%);display:flex;justify-content:center}#flyCanvas{width:min(100%,360px);height:auto;aspect-ratio:1}.motor-state{display:flex;justify-content:space-between;align-items:center;padding:0 16px 14px;gap:12px}.motor-state strong{font-size:11px;color:var(--cyan)}#flyAngle{color:var(--gold);white-space:nowrap}.telemetry{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px 18px;padding:13px 16px;border-top:1px solid var(--line);font-size:10px;color:var(--muted)}.meter-label{display:flex;justify-content:space-between;gap:6px}.meter-label b{font-weight:400;color:var(--ink)}.meter{height:3px;background:#20303d;border-radius:2px;margin-top:7px;overflow:hidden}.meter i{display:block;height:100%;width:0;background:var(--cyan);transition:width .08s linear}.meter.motor i{background:var(--gold)}.fly-foot{padding:11px 16px;border-top:1px solid var(--line);font-size:10px;line-height:1.7;color:var(--muted);display:flex;justify-content:space-between;gap:10px}.fly-foot b{font-weight:400;color:var(--ink)}.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-top:14px}.cell{padding:13px 15px;border:1px solid var(--line);border-radius:7px;background:var(--panel);min-width:0}.cell small{color:var(--muted);font-size:9px;letter-spacing:.05em}.value{display:block;color:var(--cyan);margin-top:8px;min-height:1.2em;overflow-wrap:anywhere}.reading{margin-top:14px;padding:17px 18px}.reading p{margin:0;line-height:1.8}.reading p:empty{display:none}.reading p:empty+details{margin-top:0}.reading details{margin-top:14px;color:var(--muted)}summary{cursor:pointer;font-size:11px}pre{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.7;font:11px ui-monospace,monospace;color:var(--ink)}.note{font-size:10px;color:var(--muted);line-height:1.8;margin:16px 2px 0}.key{display:inline-block;width:6px;height:6px;background:var(--cyan);border-radius:50%;margin-right:5px}
@media(max-width:820px){main{padding:15px}.visuals{grid-template-columns:1fr}.brain-area{min-height:0;height:auto}.stats{grid-template-columns:repeat(2,minmax(0,1fr))}.fly-wrap{max-height:360px}.brand{font-size:13px}.live{gap:8px}}@media(prefers-reduced-motion:reduce){.meter i{transition:none}}
</style>
</head>
<body>
<main>
<header><div class="brand">FLYWIRE <span style="color:var(--cyan)">783</span><small>ЗРИТЕЛЬНЫЙ И МОТОРНЫЙ ОТКЛИК</small></div><div class="live"><b id="status" role="status">ПОДКЛЮЧЕНИЕ</b><button id="pause" type="button" aria-pressed="false" aria-label="Приостановить визуализацию">Пауза</button></div></header>
<div class="visuals">
<article class="panel" aria-label="Проекция коннектома мозга"><div class="panel-title">01 / КОННЕКТОМ <span>Такт <b id="step">0/0</b></span></div><div class="brain-area"><canvas id="field" width="960" height="600" aria-label="Активные нейроны: анатомическая проекция FlyWire">Проекция активности нейронов. Числовая статистика находится ниже.</canvas><div class="brain-caption"><span><i class="key"></i><span id="brainMode">Спайки на такте</span></span><span id="brainCounts">Visual L — · R —</span><span>Энергия <b id="energy">0.00000</b></span></div></div></article>
<article class="panel" aria-label="Моторный отклик тела дрозофилы"><div class="panel-title">02 / ДРОЗОФИЛА <span>Вид сверху</span></div><div class="fly-wrap"><canvas id="flyCanvas" width="360" height="360" aria-label="Дрозофила: поворот по DNa, глаза по optic, отскок по DNp01">Схематическая дрозофила. Угол, моторный статус и зрительная асимметрия указаны ниже.</canvas></div><div class="motor-state"><strong id="motorStatus" role="status">ПОКОЙ</strong><span id="flyAngle">0.0°</span></div><div class="telemetry"><div><div class="meter-label">DNa L <b id="dnaL">0.00%</b></div><div class="meter motor"><i id="dnaLBar"></i></div></div><div><div class="meter-label">DNa R <b id="dnaR">0.00%</b></div><div class="meter motor"><i id="dnaRBar"></i></div></div><div><div class="meter-label">Visual L <b id="opticL">0.00%</b></div><div class="meter"><i id="opticLBar"></i></div></div><div><div class="meter-label">Visual R <b id="opticR">0.00%</b></div><div class="meter"><i id="opticRBar"></i></div></div></div><div class="fly-foot"><span>DNp01 <b id="flyEscape">0.00%</b></span><span>Зрительная асимметрия <b id="flyVisualAsymmetry">0.0%</b></span></div></article>
</div>
<section class="stats" aria-label="Итоговая статистика"><div class="cell"><small>СИЛА ОТКЛИКА</small><span class="value" id="score">—</span></div><div class="cell"><small>ДОМИНАНТА</small><span class="value" id="dominant">—</span></div><div class="cell"><small>КРЕН / ИСТОЧНИК</small><span class="value" id="turn">—</span></div><div class="cell"><small>РЕКРУТИРОВАНО</small><span class="value" id="recruited">—</span></div><div class="cell"><small>ПАМЯТЬ</small><span class="value" id="memory">—</span></div><div class="cell"><small>РЕЗОНАНС</small><span class="value" id="resonance">—</span></div><div class="cell"><small>АСИММЕТРИЯ DNa</small><span class="value" id="asymmetry">—</span></div><div class="cell"><small>ЗРИТЕЛЬНАЯ АСИММЕТРИЯ</small><span class="value" id="visualAsymmetry">—</span></div></section>
<div class="reading panel"><p id="thought"></p><details><summary>Драйвы и активность функциональных групп</summary><pre id="drives">Ожидание изображения из Telegram.</pre></details></div>
<p class="note">Отправьте изображение Telegram-боту: волна проигрывается по тактам. На проекции показана выборка клеток; подписи L/R содержат полные численности. Итоговая проекция показывает рекрутированные клетки за весь стимул. Тело — схема моторного отклика коннектома; поворот задаёт DNa, подсветку глаз — optic. Отскок требует резкого контрастного затемнения более 65% поля между соседними кадрами и отклика DNp01. Одиночная фотография не подтверждает такое событие. Модель не распознаёт объекты.</p>
</main>
<script>
'use strict';
const $ = id => document.getElementById(id);
const c = $('field'), x = c.getContext('2d');
const flyCanvas = $('flyCanvas'), f = flyCanvas.getContext('2d');
const clamp = (value, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, value));
const number = value => Number.isFinite(Number(value)) ? Number(value) : 0;
const fraction = value => clamp(number(value));
const percent = value => `${(100 * number(value)).toFixed(2)}%`;
const labels = {panic:'Уход',turn_left:'Поворот влево',turn_right:'Поворот вправо',food_interest:'Пищевой интерес',passive_observation:'Наблюдение',apathy:'Апатия'};
const trail = new Map(), queue = [];
let brainViewMode = 'step';
const tickMillis = 80, maxQueue = 60, motorIdleMillis = 650;
let running = false, paused = false, connected = false, nextTick = 0, phase = 'ОЖИДАНИЕ';
let angle = 0, targetAngle = 0, eyeL = 0, eyeR = 0, targetEyeL = 0, targetEyeR = 0;
let escapePulse = 0, pulseClock = 0, hadEscape = false, lastFrame = 0;
let currentMotor = {dna_l:0,dna_r:0,dnp01:0,optic_l:0,optic_r:0};
let currentVisualAsymmetry = 0, motorAgeMillis = motorIdleMillis;
let socket, reconnectTimer;
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
function setPhase(value) { phase = value; $('status').textContent = paused && connected ? 'ПАУЗА' : value; }
function updatePause() {
    $('pause').textContent = paused ? 'Продолжить' : 'Пауза';
    $('pause').setAttribute('aria-pressed', String(paused));
    $('pause').setAttribute('aria-label', paused ? 'Продолжить визуализацию' : 'Приостановить визуализацию');
    setPhase(phase);
}
$('pause').addEventListener('click', () => { paused = !paused; nextTick = 0; updatePause(); });
function resetMotion() {
    // Reset the target, preserving the rendered pose for a smooth return.
    targetAngle = targetEyeL = targetEyeR = currentVisualAsymmetry = 0;
    motorAgeMillis = motorIdleMillis;
    hadEscape = false;
    currentMotor = {dna_l:0,dna_r:0,dnp01:0,optic_l:0,optic_r:0};
    updateTelemetry();
}
function updateTelemetry() {
    for (const [name, key] of [['dnaL','dna_l'],['dnaR','dna_r'],['opticL','optic_l'],['opticR','optic_r']]) {
        $(name).textContent = percent(currentMotor[key]);
        $(name + 'Bar').style.width = `${100 * currentMotor[key]}%`;
    }
    const asymmetry = Number((currentVisualAsymmetry * 100).toFixed(1));
    $('flyVisualAsymmetry').textContent = `${asymmetry > 0 ? '+' : ''}${asymmetry.toFixed(1)}%`;
    $('flyEscape').textContent = percent(currentMotor.dnp01);
}
function triggerEscape() { escapePulse = 1; hadEscape = true; }
function loadBrainSnapshot(nodes, counts, mode = 'step') {
    // Each point set describes one population: the current tick or all
    // recruited cells. Mixing decaying ticks would bias the result to its tail.
    trail.clear();
    brainViewMode = mode;
    for (const n of (Array.isArray(nodes) ? nodes : []).slice(0, 5000)) {
        trail.set(n.id, {x:clamp(number(n.x)),y:clamp(number(n.y)),a:fraction(n.a),life:1});
    }
    $('brainMode').textContent = mode === 'recruited'
        ? 'Рекрутированные клетки за стимул' : 'Спайки на такте';
    const count = value => Math.round(Math.max(0, number(value))).toLocaleString('ru-RU');
    $('brainCounts').textContent = counts
        ? `Visual L ${count(counts.optic_left)} · R ${count(counts.optic_right)}` : 'Visual L — · R —';
    $('brainCounts').title = counts
        ? `Полные численности: всего ${count(counts.total)}, другие ${count(counts.other)}, без координат ${count(counts.unmapped)}`
        : 'Численности не переданы';
}
function showStep(d) {
    if (d.step === 1) {
        trail.clear(); resetMotion();
        for (const id of ['score','dominant','turn','recruited','memory','resonance','asymmetry','visualAsymmetry']) $(id).textContent = '—';
        $('thought').textContent = ''; $('drives').textContent = 'Вычисляется отклик…';
    }
    running = true; setPhase('ВОЛНА');
    $('step').textContent = `${d.step}/${d.total_steps}`;
    $('energy').textContent = number(d.energy).toFixed(5);
    loadBrainSnapshot(d.nodes, d.brain_counts);
    for (const key of Object.keys(currentMotor)) currentMotor[key] = fraction(d[key]);
    const {dna_l:l, dna_r:r, dnp01:escape, optic_l:ol, optic_r:or} = currentMotor;
    const sum = l + r;
    const hasYaw = d.steering_yaw != null && Number.isFinite(Number(d.steering_yaw));
    const supplied = key => d[key] != null && Number.isFinite(Number(d[key]));
    const hasVisual = supplied('visual_asymmetry') || (supplied('activity_l') && supplied('activity_r'))
        || (supplied('optic_l') && supplied('optic_r'));
    // Integrated spike densities are normalized by each anatomical pool;
    // legacy optic_l/r packets already carry per-node instantaneous rates.
    const vl = supplied('activity_l') ? Math.max(0, number(d.activity_l)) : ol;
    const vr = supplied('activity_r') ? Math.max(0, number(d.activity_r)) : or;
    const visualSum = vl + vr;
    const visualAsymmetry = supplied('visual_asymmetry')
        ? clamp(number(d.visual_asymmetry), -1, 1) : visualSum > 0 ? (vr - vl) / visualSum : 0;
    const steeringDeadzone = supplied('steering_deadzone')
        ? clamp(number(d.steering_deadzone), 0.07, 1) : 0.07;
    const straightAhead = hasVisual && Math.abs(visualAsymmetry) < steeringDeadzone;
    currentVisualAsymmetry = straightAhead ? 0 : visualAsymmetry;
    const legacyYaw = sum > 0.002 ? (r - l) / sum * clamp((sum - 0.002) / 0.025) : 0;
    // A calibrated zero is authoritative even if the anatomical populations
    // contain different numbers of active neurons. Apply the same deadzone
    // to older senders; raw instantaneous meters remain raw.
    const yaw = straightAhead ? 0 : hasYaw ? clamp(number(d.steering_yaw), -1, 1) : legacyYaw;
    targetAngle = yaw * Math.PI / 4;
    motorAgeMillis = 0;
    // The nonlinear display scale makes small population fractions visible;
    // numerical meters retain the original fraction of spiking cells.
    targetEyeL = 1 - Math.exp(-25 * ol); targetEyeR = 1 - Math.exp(-25 * or);
    // Raw DNp01 activity is diagnostic. Only a verified temporal escape event
    // moves the body; missing legacy event fields still require explicit eligibility.
    const escapeEvent = d.escape_event === true
        || (d.escape_event === undefined && d.escape_eligible === true && escape > 0);
    if (escapeEvent && !hadEscape) triggerEscape();
    updateTelemetry();
}
function showResult(d) {
    running = false; setPhase('ГОТОВО');
    const s = d.stats || {}, dominant = d.dominant || s.dominant || '';
    if (s.brain_snapshot && Array.isArray(s.brain_snapshot.nodes)) {
        loadBrainSnapshot(s.brain_snapshot.nodes, s.brain_snapshot.counts, 'recruited');
    } else {
        $('brainMode').textContent = 'Спайки последнего такта';
    }
    // Integrated result metrics describe the completed wave, not a motor
    // command that should keep rotating the resting body indefinitely.
    resetMotion();
    $('score').textContent = `${number(s.score).toFixed(1)} / 10`;
    $('dominant').textContent = labels[dominant] || dominant || '—';
    $('turn').textContent = `${s.turn_direction || '—'} / ${s.turn_source || '—'}`;
    $('recruited').textContent = s.total_recruited ?? '—';
    $('memory').textContent = s.memory_active ?? '—';
    $('resonance').textContent = `${number(s.resonance_rate).toFixed(1)}%`;
    const steering = s.steering_asymmetry === undefined ? number(s.hemisphere_asymmetry) : 100 * number(s.steering_asymmetry);
    $('asymmetry').textContent = `${steering.toFixed(1)}%`;
    $('visualAsymmetry').textContent = `${(100 * number(s.visual_asymmetry)).toFixed(1)}%`;
    $('thought').textContent = s.thought || '';
    $('drives').textContent = JSON.stringify({
        drives:s.drives, group_activity:s.group_activity,
        visual_density:{unit:'спайков / клетку за стимул', left:s.activity_l, right:s.activity_r},
        directional_input:s.directional_input,
        raw_asymmetry:{dna:s.raw_dna_asymmetry, visual:s.raw_visual_asymmetry},
        escape:{event:s.escape_event, eligible:s.escape_eligible,
            darkened_fraction:s.darkened_fraction, frame_interval_ms:s.frame_interval_ms,
            dnp01_spike:s.dnp01_spike},
    }, null, 2);
}
function enqueue(d) {
    if (d.type !== 'step' && d.type !== 'result') return;
    // Bound memory even if paused or the tab is hidden. Under overload,
    // discard the oldest intermediate ticks before completed results.
    if (queue.length >= maxQueue) {
        const intermediate = queue.findIndex(item => item.type === 'step' && item.step !== 1);
        queue.splice(intermediate >= 0 ? intermediate : 0, 1);
    }
    queue.push(d);
}
function connect() {
    socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
    socket.onopen = () => { connected = true; setPhase('ОЖИДАНИЕ'); };
    socket.onmessage = event => {
        try { enqueue(JSON.parse(event.data)); }
        catch (error) { console.warn('Invalid FlyWire message', error); }
    };
    socket.onclose = () => {
        connected = false; running = false; queue.length = 0; nextTick = 0;
        resetMotion(); setPhase('НЕТ СОЕДИНЕНИЯ'); updatePause();
        clearTimeout(reconnectTimer); reconnectTimer = setTimeout(connect, 2000);
    };
    socket.onerror = () => { socket.close(); };
}
function ellipse(context, cx, cy, rx, ry, fill, stroke, rotation = 0) {
    context.beginPath(); context.ellipse(cx, cy, rx, ry, rotation, 0, Math.PI * 2);
    if (fill) { context.fillStyle = fill; context.fill(); }
    if (stroke) { context.strokeStyle = stroke; context.stroke(); }
}
function drawBrain(dt) {
    x.fillStyle = '#060b12'; x.fillRect(0, 0, c.width, c.height);
    x.strokeStyle = '#11202c'; x.lineWidth = 1;
    for (let gx = 0; gx <= c.width; gx += 60) { x.beginPath(); x.moveTo(gx,0); x.lineTo(gx,c.height); x.stroke(); }
    for (let gy = 0; gy <= c.height; gy += 60) { x.beginPath(); x.moveTo(0,gy); x.lineTo(c.width,gy); x.stroke(); }
    x.strokeStyle = '#263d49'; x.setLineDash([5,7]);
    x.beginPath(); x.moveTo(c.width/2,0); x.lineTo(c.width/2,c.height); x.stroke(); x.setLineDash([]);
    x.fillStyle = '#526e81'; x.font = '13px ui-monospace,monospace';
    x.textAlign = 'left'; x.fillText('L',16,22);
    x.textAlign = 'right'; x.fillText('R',c.width-16,22);
    for (const n of trail.values()) {
        // Modern packets use a=1 for binary spikes: every sampled cell has
        // equal visual weight. Older amplitude packets retain their scaling.
        const alpha = clamp(0.1 + n.a);
        ellipse(x,n.x*c.width,n.y*c.height,1.4+n.a*3.2,1.4+n.a*3.2,`rgba(101,224,236,${alpha})`);
    }
    if (!trail.size) {
        x.fillStyle = '#506778'; x.textAlign = 'center'; x.font = '15px ui-monospace,monospace';
        const message = brainViewMode === 'recruited' ? 'Нет рекрутированных клеток'
            : running ? 'Нет спайков на этом такте' : 'Ожидание зрительного стимула';
        x.fillText(message, c.width/2, c.height/2);
    }
}
function drawWing(side, spread) {
    f.save(); f.scale(side,1); f.translate(10,-18); f.rotate(-spread * 0.72);
    f.beginPath(); f.moveTo(0,0); f.bezierCurveTo(29,-4,72,25,68,57); f.bezierCurveTo(61,80,21,61,0,7);
    f.fillStyle = 'rgba(181,217,227,.16)'; f.strokeStyle = 'rgba(178,216,228,.58)'; f.lineWidth = 1.2; f.fill(); f.stroke();
    f.strokeStyle = 'rgba(186,219,227,.24)'; f.lineWidth = 0.7;
    for (let i = 0; i < 3; i++) { f.beginPath(); f.moveTo(3,3); f.quadraticCurveTo(22+i*9,16,27+i*17,57+i*2); f.stroke(); }
    f.beginPath(); f.moveTo(17,18); f.lineTo(46,21); f.lineTo(39,46); f.lineTo(62,44); f.stroke(); f.restore();
}
function drawEye(side, activation) {
    const blue = Math.round(87 + 161*activation), green = Math.round(84 + 159*activation);
    f.save(); f.translate(side*21,-46); f.rotate(side*0.19); f.lineWidth = 1;
    f.shadowColor = '#64f6ff'; f.shadowBlur = 16*activation;
    ellipse(f,0,0,9.5,15.5,`rgb(${Math.round(18+72*activation)},${green},${blue})`,'#6bc2c4');
    f.shadowBlur = 0; f.clip(); f.fillStyle = `rgba(190,255,255,${0.17+activation*0.35})`;
    for (let row=-4; row<=4; row++) for (let col=-3; col<=3; col++) {
        const xx=col*4+(row%2)*2, yy=row*3.5;
        f.beginPath();
        for (let k=0;k<6;k++) { const a=k*Math.PI/3; const px=xx+1.2*Math.cos(a),py=yy+1.2*Math.sin(a); if(k===0) f.moveTo(px,py); else f.lineTo(px,py); }
        f.closePath(); f.fill();
    }
    f.restore();
}
function drawFly(dt) {
    if (!paused) {
        const ease = 1-Math.pow(0.8,dt);
        angle += (targetAngle-angle)*ease; eyeL += (targetEyeL-eyeL)*ease; eyeR += (targetEyeR-eyeR)*ease;
        if (targetAngle === 0 && Math.abs(angle) < 0.000001) angle = 0;
        escapePulse *= Math.pow(0.93,dt); if (escapePulse < 0.008) escapePulse=0;
        pulseClock += dt/60;
    }
    const jump=escapePulse, recoil=(reducedMotion?8:27)*jump;
    f.clearRect(0,0,360,360); f.lineWidth=1;
    for (const radius of [54,104,148]) ellipse(f,180,180,radius,radius,null,'rgba(111,150,167,.10)');
    f.strokeStyle='rgba(111,150,167,.14)';
    for (let i=0;i<4;i++) { f.save(); f.translate(180,180); f.rotate(i*Math.PI/2); f.beginPath(); f.moveTo(0,-143); f.lineTo(0,-154); f.stroke(); f.restore(); }
    f.fillStyle='#526e81'; f.font='10px ui-monospace,monospace'; f.textAlign='center'; f.fillText('L',27,184); f.fillText('R',333,184);
    ellipse(f,180,208,42,62,'rgba(0,0,0,.19)');
    // The sprite faces -Y at angle zero. Canvas positive rotation therefore
    // turns the head clockwise/right, with no extra -PI/2 offset.
    f.save(); f.translate(180,181); f.rotate(angle); f.translate(0,recoil);
    if (jump>0.02) {
        f.lineWidth=1.5+2*jump; f.shadowColor='#ff595e'; f.shadowBlur=18*jump;
        const redAlpha=jump*(reducedMotion?0.65:0.5+0.28*Math.sin(pulseClock*28));
        ellipse(f,0,2,72+8*jump,97+8*jump,null,`rgba(255,89,94,${redAlpha})`); f.shadowBlur=0;
    }
    f.lineWidth=2.3; f.lineCap='round'; f.lineJoin='round';
    for (const side of [-1,1]) for (let i=0;i<3;i++) {
        const y=-21+i*21, sway=(jump*8)*(i-1);
        f.strokeStyle='#7b8d86'; f.beginPath(); f.moveTo(side*13,y); f.lineTo(side*(32+i*3),y-15+sway); f.lineTo(side*(54+i*2),y+(i-1)*17+sway); f.lineTo(side*(63+i*2),y+(i-1)*21+7+sway); f.stroke();
        ellipse(f,side*(32+i*3),y-15+sway,2,2,'#a5b4a4');
    }
    const abdomen=f.createLinearGradient(-25,0,25,0); abdomen.addColorStop(0,'#6b593a'); abdomen.addColorStop(0.5,'#c1a268'); abdomen.addColorStop(1,'#635337');
    f.lineWidth=1.4; ellipse(f,0,32,24,46,abdomen,'#cab477');
    f.save(); f.clip(); f.strokeStyle='#493e2d'; f.lineWidth=5;
    for (const y of [10,25,40,54,66]) { f.beginPath(); f.moveTo(-28,y-3); f.quadraticCurveTo(0,y+8,28,y-3); f.stroke(); } f.restore();
    // Wings sit above the abdomen and below the thorax, as seen from above.
    drawWing(-1,jump); drawWing(1,jump);
    const thorax=f.createLinearGradient(-18,-15,18,-15); thorax.addColorStop(0,'#5e614d'); thorax.addColorStop(0.5,'#ada878'); thorax.addColorStop(1,'#525743');
    ellipse(f,0,-12,18,25,thorax,'#b2b189');
    f.strokeStyle='rgba(56,61,46,.7)'; f.lineWidth=2;
    for (const offset of [-6,6]) { f.beginPath(); f.moveTo(offset,-32); f.quadraticCurveTo(offset*0.6,-14,offset,7); f.stroke(); }
    ellipse(f,0,-44,25,19,'#a29460','#c2b787'); drawEye(-1,eyeL); drawEye(1,eyeR);
    f.strokeStyle='#b6aa81'; f.lineWidth=1.4;
    for (const side of [-1,1]) { f.beginPath(); f.moveTo(side*5,-59); f.lineTo(side*11,-68); f.lineTo(side*17,-71); f.stroke(); }
    ellipse(f,0,-57,3,3,'#655c40');
    const moving=Math.abs(targetAngle)>0.01;
    f.strokeStyle=jump>0.08?'#ff7476':moving?'#e6c482':'#648797'; f.lineWidth=2;
    f.beginPath(); f.moveTo(0,-83); f.lineTo(0,-113); f.moveTo(-6,-106); f.lineTo(0,-114); f.lineTo(6,-106); f.stroke();
    f.restore();
    const rawDegrees=angle*180/Math.PI, degrees=Math.abs(rawDegrees)<0.05?0:rawDegrees;
    $('flyAngle').textContent=`${degrees>0?'+':''}${degrees.toFixed(1)}°`;
    const motor=jump>0.08?'ПРЫЖОК DNp01':targetAngle<-.01?'ДОВОРОТ L':targetAngle>.01?'ДОВОРОТ R':'ПОКОЙ';
    if ($('motorStatus').textContent!==motor) $('motorStatus').textContent=motor;
    $('motorStatus').style.color=jump>0.08?'#ff7476':'var(--cyan)';
}
function frame(now) {
    const dt=lastFrame?clamp((now-lastFrame)/16.667,0.1,3):1; lastFrame=now;
    if (!paused && queue.length && now>=nextTick) {
        const d=queue.shift(); if(d.type==='step') showStep(d); else showResult(d);
        nextTick=now+tickMillis;
    }
    if (!paused && motorAgeMillis < motorIdleMillis) {
        // Count playback time, not paused wall time. This also recovers if a
        // sender stops after a step without ever delivering a result packet.
        motorAgeMillis += dt * 16.667;
        if (motorAgeMillis >= motorIdleMillis && !queue.length) resetMotion();
    }
    drawBrain(dt); drawFly(dt); requestAnimationFrame(frame);
}
resetMotion(); connect(); requestAnimationFrame(frame);
</script>
</body>
</html>"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    await get_simulator()
    bot: Bot | None = None
    polling: asyncio.Task[Any] | None = None
    resolver: ThreadedResolver | None = None
    if BOT_TOKEN:
        resolver = ThreadedResolver()
        session = AiohttpSession()
        session._connector_init["resolver"] = resolver
        bot = Bot(token=BOT_TOKEN, session=session)
        polling = asyncio.create_task(dp.start_polling(bot, handle_signals=False))
        LOG.info("Telegram polling enabled")
    else:
        LOG.info("Telegram polling disabled: TELEGRAM_BOT_TOKEN is empty")
    try:
        yield
    finally:
        if polling is not None:
            polling.cancel()
            try:
                await polling
            except asyncio.CancelledError:
                pass
        if bot is not None:
            await bot.session.close()
        if resolver is not None:
            await resolver.close()
        await hub.close()


app = FastAPI(title="FlyWire image critic", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    return HTML_PAGE


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await hub.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await hub.disconnect(websocket)


async def run_image(data: bytes) -> dict[str, Any]:
    brain = await get_simulator()

    async def emit(frame: dict[str, Any]) -> None:
        await hub.send(frame)

    async with run_lock:
        stats = await brain.evaluate(data, emit, SIM_STEPS)
        await hub.send({
            "type": "result", "stats": stats,
            "steering_yaw": stats["steering_yaw"],
            "dnp01_spike": stats["dnp01_spike"],
            "escape_event": stats["escape_event"],
            "dominant": stats["dominant"],
        })
        return stats


async def handle_image(message: Message, file_id: str) -> None:
    status = await message.reply("Сетчатка получила изображение, запускаю волну...")
    try:
        file_info = await message.bot.get_file(file_id)
        buffer = io.BytesIO()
        await message.bot.download_file(file_info.file_path, destination=buffer)
        stats = await run_image(buffer.getvalue())
        await status.edit_text(telegram_text(stats))
    except Exception as exc:
        LOG.exception("image simulation failed")
        await status.edit_text(f"Ошибка симуляции: {exc}")


@dp.message(F.photo)
async def on_photo(message: Message) -> None:
    await handle_image(message, message.photo[-1].file_id)


@dp.message(F.document)
async def on_document(message: Message) -> None:
    if message.document and (message.document.mime_type or "").startswith("image/"):
        await handle_image(message, message.document.file_id)
    else:
        await message.reply("Отправьте изображение как фото или файл.")


@dp.message()
async def fallback(message: Message) -> None:
    await message.reply("Отправьте изображение для оценки.")


if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
