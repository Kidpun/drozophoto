"""Low-memory connectome simulator for image-driven experiments.

The edge table contains no cell-type or spatial metadata.  The loader therefore
keeps the measured graph weights and derives deterministic pseudo anatomy from
the FlyWire root id.  Replace ``source_sign`` and ``node_xy`` when those
annotations become available.
"""

from __future__ import annotations

import asyncio
import gc
import inspect
import io
import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pyarrow.ipc as pa_ipc
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix


Array = np.ndarray
StepCallback = Callable[[dict[str, Any]], Any]


@dataclass(slots=True)
class ConnectomeGraph:
    """CSR graph and compact node metadata.

    ``W[post, pre]`` is row-normalised by incoming synaptic weight.  The
    matrix is always float32 and node indices are int32, which keeps the graph
    well below the memory budget on an 8 GB MacBook.
    """

    node_ids: Array
    W: csr_matrix
    node_x: Array
    node_y: Array
    source_sign: Array
    in_degree: Array
    out_degree: Array

    @property
    def n_nodes(self) -> int:
        return int(self.node_ids.size)


def _stable_uniform(ids: Array, salt: int = 0) -> Array:
    """SplitMix64 hash mapped to [0, 1), without a global RNG state."""

    x = np.asarray(ids, dtype=np.uint64) + np.uint64(salt)
    x += np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return (x.astype(np.float64) / float(1 << 64)).astype(np.float32)


def load_connectome(path: str, incoming_gain: float = 0.90) -> ConnectomeGraph:
    """Load a Feather edge table into a compact, stable CSR graph.

    Only three columns are materialised.  The temporary arrays are released as
    soon as CSR construction completes; with the supplied 16.8 M edge file
    peak Python-side memory is typically below 1.0 GB.
    """

    if not 0.0 < incoming_gain <= 1.0:
        raise ValueError("incoming_gain must be in (0, 1]")

    # Feather is an Arrow IPC file.  Read 64k-row batches so the unrelated
    # neurotransmitter columns never enter Python memory.
    reader = pa_ipc.open_file(path)
    names = reader.schema.names
    required = ("pre_pt_root_id", "post_pt_root_id", "syn_count")
    if any(name not in names for name in required):
        raise ValueError(f"missing required columns; expected {required}")
    pre_name, post_name, weight_name = required
    node_set: set[int] = set()
    n_edges = 0
    for batch_index in range(reader.num_record_batches):
        batch = reader.get_batch(batch_index)
        pre_batch = np.asarray(batch.column(names.index(pre_name)))
        post_batch = np.asarray(batch.column(names.index(post_name)))
        unique_batch = np.unique(np.concatenate((pre_batch, post_batch)))
        node_set.update(int(value) for value in unique_batch)
        n_edges += batch.num_rows
    if n_edges == 0:
        raise ValueError("connectome table contains no edges")

    node_ids = np.asarray(sorted(node_set), dtype=np.int64)
    del node_set
    n_nodes = int(node_ids.size)
    pre_idx = np.empty(n_edges, dtype=np.int32)
    post_idx = np.empty(n_edges, dtype=np.int32)
    weights = np.empty(n_edges, dtype=np.float32)
    offset = 0
    # Searchsorted is vectorised and only holds one IPC batch at a time.
    for batch_index in range(reader.num_record_batches):
        batch = reader.get_batch(batch_index)
        end = offset + batch.num_rows
        pre_batch = np.asarray(batch.column(names.index(pre_name)))
        post_batch = np.asarray(batch.column(names.index(post_name)))
        weight_batch = np.nan_to_num(
            np.asarray(batch.column(names.index(weight_name)), dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        pre_idx[offset:end] = np.searchsorted(node_ids, pre_batch)
        post_idx[offset:end] = np.searchsorted(node_ids, post_batch)
        weights[offset:end] = weight_batch
        offset = end

    W = csr_matrix(
        (weights, (post_idx, pre_idx)),
        shape=(n_nodes, n_nodes),
        dtype=np.float32,
    )
    W.sum_duplicates()
    W.eliminate_zeros()
    row_sum = np.asarray(W.sum(axis=1)).ravel().astype(np.float32, copy=False)
    row_sum[row_sum <= 0.0] = 1.0
    counts = np.diff(W.indptr)
    W.data /= np.repeat(row_sum, counts)
    W.data *= np.float32(incoming_gain)
    W.sort_indices()

    in_degree = np.bincount(post_idx, minlength=n_nodes).astype(np.int32)
    out_degree = np.bincount(pre_idx, minlength=n_nodes).astype(np.int32)

    # No spatial/cell-type annotations are present in this edge file.  A stable
    # hash is preferable to random coordinates: repeated runs map to the same
    # visual field and E/I groups.  Use ~20% inhibitory presynaptic neurons.
    node_x = _stable_uniform(node_ids, 17)
    node_y = _stable_uniform(node_ids, 43)
    source_sign = np.where(_stable_uniform(node_ids, 71) < 0.20, -1.0, 1.0)
    source_sign = source_sign.astype(np.float32, copy=False)

    del reader, pre_batch, post_batch, weight_batch, weights, pre_idx, post_idx, row_sum, counts
    gc.collect()
    return ConnectomeGraph(
        node_ids=node_ids,
        W=W,
        node_x=node_x,
        node_y=node_y,
        source_sign=source_sign,
        in_degree=in_degree,
        out_degree=out_degree,
    )


@dataclass(slots=True)
class RetinaFrame:
    current: Array
    left_energy: float
    right_energy: float
    center_energy: float
    on_energy: float
    off_energy: float


class RetinaEncoder:
    """Retinotopic ON/OFF DoG encoder with a small optic-chiasm overlap."""

    def __init__(
        self,
        graph: ConnectomeGraph,
        size: tuple[int, int] = (32, 32),
        input_gain: float = 0.80,
        center_fraction: float = 0.16,
    ) -> None:
        height, width = map(int, size)
        if height < 4 or width < 4 or width % 2:
            raise ValueError("retina size must be at least 4x4 and have even width")
        self.height, self.width = height, width
        self.input_gain = np.float32(input_gain)
        self.graph = graph
        self._maps = self._build_maps(center_fraction)

    def _sensory_pool(self, mask: Array) -> Array:
        g = self.graph
        d30 = np.percentile(g.in_degree, 30)
        d50 = np.percentile(g.out_degree, 50)
        pool = np.flatnonzero(mask & (g.in_degree <= d30) & (g.out_degree >= d50))
        if pool.size < 8:
            pool = np.flatnonzero(mask)
        if pool.size == 0:
            pool = np.arange(g.n_nodes, dtype=np.int32)
        return pool.astype(np.int32, copy=False)

    @staticmethod
    def _tile(pool: Array, count: int) -> Array:
        # Evenly spaced assignment avoids a giant random-choice temporary.
        if count <= 0:
            return np.empty(0, dtype=np.int32)
        return pool[np.arange(count, dtype=np.int64) % pool.size].astype(np.int32, copy=False)

    @staticmethod
    def _channel_pools(pool: Array) -> tuple[Array, Array]:
        """Keep ON and OFF sampling disjoint when a pool is large enough."""
        on_pool, off_pool = pool[::2], pool[1::2]
        if on_pool.size == 0:
            on_pool = pool
        if off_pool.size == 0:
            off_pool = pool
        return on_pool, off_pool

    def _build_maps(self, center_fraction: float) -> dict[str, tuple[Array, Array]]:
        g = self.graph
        x = g.node_x
        c0 = 0.5 - float(center_fraction) / 2.0
        c1 = 0.5 + float(center_fraction) / 2.0
        left_nodes = self._sensory_pool(x < c0)
        right_nodes = self._sensory_pool(x > c1)
        chiasm_nodes = self._sensory_pool((x >= c0) & (x <= c1))

        yy, xx = np.indices((self.height, self.width))
        mid = self.width // 2
        center_half = max(1, int(round(self.width * center_fraction / 2.0)))
        groups = {
            "left": xx < mid - center_half,
            "right": xx >= mid + center_half,
            "center": (xx >= mid - center_half) & (xx < mid + center_half),
        }
        flat = np.arange(self.height * self.width, dtype=np.int32).reshape(self.height, self.width)
        maps: dict[str, tuple[Array, Array]] = {}
        for side, mask in groups.items():
            source = flat[mask].ravel()
            if side == "left":
                target_pool = left_nodes
            elif side == "right":
                target_pool = right_nodes
            else:
                target_pool = chiasm_nodes
            on_pool, off_pool = self._channel_pools(target_pool)
            maps[f"{side}_on"] = (source, self._tile(on_pool, source.size))
            maps[f"{side}_off"] = (source, self._tile(off_pool, source.size))
        return maps

    def _image_array(self, image: Any) -> Array:
        if isinstance(image, (bytes, bytearray, memoryview)):
            from PIL import Image

            image = Image.open(io.BytesIO(bytes(image))).convert("L")
        if hasattr(image, "convert") and hasattr(image, "resize"):
            image = image.convert("L").resize((self.width, self.height))
            image = np.asarray(image, dtype=np.float32)
        else:
            image = np.asarray(image, dtype=np.float32)
            if image.ndim == 3:
                image = image.mean(axis=2)
            if image.ndim != 2:
                raise ValueError("image must be a 2-D grayscale array or encoded bytes")
            image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
            from PIL import Image
            # Preserve arrays already expressed in [0, 1] before uint8 resize.
            if float(np.nanmax(image, initial=0.0)) <= 1.0:
                image = image * 255.0
            image = np.asarray(
                Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).resize(
                    (self.width, self.height)
                ),
                dtype=np.float32,
            )
        image = np.nan_to_num(image.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
        lo, hi = float(image.min()), float(image.max())
        if hi > lo:
            image = (image - lo) / np.float32(hi - lo)
        else:
            image = np.zeros_like(image, dtype=np.float32)
        return image

    def encode(self, image: Any) -> RetinaFrame:
        image = self._image_array(image)
        # Center-surround filtering approximates lamina/medulla edge channels.
        dog = gaussian_filter(image, 0.9, mode="reflect") - gaussian_filter(
            image, 2.6, mode="reflect"
        )
        scale = np.float32(np.percentile(np.abs(dog), 95) + 1e-5)
        on = np.clip(dog / scale, 0.0, 1.0).astype(np.float32, copy=False)
        off = np.clip(-dog / scale, 0.0, 1.0).astype(np.float32, copy=False)

        current = np.zeros(self.graph.n_nodes, dtype=np.float32)
        for side in ("left", "right", "center"):
            src, dst = self._maps[f"{side}_on"]
            np.add.at(current, dst, on.ravel()[src] * self.input_gain)
            src, dst = self._maps[f"{side}_off"]
            np.add.at(current, dst, off.ravel()[src] * self.input_gain)

        center = np.abs(dog[:, self.width // 2 - max(1, self.width // 16) : self.width // 2 + max(1, self.width // 16)])
        left = np.abs(dog[:, : self.width // 2])
        right = np.abs(dog[:, self.width // 2 :])
        return RetinaFrame(
            current=current,
            left_energy=float(left.mean()),
            right_energy=float(right.mean()),
            center_energy=float(center.mean()),
            on_energy=float(on.mean()),
            off_energy=float(off.mean()),
        )


def _entropy(counts: Array) -> float:
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0
    p = counts[counts > 0].astype(np.float64) / total
    return float(-np.sum(p * np.log2(p)) / math.log2(max(2, counts.size)))


def _lz_complexity(bits: Array) -> float:
    """Normalised Lempel-Ziv-76 complexity for a short binary sequence."""

    s = "".join("1" if bool(x) else "0" for x in np.asarray(bits).ravel())
    n = len(s)
    if n < 2:
        return 0.0
    i, k, start, c = 0, 1, 1, 1
    while True:
        if i + k > n or start + k > n:
            c += 1
            break
        if s[i : i + k] == s[start : start + k]:
            k += 1
            if start + k > n:
                c += 1
                break
        else:
            i += 1
            if i == start:
                c += 1
                start += k
                if start >= n:
                    break
                i, k = 0, 1
            else:
                k = 1
    # c_max is approximately n/log2(n) for a random sequence.
    return float(np.clip(c * math.log2(max(2, n)) / n, 0.0, 1.0))


class BehaviorDecoder:
    """Decode central-complex-like activity into continuous motor variables."""

    def __init__(self, graph: ConnectomeGraph) -> None:
        x, y = graph.node_x, graph.node_y

        def nonempty(mask: Array, fallback: Array) -> Array:
            result = np.asarray(mask, dtype=bool).copy()
            if not result.any():
                result[int(np.flatnonzero(fallback)[0])] = True
            return result

        self.left = x < 0.5
        self.right = ~self.left
        all_nodes = np.ones(x.size, dtype=bool)
        self.left = nonempty(self.left, all_nodes)
        self.right = nonempty(self.right, all_nodes)
        self.central = nonempty((x >= 0.30) & (x <= 0.70) & (y >= 0.20) & (y <= 0.80), all_nodes)
        self.pb = nonempty((x >= 0.43) & (x <= 0.57) & (y >= 0.22) & (y <= 0.48), self.central)
        self.ellipsoid = nonempty((x >= 0.35) & (x <= 0.65) & (y >= 0.45) & (y <= 0.68), self.central)
        self.descending = nonempty(y >= 0.80, all_nodes)
        self.dorsal = nonempty(self.central & (y < 0.50), self.central)
        self.ventral = nonempty(self.central & (y >= 0.50), self.central)
        self.central_left = nonempty(self.central & (x < 0.50), self.central)
        self.central_right = nonempty(self.central & (x >= 0.50), self.central)
        self.x, self.y = x, y

    def decode(self, spike_hist: Array, energy: Array, frame: RetinaFrame) -> dict[str, Any]:
        if spike_hist.size == 0:
            raise ValueError("cannot decode an empty spike history")
        recent = slice(max(0, spike_hist.shape[0] // 3), None)
        left = spike_hist[recent][:, self.left].mean(axis=1)
        right = spike_hist[recent][:, self.right].mean(axis=1)
        pb = spike_hist[recent][:, self.pb].mean(axis=1)
        ellipsoid = spike_hist[recent][:, self.ellipsoid].mean(axis=1)
        descending = spike_hist[recent][:, self.descending].mean(axis=1)
        dorsal = spike_hist[recent][:, self.dorsal].mean(axis=1)
        ventral = spike_hist[recent][:, self.ventral].mean(axis=1)
        c_left = spike_hist[recent][:, self.central_left].mean(axis=1)
        c_right = spike_hist[recent][:, self.central_right].mean(axis=1)

        def signed(a: Array, b: Array) -> float:
            return float(np.mean((a - b) / (a + b + 1e-5)))

        yaw_cmd = float(np.clip(signed(right, left), -1.0, 1.0))
        roll_cmd = float(np.clip(signed(c_right, c_left), -1.0, 1.0))
        pitch_cmd = float(np.clip(signed(dorsal, ventral), -1.0, 1.0))
        motion = float(np.mean(np.abs(np.diff(energy)))) if energy.size > 1 else 0.0
        late_energy = float(np.mean(energy[recent]))
        early_energy = float(np.mean(energy[: max(1, energy.size // 4)]))
        looming = float(np.clip((late_energy - early_energy) * 18.0 + motion * 30.0, 0.0, 1.0))
        escape = bool(looming > 0.42 and late_energy > 0.004)
        proboscis = float(np.clip(np.mean(pb) * 80.0 + frame.center_energy * 2.0, 0.0, 1.0))
        ellipsoid_drive = float(np.clip(np.mean(ellipsoid) * 20.0, 0.0, 1.0))
        descending_drive = float(np.clip(np.mean(descending) * 20.0, 0.0, 1.0))
        freeze = bool(not escape and motion < 0.003 and late_energy > 0.001)

        # Occupancy entropy is computed over a coarse 8x8 visual map.
        bins = np.zeros(64, dtype=np.int32)
        bx = np.minimum((self.x * 8).astype(np.int32), 7)
        by = np.minimum((self.y * 8).astype(np.int32), 7)
        dominant = np.zeros(spike_hist.shape[0], dtype=bool)
        for t in range(spike_hist.shape[0]):
            ids = np.flatnonzero(spike_hist[t])
            if ids.size:
                np.add.at(bins, by[ids] * 8 + bx[ids], 1)
                dominant[t] = bool(np.mean(self.x[ids]) > 0.5)
        entropy = _entropy(bins)
        lz = _lz_complexity(dominant)
        if escape:
            behavior = "escape / looming"
        elif freeze:
            behavior = "freezing"
        elif proboscis > 0.35:
            behavior = "proboscis extension / attraction"
        elif abs(yaw_cmd) > 0.15:
            behavior = "optic-flow turn"
        else:
            behavior = "passive visual observation"
        score = float(np.clip(5.0 + 2.0 * entropy + 1.5 * lz - 2.0 * float(escape), 0.0, 10.0))
        return {
            "behavior": behavior,
            "score": round(score, 2),
            "yaw": round(yaw_cmd * 90.0, 2),
            "roll": round(roll_cmd * 90.0, 2),
            "pitch": round(pitch_cmd * 90.0, 2),
            "orientation": {"yaw": yaw_cmd * 90.0, "roll": roll_cmd * 90.0, "pitch": pitch_cmd * 90.0},
            "proboscis_extension": round(proboscis, 4),
            "ellipsoid_drive": round(ellipsoid_drive, 4),
            "descending_drive": round(descending_drive, 4),
            "freezing": freeze,
            "escape": escape,
            "looming": round(looming, 4),
            "behavioral_entropy": round(entropy, 4),
            "lz_complexity": round(lz, 4),
            "steering_bias": round(yaw_cmd * 100.0, 2),
        }


class ConnectomeSimulator:
    """Vectorised LIF network with adaptation, refractory state and STD."""

    def __init__(
        self,
        graph: ConnectomeGraph,
        retina: Optional[RetinaEncoder] = None,
        dt: float = 1.0,
        max_snapshot_nodes: int = 400,
    ) -> None:
        self.graph = graph
        self.W = graph.W
        self.dt = np.float32(dt)
        self.retina = retina or RetinaEncoder(graph)
        self.decoder = BehaviorDecoder(graph)
        self.max_snapshot_nodes = int(np.clip(max_snapshot_nodes, 1, 500))
        n = graph.n_nodes
        self.v = np.zeros(n, dtype=np.float32)
        self.adaptation = np.zeros(n, dtype=np.float32)
        self.refractory = np.zeros(n, dtype=np.int16)
        self.relative_ref = np.zeros(n, dtype=np.float32)
        self.resource = np.ones(n, dtype=np.float32)
        self.streak = np.zeros(n, dtype=np.uint8)
        self.silence = np.zeros(n, dtype=np.uint8)
        self.spikes = np.zeros(n, dtype=bool)
        self.ei_gain = np.float32(1.0)

    def reset(self) -> None:
        for arr in (self.v, self.adaptation, self.refractory, self.relative_ref, self.streak, self.silence, self.spikes):
            arr.fill(0)
        self.resource.fill(1.0)
        self.ei_gain = np.float32(1.0)

    def _snapshot(self, step: int, total: int) -> dict[str, Any]:
        ids = np.flatnonzero(self.spikes)
        if ids.size > self.max_snapshot_nodes:
            values = self.v[ids]
            ids = ids[np.argpartition(values, -self.max_snapshot_nodes)[-self.max_snapshot_nodes :]]
        nodes = [
            {"id": int(i), "x": float(self.graph.node_x[i]), "y": float(self.graph.node_y[i]), "a": round(float(max(self.v[i], 0.0)), 4)}
            for i in ids
        ]
        return {"step": step, "total_steps": total, "energy": float(np.mean(self.spikes)), "nodes": nodes}

    async def evaluate_image(
        self,
        image: Any,
        step_callback: Optional[StepCallback] = None,
        steps: int = 30,
    ) -> dict[str, Any]:
        if not 1 <= int(steps) <= 1000:
            raise ValueError("steps must be between 1 and 1000")
        self.reset()
        frame = self.retina.encode(image)
        n = self.graph.n_nodes
        spike_hist = np.zeros((int(steps), n), dtype=bool)
        energy_hist = np.zeros(int(steps), dtype=np.float32)
        decay = np.float32(math.exp(-1.0 / 8.0))
        adapt_decay = np.float32(math.exp(-1.0 / 20.0))
        resource_recovery = np.float32(1.0 / 10.0)
        ref_decay = np.float32(math.exp(-1.0 / 4.0))
        previous = np.zeros(n, dtype=np.float32)

        excitatory = self.graph.source_sign > 0.0
        inhibitory = ~excitatory
        for t in range(int(steps)):
            # Keep conductances separate so the homeostatic controller can
            # observe excitation and inhibition before they cancel at a cell.
            source = previous * self.resource
            exc = self.W.dot(source * excitatory).astype(np.float32, copy=False)
            inh = self.W.dot(source * inhibitory).astype(np.float32, copy=False)
            exc_mean, inh_mean = float(exc.mean()), float(inh.mean())
            # Slow controller tracks a balanced E/I operating point.
            self.ei_gain = np.float32(np.clip(self.ei_gain + 0.04 * (exc_mean - inh_mean), 0.35, 2.5))
            current = exc - self.ei_gain * inh
            overload = max(0.0, float(np.mean(previous)) - 0.025)
            current -= np.float32(overload * 1.5)
            if t < 2:
                current += frame.current

            self.refractory = np.maximum(self.refractory - 1, 0)
            self.relative_ref *= ref_decay
            self.v = decay * self.v + current - self.adaptation
            threshold = np.float32(0.45) + self.adaptation + 0.18 * self.relative_ref
            spikes = (self.v >= threshold) & (self.refractory == 0)
            self.v[spikes] = np.float32(0.08)
            self.adaptation = adapt_decay * self.adaptation
            self.adaptation[spikes] += np.float32(0.16)
            self.relative_ref[spikes] += np.float32(1.0)
            # One silent tick is an absolute refractory period; relative
            # refractoriness still raises the threshold after every spike.
            self.refractory[spikes] = np.int16(1)
            # Count closely spaced spike events independently of the absolute
            # refractory timer.  Three silent ticks reset the burst counter.
            streak_up = np.minimum(self.streak.astype(np.uint16) + 1, 255)
            self.silence = np.where(spikes, 0, np.minimum(self.silence + 1, 3)).astype(np.uint8)
            self.streak = np.where(spikes, streak_up, self.streak).astype(np.uint8)
            self.streak[self.silence >= 3] = 0
            self.resource += resource_recovery * (1.0 - self.resource)
            depressed = spikes & (self.streak >= 3)
            # Three consecutive spikes consume vesicles; recovery is gradual.
            self.resource[depressed] *= np.float32(0.35)
            self.spikes = spikes
            previous = spikes.astype(np.float32)
            spike_hist[t] = spikes
            energy_hist[t] = np.float32(np.mean(spikes))

            if step_callback is not None:
                payload = self._snapshot(t + 1, int(steps))
                result = step_callback(payload)
                if inspect.isawaitable(result):
                    await result
            # Let the WebSocket sender flush this frame.  Without a small
            # scheduling point, a fast simulation can enqueue all frames at
            # once and the bounded queue will retain only the final empty one.
            await asyncio.sleep(0.01)

        stats = self.decoder.decode(spike_hist, energy_hist, frame)
        active_now = int(spike_hist[-1].sum())
        total_recruited = int(np.any(spike_hist, axis=0).sum())
        resonance = 0.0
        if spike_hist.shape[0] >= 2 and spike_hist[-1].any() and spike_hist[-2].any():
            a, b = spike_hist[-1], spike_hist[-2]
            resonance = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))
        stats.update(
            {
                "active_now": active_now,
                "total_recruited": total_recruited,
                "peak_excitation": round(float(energy_hist.max()) * 1000.0, 3),
                "resonance_rate": round(resonance * 100.0, 2),
                "left_input": round(frame.left_energy, 5),
                "right_input": round(frame.right_energy, 5),
            }
        )
        return stats


__all__ = [
    "BehaviorDecoder",
    "ConnectomeGraph",
    "ConnectomeSimulator",
    "RetinaEncoder",
    "RetinaFrame",
    "load_connectome",
]
