"""Low-overhead WebSocket snapshots for connectome simulations.

The simulator keeps its state in NumPy arrays.  This module deliberately does
not copy the full state: a frame contains only the strongest active neurons.
It can be used from a FastAPI endpoint and from an async simulation loop::

    hub = WebSocketHub()
    await hub.connect(websocket)
    frame = SpikeSnapshot.from_activity(step, activity, x, y)
    await hub.broadcast(frame)

``SpikeSnapshot`` is JSON by default so it can be consumed directly by a
Canvas client.  ``to_binary`` is available when transport overhead matters.
"""

from __future__ import annotations

import asyncio
import json
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # FastAPI is optional for users who only need the encoder.
    from fastapi import WebSocket
except Exception:  # pragma: no cover - allows importing encoder standalone
    WebSocket = Any  # type: ignore[misc,assignment]


DEFAULT_MAX_NODES = 500


@dataclass(slots=True)
class SpikeSnapshot:
    """One simulation tick, containing only active neurons.

    ``nodes`` entries are ``[index, x, y, activity]``.  A list instead of a
    dictionary reduces JSON size and avoids repeated key strings at 60 FPS.
    """

    step: int
    total_steps: int | None
    energy: float
    nodes: list[list[float | int]]
    kind: str = "step"

    @classmethod
    def from_activity(
        cls,
        step: int,
        activity: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        *,
        total_steps: int | None = None,
        threshold: float = 0.04,
        max_nodes: int = DEFAULT_MAX_NODES,
    ) -> "SpikeSnapshot":
        """Build a frame without sorting all ``n_nodes`` activities.

        ``argpartition`` is O(n) and only the selected values are converted to
        Python objects.  Inputs are read-only and can be the simulator's live
        float32 arrays.
        """
        if max_nodes < 1:
            raise ValueError("max_nodes must be positive")
        a = np.asarray(activity)
        xx = np.asarray(x)
        yy = np.asarray(y)
        if a.ndim != 1 or xx.ndim != 1 or yy.ndim != 1:
            raise ValueError("activity and coordinates must be one-dimensional")
        if not (a.size == xx.size == yy.size):
            raise ValueError("activity and coordinates must have equal length")

        active = np.flatnonzero(a > np.float32(threshold))
        if active.size > max_nodes:
            # Keep the most intense units, with deterministic tie behaviour for
            # the common case where several neurons have identical activity.
            chosen = np.argpartition(a[active], -max_nodes)[-max_nodes:]
            active = active[chosen]
            active = active[np.argsort(a[active])[::-1]]
        else:
            active = active[np.argsort(a[active])[::-1]]

        # float() also handles float16/float32 and prevents NumPy scalar JSON
        # encoder failures.  Clamp activity to keep a malformed value from
        # breaking the Canvas alpha channel.
        nodes: list[list[float | int]] = []
        for idx in active:
            value = float(np.clip(a[idx], 0.0, 1.0))
            nodes.append([int(idx), float(xx[idx]), float(yy[idx]), round(value, 4)])
        return cls(
            step=int(step),
            total_steps=None if total_steps is None else int(total_steps),
            energy=float(np.mean(a, dtype=np.float32)) if a.size else 0.0,
            nodes=nodes,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.kind,
            "step": self.step,
            "total_steps": self.total_steps,
            "energy": round(self.energy, 6),
            "nodes": self.nodes,
        }

    def to_json(self) -> str:
        # Compact separators are significant for a 60 FPS stream.
        return json.dumps(self.as_dict(), separators=(",", ":"), ensure_ascii=False)

    def to_binary(self) -> bytes:
        """Return a compact little-endian frame for a binary WebSocket.

        Header: ``magic(2), version(u8), flags(u8), step(u32), count(u16), energy(f32)``.
        Each node: ``index(u32), x(f32), y(f32), activity(f32)``.  Coordinates
        are quantized to float32, matching the simulator's coordinate arrays.
        """
        count = min(len(self.nodes), 0xFFFF)
        # magic, version, flags, step, count, energy
        out = bytearray(struct.pack("<2sBBIHf", b"SP", 1, 0, self.step, count, self.energy))
        if count:
            packed = bytearray()
            for idx, x, y, activity in self.nodes[:count]:
                packed.extend(struct.pack("<Ifff", int(idx), float(x), float(y), float(activity)))
            out.extend(packed)
        return bytes(out)


class WebSocketHub:
    """Small connection registry with bounded per-client queues.

    A slow browser cannot stall the simulation: each client keeps at most
    ``queue_size`` frames and older frames are dropped when the queue fills.
    """

    def __init__(self, queue_size: int = 2) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        self.queue_size = queue_size
        self._clients: dict[WebSocket, asyncio.Queue[str | bytes]] = {}
        self._tasks: dict[WebSocket, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    @property
    def clients(self) -> tuple[WebSocket, ...]:
        return tuple(self._clients)

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        queue: asyncio.Queue[str | bytes] = asyncio.Queue(self.queue_size)
        async with self._lock:
            self._clients[websocket] = queue
            self._tasks[websocket] = asyncio.create_task(self._sender(websocket, queue))

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.pop(websocket, None)
            task = self._tasks.pop(websocket, None)
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _sender(self, websocket: WebSocket, queue: asyncio.Queue[str | bytes]) -> None:
        try:
            while True:
                payload = await queue.get()
                if isinstance(payload, bytes):
                    await websocket.send_bytes(payload)
                else:
                    await websocket.send_text(payload)
        except Exception:
            # The receive loop normally removes the client.  This also handles
            # a browser disappearing while its sender is blocked.
            async with self._lock:
                self._clients.pop(websocket, None)
                self._tasks.pop(websocket, None)

    async def broadcast(self, payload: SpikeSnapshot | dict[str, Any] | str | bytes) -> None:
        if isinstance(payload, SpikeSnapshot):
            message: str | bytes = payload.to_json()
        elif isinstance(payload, dict):
            message = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        else:
            message = payload
        for websocket, queue in tuple(self._clients.items()):
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                # Keep the newest frame.  A stale frame has no value for an
                # animation and dropping it bounds memory and latency.
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(message)
                except asyncio.QueueFull:
                    pass

    async def close(self) -> None:
        for websocket in tuple(self._clients):
            await self.disconnect(websocket)


__all__ = ["SpikeSnapshot", "WebSocketHub", "DEFAULT_MAX_NODES"]
