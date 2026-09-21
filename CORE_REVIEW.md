`connectome_core.py` now contains the graph loader, RetinaEncoder, adaptive
LIF/ST simulator and BehaviorDecoder.  The loader uses Arrow IPC record batches
to keep peak memory bounded (only the three required columns are touched),
then creates float32 CSR `W[post, pre]` with incoming L1 gain <= 0.90.  If the
public API should stay Polars-only, replace the two-pass `pyarrow.ipc` loop with
an equivalent Polars scanner; Arrow is used here because it exposes bounded
record-batch reads for Feather.

The edge file has no cell types or coordinates, so `source_sign` (20% E/I) and
node x/y are deterministic SplitMix64 hashes of root ids.  They must be
replaced with annotations when available.  The retina maps noncentral columns
to left/right pools and a central band to a chiasm pool; ON/OFF values are
positive/negative DoG lobes.  `ConnectomeSimulator.evaluate_image` emits only
spiking nodes (max 500) and `BehaviorDecoder` returns yaw/roll/pitch,
proboscis, freeze/escape, Shannon occupancy entropy and LZ complexity.

Smoke checks passed with a 30k-edge generated Feather fixture:

    python3 -m py_compile connectome_core.py
    load_connectome(...); await ConnectomeSimulator.evaluate_image(...)

One dynamic detail for integration: `streak` currently decays on silent ticks
so the three-spike STD trigger means three closely spaced spikes.  If strict
adjacent timesteps are needed, retain streak during absolute refractory and
reset only after an un-refractory silent tick.

Real-file verification (2026-09-19): `load_connectome` completed in 11.5 s, produced 138,639 nodes / 15,091,983 CSR nonzeros (121 MB CSR), and peak RSS 612 MB. A 3-step random-image simulation completed after loading with peak RSS 710 MB and returned motor stats; both are below the 1.2 GB process limit.
