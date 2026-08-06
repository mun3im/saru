# Goertzel filter vs. DrongoNet-Micro: compute latency across all three Cortex-M4(F) platforms

Cross-workload comparison (Goertzel vs. DrongoNet-Micro), extended
cross-chip across all three platforms already benchmarked for DrongoNet:
Portenta H7 M4, Portenta H7 M7, and Wio Terminal (SAMD51). All three run
the same algorithm over the same 3s/16kHz/48000-sample capture window,
so filter-vs-filter and platform-vs-platform comparisons are both valid
within this doc.

**Algorithm**: single fixed-frequency Goertzel filter (Fs=16kHz,
16-sample block, 3kHz centre — the calibrated `h7_goertzel_rpc` params,
see `argus_h7_goertzel_calibration` memory), scanned as 3000 consecutive
non-overlapping 16-sample blocks across one full window. This is the same
algorithm as the live `goertzel_m4.ino` field detector, just timed
against a pre-captured/synthetic buffer instead of the live
analogRead()-paced loop (isolates compute from real-time sample-interval
waiting — see `goertzel_m4_instrumented.ino`'s file header for why that
split matters). `MAG_THRESHOLD=150` was calibrated at 10-bit ADC (the
original live sketch's assumption) and is **not** re-calibrated on any
platform here — every number in this doc is a timing bench, not a
validated field-detection config.

Sketches:
- Portenta M4: `h7_goertzel_rpc/goertzel_m4_instrumented/goertzel_m4_instrumented.ino`
  (RPC-relays to M7, no direct Serial on M4; real mic input, 12-bit ADC)
- Portenta M7: `h7_goertzel_m7_instrumented/h7_goertzel_m7_instrumented.ino`
  (standalone, no RPC, direct Serial — same pattern as `h7_drongonet_m7`;
  real mic input, 12-bit ADC)
- Wio Terminal: `wt_goertzel_bench/wt_goertzel_bench.ino` (standalone,
  direct Serial; **synthetic** input — same rationale as
  `wt_drongonet_micro_bench.ino`: fixed-block-count compute is
  content-independent, and no live-mic capture sketch exists for this
  board yet, matching `WIO_TERMINAL_DRONGONET_LATENCY.md`'s own scope
  note)

## Results (100-window characterization, DWT/`micros()` timed)

| Platform | Core | Clock | avg window (us) | avg window (ms) | Input |
|---|---|---:|---:|---:|---|
| Portenta H7 | M4 | 240MHz | 2,439 | 2.44 | real mic |
| Portenta H7 | M7 | 480MHz | 2,290 | 2.29 | real mic |
| Wio Terminal | SAMD51 (Cortex-M4F) | 120MHz | 8,541.6 | 8.54 | synthetic |

All three clean 100/100 runs, no errors. Per-window jitter was tight on
every platform (Portenta M4: 2435-2440us; Portenta M7: 2286-2293us; Wio
Terminal: 8540-8543us) — expected for fixed-size, content-independent
block processing (3000 blocks x 16 MACs + 1 magnitude calc, same work
every window regardless of audio content).

**Portenta M4/M7 ratio is only ~1.07x** (M7 slightly faster) — much
narrower than the ~1.5-1.9x M4/M7 gap DrongoNet shows for mel+inference
(see below). Makes sense: Goertzel's inner loop is a handful of scalar
float ops per sample (1 multiply, 2 adds), cheap enough that M4's weaker
FPU pipeline barely shows up relative to M7's; DrongoNet's mel stage (FFT
+ log) and inference (CNN) both lean much more heavily on
throughput-sensitive operations (transcendental calls, larger matrix-ish
ops) where M7's faster/wider FPU pipeline actually matters.

**Wio Terminal is ~3.7x slower than Portenta M4 despite both being
Cortex-M4(F)** — at raw clock this tracks its 2x-lower clock (120MHz vs.
240MHz) plus some. See the clock-normalized table below for a closer
per-cycle-efficiency read.

## Comparison vs. DrongoNet-Micro (mel + inference, same board/window)

| Platform | Core | Goertzel (us) | DrongoNet-Micro mel+infer (us) | DrongoNet / Goertzel |
|---|---|---:|---:|---:|
| Portenta H7 | M4 | 2,439 | 75,132 <sup>1</sup> | **30.8x slower** |
| Portenta H7 | M7 | 2,290 | 49,639 <sup>2</sup> | **21.7x slower** |
| Wio Terminal | SAMD51 | 8,541.6 | 238,580 <sup>3</sup> | **27.9x slower** |

<sup>1</sup> This session's own 100-run M4 characterization (`h7_drongonet_m4_instrumented`,
Micro, fastlog+sparse mel — the current default config), see commit
`152b303`: mel 44,853us + infer 30,279us = 75,132us.

<sup>2</sup> `DRONGONET_SPARSE_MEL_BENCHMARK.md`, M7 Micro fastlog row
(10-sample average, not 100 — only M4 Goertzel and M4 DrongoNet have been
re-run at N=100 so far): mel 29,901.9us + infer 19,737.4us = 49,639.3us.

<sup>3</sup> `WIO_TERMINAL_DRONGONET_LATENCY.md`, Micro row (10-sample
average, CMSIS-DSP-optimized build): mel 141.12ms + infer 97.47ms =
238.58ms = 238,580us.

**The ~22-31x ratio holds across all three platforms**, not just
Portenta — this is the whole point of ARGUS's asymmetric cascade design
(see `HOW_ARGUS_WORKS.md`): a consistently ~20-30x-cheaper always-on gate
is what makes running the heavier classifier conditionally, rather than
continuously, worth doing at all, and that ratio is apparently fairly
architecture-independent for this specific filter-vs-CNN comparison.
Note ARGUS's actual production Tier-1 gate is DrongoNet itself (not
Goertzel) — `h7_goertzel_rpc` is a separate/earlier single-frequency-
filter experiment exploring an even-cheaper pre-gate, not what's
currently wired into `src/ARGUS/ARGUS.ino`'s cascade. This comparison
quantifies what an even-cheaper Tier-0 gate ahead of DrongoNet could buy,
not a claim about what's currently deployed.

## Clock-normalized comparison (Goertzel only, x48MHz-equivalent)

Same normalization convention as `WIO_TERMINAL_DRONGONET_LATENCY.md`'s
own AudioMoth-clock table (`normalized = measured × chip_MHz / 48`) — a
common-denominator intuition anchor, not a claim about real per-cycle
behavior transferring across architectures (see that doc's own caveat,
which applies equally here).

| Platform | Core | Clock | ×48MHz | measured (us) | **normalized (us)** |
|---|---|---:|---:|---:|---:|
| Portenta H7 | M4 | 240MHz | 5x | 2,439 | **12,195** |
| Portenta H7 | M7 | 480MHz | 10x | 2,290 | **22,900** |
| Wio Terminal | SAMD51 | 120MHz | 2.5x | 8,541.6 | **21,354** |

**Unlike DrongoNet's normalized table** (where M4 and M7 end up close
together, both well ahead of Wio Terminal), **Goertzel's normalized M4
figure is the clear outlier here — nearly 2x cheaper per-cycle than
either M7 or Wio Terminal**, which land close to each other. Not fully
explained by this session's data: both Wio Terminal and Portenta M4 are
Cortex-M4(F) cores at comparable per-op costs for simple scalar
arithmetic, so a ~1.75x per-cycle gap for a workload this simple (no FFT,
no transcendental calls, just multiply-add + one sqrt per block) is
larger than expected from core architecture alone. Plausible contributors
not isolated here: compiler optimization level defaults differing
between the `arduino:mbed_portenta` and `Seeeduino:samd` board packages,
Portenta M4's tightly-coupled-memory (TCM) access vs. Wio Terminal's
flash-wait-state-affected SRAM access pattern, or measurement-method
differences (DWT cycle count on Portenta vs. `micros()` on Wio Terminal
— though at ~2-8ms scale, `micros()`'s 1us resolution shouldn't itself
explain a 1.75x gap). Flagged as an open question, not a settled
conclusion — worth a closer per-instruction look if this specific
Portenta-M4-vs-SAMD51 efficiency gap matters for a future writeup.

## Caveats

- Single fixed-frequency filter vs. a full 16-mel/184-frame spectrogram +
  4-layer CNN — this is not an apples-to-apples "detector quality"
  comparison, only compute cost. Goertzel's ~22-31x latency advantage
  comes with materially narrower/coarser detection (single ~1kHz-wide
  passband at 3kHz vs. DrongoNet's full learned spectral+temporal
  pattern), matching the known Zebra Dove miss documented in
  `argus_h7_goertzel_calibration` (content below ~1kHz falls entirely
  outside this filter's passband).
- `MAG_THRESHOLD`/peak-magnitude figures reported by these instrumented
  sketches are for sanity-checking only (confirms the filter is actually
  responding to input, not returning a constant) — not a recalibrated,
  validated detection threshold on any of these three platforms/ADC
  resolutions.
- **Input differs by platform**: Portenta M4/M7 numbers used real mic
  capture (12-bit ADC); Wio Terminal used a deterministic synthetic tone
  (no live-mic capture sketch exists for that board yet). This doesn't
  affect the latency figures themselves (compute is content-independent
  for fixed block/window size on all three platforms, confirmed by each
  platform's own tight jitter), but it does mean the reported
  `peakMag`/`blocksAboveThreshold` sanity values aren't comparable
  across platforms.
- DrongoNet comparison figures are a mix of N=100 (Portenta M4 only) and
  N=10 (Portenta M7, Wio Terminal) characterization runs — not
  independently re-verified at matching sample sizes across platforms in
  this pass.
