# Goertzel filter vs. DrongoNet-Micro: compute latency across four platforms

Cross-workload comparison (Goertzel vs. DrongoNet-Micro), extended
cross-chip across all four platforms already benchmarked for DrongoNet:
Portenta H7 M4, Portenta H7 M7, Wio Terminal (SAMD51, Cortex-M4F), and
M5Stack Core2 (ESP32-D0WDQ6-V3, Xtensa LX6 — not Cortex-M at all, despite
this doc's original three-platform title). All four run the same
algorithm (same Fs/block-size/centre-frequency coefficients), but **the
M5Stack figure comes from a different measurement methodology** than the
other three — see its own section below before comparing it directly.

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
- M5Stack Core2: `m5_goertzel_bench/` (dual-core FreeRTOS — Core 0 runs
  the Goertzel detector, Core 1 concurrently MP3-encodes and SD-writes
  the same captured audio; real mic input via I2S PDM. Coefficients
  "ported from `h7_goertzel_rpc/goertzel_m4/goertzel_m4.ino`" per that
  sketch's own header. Full writeup: `M5STACK_DUALCORE_BENCHMARK.md`.)

## Results (100-window characterization, DWT/`micros()` timed)

| Platform | Core | Clock | avg window (us) | avg window (ms) | Input |
|---|---|---:|---:|---:|---|
| Portenta H7 | M4 | 240MHz | 2,439 | 2.44 | real mic |
| Portenta H7 | M7 | 480MHz | 2,290 | 2.29 | real mic |
| Wio Terminal | SAMD51 (Cortex-M4F) | 120MHz | 8,541.6 | 8.54 | synthetic |
| M5Stack Core2 <sup>†</sup> | ESP32 (Xtensa LX6) | 240MHz | 11,550 | 11.55 | real mic |

<sup>†</sup> **Extrapolated, not directly measured at this window size —
see "M5Stack methodology" below.** All three other rows are a genuine
100-run average of one 3000-block/48000-sample window computed in a
single timed call, idle chip. The M5Stack figure is linearly scaled up
from a much smaller per-chunk average, measured under concurrent
MP3-encode + SD-write load on the other core, not on an idle chip.

All three Cortex-M runs were clean 100/100 windows, no errors. Per-window
jitter was tight on every platform (Portenta M4: 2435-2440us; Portenta
M7: 2286-2293us; Wio Terminal: 8540-8543us) — expected for fixed-size,
content-independent block processing (3000 blocks x 16 MACs + 1 magnitude
calc, same work every window regardless of audio content).

### M5Stack methodology (why it's a separate case)

`m5_goertzel_bench` (`M5STACK_DUALCORE_BENCHMARK.md`) was built for a
different question — realistic concurrent-load throughput during a 30s
live-mic run, not an idle-chip single-window characterization — so its
own reported unit is different: **256-sample (16ms) chunks**, 1876 of
them completed over the 30s run, not 3000-block/48000-sample windows.
Each chunk internally runs 16 Goertzel blocks (`CHUNK_SAMPLES /
BLOCK_SIZE`, same `BLOCK_SIZE=16` as everywhere else), averaging
**61.6 us/chunk**. Scaled linearly to the same 3000-block/48000-sample
unit the other three platforms use (3000 blocks ÷ 16 blocks/chunk =
187.5x): 61.6us x 187.5 = **11,550us**. This matches the doc's own
independently-stated "~3.85ms of compute per 1s of audio" figure exactly
(3.85ms x 3s = 11.55ms), which is a good sign the linear-scaling
assumption holds — Goertzel has no cross-block state, so per-block cost
should scale exactly linearly regardless of chunking, and it does.

**What's different about the conditions, not just the units:** MP3+SD
runs on Core 1, a **physically separate core** from Goertzel's Core 0 —
not the same core under FreeRTOS scheduling contention, so this isn't
CPU cycles being stolen from Goertzel the way, say, a higher-priority
same-core task would (see `argus_freertos_priority_starvation` memory
for what that failure mode actually looks like — this isn't it). Still,
concurrent operation was measured to cost DrongoNet Micro a real but
modest ~5-7% mel/inference slowdown vs. its own idle-chip baseline
(`M5STACK_DUALCORE_BENCHMARK.md`'s "Concurrent-load cost" table), which
that doc itself attributes to shared-bus-level effects (FreeRTOS
inter-core signalling, cache line effects) rather than PSRAM contention
specifically — a real cross-core cost, just not a CPU-scheduling one.
Goertzel wasn't independently re-measured idle on this chip to confirm
whether it pays a comparable tax, so the 11,550us figure likely includes
some amount of that same cross-core effect that the other three
platforms' genuinely-idle-chip figures don't.

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
| Portenta H7 | M7 | 2,290 | 44,357 <sup>2</sup> | **19.4x slower** |
| Wio Terminal | SAMD51 | 8,541.6 | 238,684 <sup>3</sup> | **27.9x slower** |
| M5Stack Core2 <sup>†</sup> | ESP32 | 11,550 | 401,112 <sup>4</sup> | **34.7x slower** |

<sup>1</sup> This session's own 100-run M4 characterization (`h7_drongonet_m4_instrumented`,
Micro, fastlog+sparse mel — the current default config), see commit
`152b303`: mel 44,853us + infer 30,279us = 75,132us.

<sup>2</sup> Re-run this session at N=100 (`h7_drongonet_m7_instrumented`,
Micro, fastlog+sparse — same config as the M4 figure), superseding
`DRONGONET_SPARSE_MEL_BENCHMARK.md`'s original 10-sample average (mel
29,901.9us + infer 19,737.4us = 49,639.3us, ~10.6% higher than the
N=100 re-run — a real difference, not just noise narrowing with more
samples): mel 26,849.0us + infer 17,508.2us = 44,357.3us. Clean 100/100
run, no gate skips, no errors. `h7_drongonet_m7_instrumented.ino` gained
a `CHARACTERIZE_COUNT`-based averaging block for this (previously ran
unbounded, printing every loop) — same convention the M4 sketch already
used.

<sup>3</sup> Re-run this session at N=100 (`wt_drongonet_micro_bench.ino`,
`NUM_RUNS` bumped 10→100), superseding `WIO_TERMINAL_DRONGONET_LATENCY.md`'s
original 10-sample average (mel 141.12ms + infer 97.47ms = 238.58ms =
238,580us): mel 141,125.1us + infer 97,559.0us = 238,684.1us — only
~0.04% higher than the N=10 figure, unlike M7's ~10.6% shift. Makes
sense: this bench uses synthetic/deterministic input with fixed-shape
compute (no real-mic variance to average out), so N=10 vs. N=100
shouldn't differ much, and it didn't. Clean 100/100 run, tight jitter
(mel 141,120-141,130us; infer 97,558-97,561us).

<sup>4</sup> `M5STACK_DUALCORE_BENCHMARK.md`'s own DrongoNet Micro figure,
same 30s concurrent-MP3+SD-load run as the Goertzel figure it's paired
against here (10 windows): mel 314,546.0us + infer 86,565.6us =
401,111.6us — **not** the idle-chip `M5STACK_INFERENCE_LATENCY.md`
figure (376.7ms), deliberately matched-condition against the Goertzel
row above it. This ratio (34.7x) is the same comparison
`M5STACK_DUALCORE_BENCHMARK.md` itself already reports, rounded there to
"~35x".

**The ~19-35x ratio holds across all four platforms**, not just
Portenta — this is the whole point of ARGUS's asymmetric cascade design
(see `HOW_ARGUS_WORKS.md`): a consistently ~20-35x-cheaper always-on gate
is what makes running the heavier classifier conditionally, rather than
continuously, worth doing at all, and that ratio is apparently fairly
architecture-independent for this specific filter-vs-CNN comparison —
holding up even under M5Stack's additional concurrent I/O load, not just
on an idle chip. Note ARGUS's actual production Tier-1 gate is DrongoNet
itself (not Goertzel) — `h7_goertzel_rpc`/`m5_goertzel_bench` are a
separate/earlier single-frequency-filter experiment exploring an
even-cheaper pre-gate, not what's currently wired into
`src/ARGUS/ARGUS.ino`'s cascade. This comparison quantifies what an
even-cheaper Tier-0 gate ahead of DrongoNet could buy, not a claim about
what's currently deployed.

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
| M5Stack Core2 <sup>†</sup> | ESP32 | 240MHz | 5x | 11,550 | **57,750** |

M5Stack's normalized figure is the highest of the four by a wide margin
— but per the methodology section above, its 11,550us input already
mixes in whatever cross-core (not same-core CPU-contention) cost the
other three platforms' genuinely-idle-chip figures don't pay, so this
row isn't a clean "ESP32 core is this much less efficient than
Cortex-M4F" reading on its own. Treat it as a weaker data point than the
other three in this table specifically.

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
  comparison, only compute cost. Goertzel's ~19-35x latency advantage
  comes with materially narrower/coarser detection (single ~1kHz-wide
  passband at 3kHz vs. DrongoNet's full learned spectral+temporal
  pattern), matching the known Zebra Dove miss documented in
  `argus_h7_goertzel_calibration` (content below ~1kHz falls entirely
  outside this filter's passband).
- `MAG_THRESHOLD`/peak-magnitude figures reported by these instrumented
  sketches are for sanity-checking only (confirms the filter is actually
  responding to input, not returning a constant) — not a recalibrated,
  validated detection threshold on any of these four platforms/ADC
  resolutions.
- **Input differs by platform**: Portenta M4/M7 and M5Stack numbers used
  real mic capture; Wio Terminal used a deterministic synthetic tone (no
  live-mic capture sketch exists for that board yet). This doesn't affect
  the Cortex-M platforms' latency figures (compute is content-independent
  for fixed block/window size, confirmed by each platform's own tight
  jitter), but it does mean the reported `peakMag`/`blocksAboveThreshold`
  sanity values aren't comparable across platforms.
- DrongoNet comparison figures are a mix of N=100 (Portenta M4, Portenta
  M7, Wio Terminal) and N=10 (M5Stack) characterization runs — M5Stack
  not yet re-verified at matching sample size (pending hardware swap).
- **M5Stack's figure is extrapolated and condition-mismatched with the
  other three**, not a like-for-like measurement: it's a 187.5x linear
  scale-up from a 256-sample-chunk average (not a direct 3000-block/
  48000-sample single-window timing), taken under concurrent MP3-encode +
  SD-write load on the other core rather than an idle chip. The linear
  scaling itself is on solid footing (Goertzel has no cross-block state,
  and the scaled figure independently matches the source doc's own
  "~3.85ms/s of audio" statement exactly), but the concurrent-load
  condition is a real, unquantified difference from the other three
  platforms' idle-chip numbers — treat the M5Stack row as directionally
  right, not precision-matched to the others.
- `MAG_THRESHOLD` on M5Stack was separately retuned to 2000 (vs. the
  10-bit-calibrated 150 the other platforms report against) — see
  `M5STACK_DUALCORE_BENCHMARK.md`'s own threshold section. Irrelevant to
  the latency figures (threshold doesn't affect compute time), flagged
  only so the `MAG_THRESHOLD` values aren't mixed up across platforms.
