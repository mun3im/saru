# DrongoNet latency on Wio Terminal (Nano, Micro, Edge)

> **Micro re-run at N=100**: the Micro row below (mel 141.12ms, infer
> 97.47ms, both 238.58ms) is this doc's original 10-sample average.
> Re-run at N=100 (`wt_drongonet_micro_bench.ino`, `NUM_RUNS` bumped
> 10→100): mel 141,125.1us + infer 97,559.0us = 238,684.1us — only
> ~0.04% higher, well within noise (unlike the Portenta M7 DrongoNet
> figure, which shifted ~10.6% between N=10 and N=100 — see
> `DRONGONET_SPARSE_MEL_BENCHMARK.md`'s own pointer note). Makes sense:
> this bench uses synthetic/deterministic input with fixed-shape compute,
> not real captured audio, so sample count shouldn't matter much here.
> Nano/Edge below are unchanged, still N=10. Canonical N=100 figure lives
> in `GOERTZEL_VS_DRONGONET_LATENCY.md`.

Third chip DrongoNet has been benchmarked on, alongside the Portenta H7
(`DRONGONET_SPARSE_MEL_BENCHMARK.md`) and M5Stack Core2
(`M5STACK_INFERENCE_LATENCY.md`). Hardware: **Seeed Wio Terminal**
(Microchip **ATSAMD51P19A**, single Cortex-M4F @ 120MHz, **192 KB total
SRAM**). All three DrongoNet variants attempted; Nano and Micro run and
are characterized below, Edge does not fit this chip's RAM at all (see its
own section — a real, documented result, not a gap in coverage).

Sketches: `wt_drongonet_micro_bench/`, `wt_drongonet_nano_bench/`,
`wt_drongonet_edge_bench/`. Nano and Micro are near-identical copies of
each other (same 16-mel filterbank, same CMSIS-DSP mel pipeline, same
NUM_RUNS/LCD/profiling structure) — only the model header and TFLite op
resolver differ, since Nano's graph isn't folded (batch-norm not fused
into Conv2D, carrying extra QUANTIZE/DEQUANTIZE boundary nodes Micro's
folded graph doesn't need). I/O is INT8 either way.

**Scope: mel + inference only, synthetic input**, same reasoning as the
other two chips' latency sketches — TFLite Micro's INT8 `Invoke()` and the
FFT/filterbank arithmetic are both data-independent for a fixed
graph/buffer shape, so a deterministic synthetic 3s/16kHz tone gives the
same latency a real captured clip would; only correctness/accuracy would
need real audio, and that's not what this measures. No live-mic capture
sketch was built for this chip (unlike M5Stack's `m5_mic_capture`) — out
of scope for this pass.

## Results (10 runs, `micros()`-timed, CMSIS-DSP-optimized build)

| Model | mel | inference | **both** | arena used |
|---|---:|---:|---:|---:|
| Nano  | 141.13 ms | 100.92 ms | **242.04 ms** | 23,712 / 32,768 B |
| Micro | 141.12 ms | 97.47 ms  | **238.58 ms** | 23,572 / 32,768 B |
| Edge  | — | — | **does not fit (see below)** | — |

Mel is essentially identical between Nano and Micro (both within
0.01% of each other — expected, byte-for-byte the same mel code and
filterbank). Nano's inference is ~3.5% slower than Micro's (100.92ms vs.
97.47ms), consistent with its two extra ops (QUANTIZE/DEQUANTIZE boundary
nodes the folded Micro graph doesn't carry). Run-to-run variance is
µs-level on every stage in both — expected for fixed-shape,
content-independent work.

**Micro reproduced across three separate flashes** (582,370µs→141,114µs
optimization run, then 141,113.6µs and 141,116.0µs on two more flashes of
the same optimized build — within 0.002%). Nano flashed and captured once
so far; no reason to expect it's less stable given identical DSP code and
comparable inference-timing tightness (min/max spread 6µs on 100,910-
100,967µs).

## Mel optimization: CMSIS-DSP real FFT + fast log10 (4.1x mel speedup)

Applies to both Nano and Micro (shared mel code). Per-phase profiling
(`micros()` around each stage inside the 184-frame mel loop, summed) on
the naive `arduinoFFT` baseline showed the FFT itself was **80% of mel's
cost** — the filterbank, despite doing real work (931 multiply-adds/frame
across 16 overlapping triangular filters), was already cheap because it's
applied **sparsely**: `melWeightFlat`/`melFiltersFlat` store and iterate
only the non-zero triangle-filter weights (931 total), never a dense
16×513 matrix. So the "sparse matrix" half of the original ask was
already done going in — the real lever was the FFT algorithm itself.

| Phase (summed over 184 frames) | naive (arduinoFFT) | optimized (CMSIS) | speedup |
|---|---:|---:|---:|
| windowing | 59,983 µs | 22,179 µs | 2.7x |
| **FFT** | 465,672 µs | 93,421 µs | **5.0x** |
| power spectrum | 10,356 µs | 8,577 µs | 1.2x |
| filter + log (already sparse) | 43,288 µs | 15,710 µs | 2.75x |
| normalize | 813 µs | 1,080 µs | (SIMD call overhead on a small array — net negligible either way) |
| **mel total** | **582,370 µs** | **141,114 µs** | **4.13x** |

Two changes, both ported directly from the already-validated Portenta
build (`src/DrongoNet/DrongoNet_Micro/Drongonet.ino`):

1. **`arduinoFFT` → CMSIS-DSP `arm_rfft_fast_f32`.** `arduinoFFT` computed
   a full 1024-point *complex* FFT on a real-valued signal (imaginary part
   zeroed every frame) — half the arithmetic wasted on a mathematically
   redundant imaginary half. `arm_rfft_fast_f32` is a real-input-optimized
   half-size algorithm, on top of which CMSIS's implementation is more
   aggressively optimized for the Cortex-M4F pipeline than `arduinoFFT`'s
   more general one — together good for 5x on the FFT phase alone. This
   also removes the separate `vImag` buffer and its per-frame `memset`
   entirely (real in-place transform), which is most of why windowing got
   faster too. Power-spectrum computation switched to
   `arm_cmplx_mag_squared_f32` (CMSIS SIMD) instead of a scalar
   `re*re+im*im` loop.
2. **`log10f()` → `fastLog10f()`.** An IEEE-754 bit-trick (~10 cycles vs.
   ~150 for libm `log10f`, error < 0.5 dB — negligible after min-max
   normalization), the same one already in `ARGUS_Common.h` as
   `argus_fast_log10f`. Copied as a bare inline function rather than
   `#include`-ing `ARGUS_Common.h`, which unconditionally pulls in
   `<mbed.h>`/`<mbed_stats.h>` — unavailable on this non-mbed core.

**No extra library was needed for either change.** See the toolchain note
below — CMSIS-DSP turned out to already be bundled directly in the
`Seeeduino:samd` board package itself (headers + a precompiled
`libarm_cortexM4lf_math.a`, linked into every sketch on this board
automatically), not something that had to be added.

**"int8 FFT" wasn't pursued.** CMSIS-DSP's fixed-point FFT variants are
Q15 (16-bit), not int8 — a true 8-bit FFT has nowhere near enough dynamic
range to survive a 1024-point transform's accumulated rounding error
without dedicated per-stage rescaling. Q15 *is* a realistic further step
(SAMD51's Cortex-M4F has the DSP SIMD instructions CMSIS's Q15 kernels
use, doing 2 MACs/cycle vs. float's 1), potentially worth another
meaningful cut on top of this — not attempted here, since the real-FFT
swap already closed most of the gap to Portenta's numbers and doing
fixed-point right means re-deriving the audio→Q15 scaling and re-verifying
mel output range/accuracy, not just dropping in a different function
call. Flagged as a follow-up, not done blind.

## Edge: does not fit (291,888 B / ~285 KB overflow on 192 KB total RAM)

Attempted `wt_drongonet_edge_bench/` — 80-mel filterbank
(`INPUT_SIZE=14,720` vs. Nano/Micro's 2,944), `seabadnet-edge.h` model,
`AllOpsResolver` (same choice `DrongoNet_Edge.ino` makes on Portenta).
Declared a 300 KB tensor arena (matching the ballpark of both Portenta's
SDRAM measurement, ~297 KB actual, and M5Stack's PSRAM measurement,
~296.8 KB actual) to get a real linker-reported number rather than guess.

**Link failed**: `region 'RAM' overflowed by 291,888 bytes`. Not close —
that's ~1.5x Wio Terminal's *entire* 192 KB SRAM budget, on top of what's
already used. Rough accounting for where it goes (static globals alone,
before the arena):

| Buffer | size |
|---|---:|
| `audioBuffer` (48,000 × int16) | 96,000 B |
| `melFeatures` (14,720 × float, 80-mel) | 58,880 B |
| `fftBuf` + `hannWindow` (1024 × float each) | 8,192 B |
| `powerSpectrum` (513 × float) | 2,052 B |
| `melWeightFlat` (995 non-zero weights, 80-mel) | 3,980 B |
| `melFiltersFlat` (80 × 12 B) | 960 B |
| **DSP buffers subtotal** | **~170 KB** |
| + 300 KB tensor arena | 300 KB |
| **= ~470 KB needed, vs. 192 KB available** | |

The DSP buffers alone (~170 KB, dominated by the 80-mel `melFeatures` and
the full 3s `audioBuffer`) already consume 88% of the chip's total RAM
*before* the ~297 KB arena is even considered — there is no reasonable
amount of arena-shrinking or buffer-trimming that closes a gap this size.
This matches the same conclusion already reached for Portenta's M4 core
(Edge "not attempted... doesn't fit M4's RAM", per
`M5STACK_INFERENCE_LATENCY.md`) — Edge is architecturally an
SDRAM/PSRAM-class model, and Wio Terminal has neither (no external RAM at
all, unlike Portenta's SDRAM or M5Stack Core2's 8MB PSRAM). Documented as
a real, measured result — not a gap in this characterization.

## LCD status display

All three sketches show a plain-text status on the Wio Terminal's onboard
320x240 display via `Seeed_Arduino_LCD` (`TFT_eSPI`) — bundled directly
inside the `Seeeduino:samd` board package, no separate library install
needed, and pre-configured for this board's display out of the box.
Stages shown: `Booting... / Loading model` → `Model loaded / Init mel
FFT...` → `Running benchmark / run N / 10` (updated once per loop
iteration, outside the timed mel/infer region so it doesn't affect any
reported number) → `DONE (N runs) / mel=...us cnn=...us`. (Edge never
reaches `setup()` — it fails at link time before any code runs.)

## Cross-chip comparison (mel+infer combined)

| Platform | Model | mel | inference | **total** |
|---|---|---:|---:|---:|
| Portenta M4 (actual deployment target) | Nano | 54.1 ms | 31.1 ms | **85.1 ms** |
| Portenta M4 (actual deployment target) | Micro | 54.4 ms | 30.3 ms | **84.6 ms** |
| Portenta M7 | Micro | ~29-30 ms | ~19.4-19.8 ms | ~49-50 ms |
| Wio Terminal (SAMD51, CMSIS RFFT) | Nano | 141.1 ms | 100.9 ms | **242.0 ms** |
| Wio Terminal (SAMD51, CMSIS RFFT) | Micro | 141.1 ms | 97.5 ms | **238.6 ms** |
| M5Stack Core2 (ESP32, arduinoFFT) | Nano | 294.0 ms | 85.3 ms | **379.3 ms** |
| M5Stack Core2 (ESP32, arduinoFFT) | Micro | 294.0 ms | 82.7 ms | **376.7 ms** |
| Wio Terminal (SAMD51, arduinoFFT, naive baseline) | Micro | 582.4 ms | 97.5 ms | 679.8 ms |
| Portenta M4/M7 | Edge | not attempted (doesn't fit M4's RAM; M7/SDRAM hangs on `Invoke()`, unresolved) | | |
| M5Stack Core2 (ESP32, PSRAM) | Edge | 320.9 ms | 2,584.3 ms | 2,905.2 ms |
| **Wio Terminal (SAMD51)** | **Edge** | **does not fit — 292 KB over 192 KB total RAM** | | |

Portenta M7/M4 figures from `DRONGONET_SPARSE_MEL_BENCHMARK.md`, M5Stack
figures from `M5STACK_INFERENCE_LATENCY.md`.

- **Optimized Wio Terminal now beats M5Stack Core2 on mel** (141.1 ms vs.
  294.0 ms) despite Core2's ESP32 running at 2x the clock (240MHz vs.
  120MHz) — CMSIS-DSP's real-FFT algorithm plus its Cortex-M4F-tuned
  implementation more than makes up the clock-speed gap against Core2's
  `arduinoFFT` full-complex-FFT software path. This is an
  algorithm/library difference, not a chip-capability one: `arduinoFFT`
  is the correct choice on ESP32 too (Xtensa has no CMSIS-DSP), so this
  isn't a claim that Wio Terminal's hardware is faster than Core2's, only
  that its available DSP library is much better matched to this workload.
- **Inference**: Wio Terminal is ~3.2-3.3x slower than Portenta M4 for
  both Nano and Micro, despite both chips being Cortex-M4(F) — Wio
  Terminal runs at 120MHz vs. Portenta M4's 240MHz (2x clock gap accounts
  for roughly half the difference; the rest is likely arena/cache/compiler
  differences not investigated further here). Not part of this
  optimization pass (mel was the target); TFLite Micro's `Invoke()` cost
  is a separate lever.
- **Nano vs. Micro gap is consistent across all three chips** (~3-4%,
  Nano slower) — the two extra QUANTIZE/DEQUANTIZE ops cost roughly the
  same regardless of host chip, as expected for small, fixed per-op
  overhead.
- **Edge is the one model class Wio Terminal categorically cannot run.**
  Every other chip that's tried Edge (Portenta M7, M5Stack Core2) needed
  external memory (SDRAM/PSRAM) to hold its arena — Wio Terminal has none,
  so unlike a "slow but working" result (like the naive arduinoFFT mel
  baseline), this is a hard capability boundary, not a performance one.
- **Portenta remains the only platform with comfortable margin** under the
  3s capture window for Nano/Micro (84.6-85.1ms, ~35x headroom) — Wio
  Terminal's optimized ~238-242ms totals are still comfortably under 3s
  too (~12.4-12.6x headroom), a real improvement from the naive build's
  679.8ms (~4.4x headroom). Not a claim that Wio Terminal is a candidate
  ARGUS deployment target as currently designed — Portenta's asymmetric
  dual-core architecture is the actual point of the project — but for
  Nano/Micro specifically, it's no longer wildly impractical either.

## Portenta H7 detail (M7 and M4, mel/inference/total)

Full breakdown behind the M7/M4 rows above, from
`DRONGONET_SPARSE_MEL_BENCHMARK.md`. That doc isolated three mel variants
rather than measuring a single combined build: **regular** (unoptimized
baseline — libm `log10f()`, same non-zero-only filterbank storage as
everywhere else in this doc), and **fastlog** (regular + the
`fastLog10f()` bit-trick, no further filterbank trimming). A third
variant, **sparse** (trims each filter's near-zero edge weights, 931→840
stored weights), is omitted here — its own measured contribution is tiny
(~1% M4, noise-level M7) and it isn't what Wio Terminal's build uses, so
it'd only add noise to this comparison. **fastlog is the fairer
apples-to-apples read against Wio Terminal's build** (CMSIS RFFT +
`fastLog10f()`, non-trimmed filterbank) — regular is kept alongside since
it's what the very first cross-chip table above (and
`M5STACK_INFERENCE_LATENCY.md`) already cited.

| Core | Model | variant | mel | inference | total |
|---|---|---|---:|---:|---:|
| M7 (480MHz) | Nano | regular | 30.78 ms | 19.70 ms | 50.48 ms |
| M7 (480MHz) | Nano | fastlog | 29.90 ms | 19.81 ms | 49.71 ms |
| M7 (480MHz) | Micro | regular | 30.47 ms | 19.38 ms | 49.85 ms |
| M7 (480MHz) | Micro | fastlog | 29.90 ms | 19.74 ms | 49.64 ms |
| M7 (480MHz) | Edge | — | never completed — SDRAM-arena `Invoke()` hang, unresolved (KIV) | | |
| M4 (240MHz, actual deployment target) | Nano | regular | 54.06 ms | 31.09 ms | 85.15 ms |
| M4 (240MHz, actual deployment target) | Nano | fastlog | 45.42 ms | 31.09 ms | 76.51 ms |
| M4 (240MHz, actual deployment target) | Micro | regular | 54.36 ms | 30.27 ms | 84.63 ms |
| M4 (240MHz, actual deployment target) | Micro | fastlog | 45.67 ms | 30.26 ms | 75.93 ms |
| M4 (240MHz, actual deployment target) | Edge | — | not attempted — doesn't fit M4's RAM | | |

Fastlog's win is much bigger on M4 (~16% off mel, both models landing at
exactly -16.0%) than M7 (~2-3%, within run-to-run noise for Micro) — M7's
fast hardware FPU pipeline makes even real `log10f()` comparatively
cheap, while M4's weaker FPU pays more for it. Same asymmetry Wio
Terminal's numbers sit downstream of on a third architecture: a "how
expensive is `log10f()` relative to everything else" question that gets a
different answer depending on how much the surrounding FPU/DSP hardware
is already doing the heavy lifting for free.

## All platforms combined, normalized to AudioMoth's clock (48 MHz)

**What this is and isn't.** AudioMoth (the field-deployed low-power
acoustic logger this whole project gets compared against — see
`ARGUS_RESOURCES.md`) runs a Silicon Labs EFM32 Wonder Gecko Cortex-M4F
at **48MHz** ([Hackster.io](https://www.hackster.io/news/audiomoth-dev-is-an-acoustic-development-board-34fde0c38151),
[CNX Software](https://www.cnx-software.com/2021/07/30/audiomoth-dev-is-a-full-spectrum-acoustic-development-board-based-on-silabs-efm32-mcu/)).
No DrongoNet code has actually been run on real AudioMoth hardware —
these are the measured numbers above, linearly rescaled by each chip's
clock ratio to 48MHz (`normalized = measured × chip_MHz / 48`), **not** a
prediction of real AudioMoth performance. That linear-scaling assumption
only holds if cycles-per-operation is constant across architectures,
which it isn't: Cortex-M4F@48MHz, Cortex-M4(F)@120-240MHz, Cortex-M7@480MHz,
and Xtensa LX6@240MHz have different pipelines, cache, and DSP-instruction
throughput per cycle — CMSIS-DSP's RFFT in particular is unlikely to
behave identically per-cycle on a Wonder Gecko vs. a Cortex-M7. Treat this
purely as a common-denominator intuition anchor (how big is this workload
relative to a chip a reader may already have a feel for), not as a
substitute for actually flashing AudioMoth hardware.

| Clock | ×48MHz | Model | mel@chip | infer@chip | total@chip | **total @ 48MHz-equiv** |
|---|---:|---|---:|---:|---:|---:|
| Portenta M7 (480MHz, fastlog) | 10x | Nano | 29.90 ms | 19.81 ms | 49.71 ms | 497.1 ms |
| Portenta M7 (480MHz, fastlog) | 10x | Micro | 29.90 ms | 19.74 ms | 49.64 ms | 496.4 ms |
| Portenta M4 (240MHz, fastlog) | 5x | Nano | 45.42 ms | 31.09 ms | 76.51 ms | 382.6 ms |
| Portenta M4 (240MHz, fastlog) | 5x | Micro | 45.67 ms | 30.26 ms | 75.93 ms | 379.7 ms |
| M5Stack Core2 (240MHz, arduinoFFT) | 5x | Nano | 294.0 ms | 85.3 ms | 379.3 ms | 1,896.5 ms |
| M5Stack Core2 (240MHz, arduinoFFT) | 5x | Micro | 294.0 ms | 82.7 ms | 376.7 ms | 1,883.5 ms |
| M5Stack Core2 (240MHz, PSRAM) | 5x | Edge | 320.9 ms | 2,584.3 ms | 2,905.2 ms | 14,526.0 ms |
| Wio Terminal (120MHz, CMSIS RFFT) | 2.5x | Nano | 141.13 ms | 100.92 ms | 242.04 ms | 605.1 ms |
| Wio Terminal (120MHz, CMSIS RFFT) | 2.5x | Micro | 141.12 ms | 97.47 ms | 238.58 ms | 596.5 ms |
| Wio Terminal (120MHz, arduinoFFT baseline) | 2.5x | Micro | 582.4 ms | 97.5 ms | 679.8 ms | 1,699.5 ms |

The ranking flips relative to the raw (unnormalized) table above:
**Portenta M7's real-world lead over M4 shrinks to almost nothing once
clock is factored out** (497.1ms vs. 379.7-382.6ms — M4 actually comes
out *ahead* normalized, since M7's 4x higher clock than M4 buys less than
4x the real speedup, i.e. M7 has lower work-per-cycle efficiency for this
workload, not higher). Wio Terminal's CMSIS-optimized build lands between
the two Portenta cores normalized (~596-605ms) — closer to them than to
M5Stack Core2 (~1,884-1,897ms) or its own pre-optimization arduinoFFT
baseline (~1,700ms), reinforcing that the CMSIS-DSP swap was a genuine
algorithmic win and not just riding a faster clock. This is exactly the
kind of comparison the caveat above warns about, though: it says "these
chips, at these clocks, did this much work per cycle for this specific
workload" — it does not say "an EFM32 Wonder Gecko would run DrongoNet in
380-605ms," since that chip's actual per-cycle DSP throughput hasn't been
measured here at all.

## Toolchain notes (for reproducing this)

- **Board core**: Wio Terminal isn't in arduino-cli's default index —
  added Seeed's board manager URL
  (`https://files.seeedstudio.com/arduino/package_seeeduino_boards_index.json`)
  and installed `Seeeduino:samd` (FQBN
  `Seeeduino:samd:seeed_wio_terminal`).
- **`Chirale_TensorFlowLite` doesn't compile on this core.** Its
  `system_setup.cpp` does `using namespace arduino;` unconditionally for
  any non-mbed board, which only resolves on the newer ArduinoCore-API
  (mbed, RP2040, newer SAMD/megaAVR cores) — the classic Seeeduino:samd
  core (1.8.6, forked from an older `arduino:samd`) doesn't have that
  namespace. Used the older official **`Arduino_TensorFlowLite`**
  (2.4.0-ALPHA) library instead, whose `system_setup.cpp` uses the
  classic core's global-namespace `RingBufferN`, which this core does
  provide.
- **CMSIS-DSP *is* usable here, despite first appearances** — correcting
  an earlier note in this doc's history. The *vendored Arduino wrapper
  library* `Arduino_CMSIS-DSP` really does declare
  `architectures=mbed,mbed_nano,mbed_portenta,mbed_rp2040,mbed_edge`
  (none matching this core's `samd`), so the initial mel bench used
  `arduinoFFT` instead (`architectures=*`, same choice
  `m5_drongonet_mel`/`m5_drongonet_micro_bench` made for ESP32) and
  assumed that settled it. It didn't: the **`Seeeduino:samd` board
  package bundles the real CMSIS-DSP directly** — headers under
  `<data-dir>/packages/Seeeduino/tools/CMSIS/5.7.0/CMSIS/DSP/Include/` and
  a precompiled `libarm_cortexM4lf_math.a` under
  `.../CMSIS/DSP/Lib/GCC/` — and its `platform.txt` already passes
  `-DARM_MATH_CM4 -D__FPU_PRESENT -mfloat-abi=hard -mfpu=fpv4-sp-d16` plus
  links that lib **for every sketch compiled for this board**, with zero
  extra library install. Confirmed with a minimal
  `arm_rfft_fast_init_f32()` probe sketch before committing to the port —
  `#include <arm_math.h>` alone was sufficient. Moral: a library's
  declared `architectures=` field describes that *specific Arduino
  wrapper package*, not whether the underlying vendor SDK is actually
  available on the board — worth checking what a board package bundles
  directly before assuming a whole capability class (CMSIS-DSP, in this
  case) is unavailable.
- **Core patch required**: `Arduino_TensorFlowLite` fails to compile
  against the Seeeduino:samd core's `Arduino.h` with `expected
  id-expression before '(' token` inside gemmlowp's `fixedpoint.h`, plus
  a cascading `std::max(long int, const double&)` overload-resolution
  failure. Root cause: `Arduino.h` unconditionally `#define`s
  `round(x)` (unlike `min`/`max` in the same file, which are already
  correctly guarded behind `#ifdef __cplusplus` as real templates). The
  macro's `(long)` cast silently corrupts an unqualified `std::round()`
  call deep in TFLite Micro's `FixedPoint::FromDouble()`, feeding
  `std::max()` a `long` and a `double` instead of two `double`s. Fix
  (applied to both `~/.arduino15/.../cores/arduino/Arduino.h` and the
  snap arduino-cli's isolated copy):
  ```cpp
  #ifndef __cplusplus
  #define round(x)     ((x)>=0?(long)((x)+0.5):(long)((x)-0.5))
  #endif
  ```
  C++ then falls through to newlib's real `::round()`/`std::round()`
  (double in, double out, same round-half-away-from-zero behavior) —
  same pattern the core already used for `min`/`max`.
- **Arduino IDE's sketchbook (`directories.user`) is this repo**
  (`/home/muneim/Dropbox/Conda/argus`, per
  `~/.arduinoIDE/arduino-cli.yaml`), not `~/Arduino` — so
  `Arduino_TensorFlowLite` had to be copied into this repo's own
  `libraries/` to be visible to the IDE, not just `~/Arduino/libraries/`
  (where it was needed for the standalone `arduino-cli` used during
  earlier compile/upload attempts from the CLI).
- **arduino-cli snap confinement blocked bossac** (same category of
  issue as `argus_arduino_cli_snap_quirks`/`argus_m5stack_esp32_toolchain`):
  the SAMD upload tool `bossac` is a 32-bit/i386 binary requiring
  `libstdc++.so.6`, which exists on the host but isn't visible inside the
  snap's confinement (`error while loading shared libraries`). Not an
  issue once flashing via the Arduino IDE directly (not snap-confined).
- **Uploading does *not* need a manual reset press** — the automatic
  1200bps-touch reset arduino-cli normally performs works fine, *if*
  nothing else is holding or polling the port. The first upload attempts
  this session failed and needed a manual double-press of the physical
  reset button, but that was a **port-contention artifact of the Arduino
  IDE being open**, not a hardware/toolchain limitation: the IDE's
  `arduino-language-server`/`clangd` and `serial-discovery` background
  processes (started for code-completion and port polling respectively)
  were competing for the same port, breaking the touch-reset's
  disappear/reappear detection. Confirmed by fully quitting the IDE and
  re-running `arduino-cli upload` with no button press — it uploaded and
  auto-reset cleanly. **Only need the IDE closed (or its background
  helpers killed) for CLI uploads/serial reads to work reliably** — no
  physical button interaction needed at all in that state.
- **Serial**: native USB CDC (`/dev/ttyACM0`), 115200 baud. The Arduino
  IDE's `serial-monitor` helper process (`~/.arduino15/packages/builtin/
  tools/serial-monitor/`) holds the port open and auto-relaunches after
  every build/upload even after closing its panel — reading the port
  externally (e.g. `cat`/`stty`) requires quitting the IDE entirely, not
  just closing the monitor tab.
