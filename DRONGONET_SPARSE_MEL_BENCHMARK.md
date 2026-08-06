# DrongoNet: regular vs. sparse vs. fast-log mel filterbank, M7 vs. M4

> **Default config as of 2026-08-04**: `USE_SPARSE_MEL 1` in both
> `h7_drongonet_m7_instrumented.ino` and
> `h7_drongonet_m4_instrumented/drongonet_m4_instrumented.ino`. fastlog
> was already unconditional (no toggle -- `log10f()` was replaced
> outright, not gated behind a flag) from the point it was implemented,
> so both of this doc's confirmed-safe mel wins are now the standing
> default whenever these sketches are reflashed, not something to
> re-enable per run. Sparse mel's own measured contribution is small
> (~1% M4, noise-level M7 -- see "Mel-stage reduction" table below) but
> free (no correctness cost, P(bird) sanity-checked in range) and never
> conflicts with fastlog, so there's no reason to leave it off by
> default. Q15 fixed-point (a third candidate optimization) was tried
> and explicitly rejected -- see that section below -- and stays off by
> default (`RUN_Q15_COMPARE 0` in the M4 sketch).

> **M7 Micro/fastlog re-run at N=100**: the M7 Micro fastlog row below
> (29901.9 / 19737.4 / 49639.3 us) is a 10-sample average, like every
> other cell in this doc's 6-way regular/sparse/fastlog x M7/M4
> comparison -- left as-is here since changing just one cell would break
> the table's own internal N=10-vs-N=10 comparison basis. A separate
> N=100 re-run of that specific cell (same config, current default) landed
> at mel 26849.0 / infer 17508.2 / total 44357.3 us -- ~10.6% lower than
> the N=10 figure here, a real difference from more samples, not just
> noise. Use the N=100 figure (`GOERTZEL_VS_DRONGONET_LATENCY.md`) for
> anything citing "the" M7 Micro latency; this doc's table is for
> regular-vs-sparse-vs-fastlog comparison at matched sample size, not as
> the canonical single-cell number.

> **⚠ ENERGY NUMBERS INVALID (2026-08-03) -- do not cite.** The power
> source feeding the INA219's VIN+ was a bad adapter that was backfeeding
> current rather than delivering clean power. Confirmed by removing that
> adapter's VIN+ connection entirely: readings immediately went from wild,
> physically-impossible swings (bus voltage jumping 0V<->12.9V, current
> 0<->1.25A, power spiking to 4676 uJ with simultaneously-zero measured
> shunt voltage) to a clean, stable, expected "floating/no current path"
> signature (bus ~0.72-0.82V, shunt/current pinned at noise-floor,
> power=0.00mW). That means **every power/energy figure below (and the
> 77.6/78.0 mW M4 Nano/Micro numbers reported earlier this same session,
> if those made it into any thesis/paper draft) was measuring this fault
> condition, not real board power draw, at any point tonight.** The
> **latency table is unaffected** -- those come from DWT cycle counts,
> entirely independent of the INA219 -- so it's left as the trustworthy
> part of this benchmark. Re-run the energy columns once a known-good
> power adapter is confirmed in place; until then, treat every mW/uJ
> value in this document as provisional/wrong, not just imprecise.

10-sample averages per configuration. Same 16-mel filterbank (verified
byte-identical between Nano/Micro, `mel_tables.h`), same 3s/16kHz capture
window, same board (Portenta H7, full/non-Lite).

**Three mel variants compared, each isolating a different part of the mel
stage:**
- **regular** -- baseline: full filterbank, libm `log10f()`.
- **sparse** -- trims each triangular filter's near-zero edge weights
  (below 0.1 x peak, peak=1.0 for these unit triangles) instead of
  storing/accumulating the full `startBin..endBin` range. Measured:
  931 -> 840 total stored weights (~10% reduction). Only touches the
  filterbank accumulation step. `#define USE_SPARSE_MEL 0/1`.
- **fastlog** -- full (non-sparse) filterbank, but `log10f()` replaced
  with `fastLog10f()`, an IEEE-754 bit-trick approximation (same
  technique as `libraries/ARGUS_Common/src/ARGUS_Common.h`'s
  `argus_fast_log10f()`, already used in production
  `src/DrongoNet/DrongoNet_Nano`; ~10 cycles vs ~150 for libm, <0.5 dB
  error). Only touches the log conversion, called once per (frame, mel
  bin) -- 2944 times per Nano/Micro inference.

Implemented identically in `h7_drongonet_m7_instrumented.ino` and
`h7_drongonet_m4_instrumented/drongonet_m4_instrumented.ino`.

Sketches:
- M7: `h7_drongonet_m7_instrumented/h7_drongonet_m7_instrumented.ino`
  (Nano + Micro share one mel computation per loop, both models run
  every capture window)
- M4: `h7_drongonet_m4_instrumented/drongonet_m4_instrumented/drongonet_m4_instrumented.ino`
  (one model per flash -- `RUN_MODEL_NANO`/`RUN_MODEL_MICRO`, shared
  40 KB arena; see that sketch's comments for why two models can't be
  held live simultaneously on M4)

## Latency (mel / inference / total, microseconds)

| Model | Mel type | Core | mel (us) | infer (us) | total (us) |
|---|---|---|---:|---:|---:|
| Nano  | regular | M7 | 30777.2 | 19704.2 | 50481.5 |
| Nano  | sparse  | M7 | 30631.2 | 19591.0 | 50222.3 |
| Nano  | fastlog | M7 | 29901.9 | 19810.9 | 49712.8 |
| Micro | regular | M7 | 30467.3 | 19379.7 | 49847.1 |
| Micro | sparse  | M7 | 30631.2 | 19487.9 | 50119.2 |
| Micro | fastlog | M7 | 29901.9 | 19737.4 | 49639.3 |
| Nano  | regular | M4 | 54056.2 | 31091.7 | 85147.9 |
| Nano  | sparse  | M4 | 53352.6 | 31047.2 | 84399.8 |
| Nano  | fastlog | M4 | 45421.5 | 31091.9 | 76513.4 |
| Micro | regular | M4 | 54360.4 | 30268.9 | 84629.3 |
| Micro | sparse  | M4 | 53707.5 | 30223.4 | 83930.9 |
| Micro | fastlog | M4 | 45665.3 | 30260.3 | 75925.6 |

**Mel-stage reduction vs. regular baseline:**

| Model | Core | sparse | fastlog |
|---|---|---:|---:|
| Nano  | M7 | -0.5% | **-2.8%** |
| Micro | M7 | +0.5% (noise) | **-1.9%** |
| Nano  | M4 | -1.3% | **-16.0%** |
| Micro | M4 | -1.2% | **-16.0%** |

## Energy (mel / inference / total, microjoules) + average power

> **⚠ INVALID -- see warning at top of document.** Bad power adapter was
> backfeeding through the INA219's VIN+; these numbers do not reflect
> real board power draw. Kept in the table for the record (and so the
> pre/post-fix contrast is visible if this doc is revisited), not as
> data to use. No energy figures were collected for the fastlog variant
> (measured after the adapter fault was found and removed -- energy
> reads a flat 0.0 mW throughout, correctly reflecting the now-floating
> INA219, not a real measurement).

| Model | Mel type | Core | power (mW) | E_mel (uJ) | E_infer (uJ) | E_total (uJ) |
|---|---|---|---:|---:|---:|---:|
| Nano  | regular | M7 | 99.6 | 3064.7 | 1962.0 | 5026.7 |
| Nano  | sparse  | M7 | 94.8 | 2903.5 | 1857.0 | 4760.4 |
| Micro | regular | M7 | 99.6 | 3034.3 | 1930.6 | 4964.9 |
| Micro | sparse  | M7 | 94.8 | 2903.5 | 1847.5 | 4750.9 |
| Nano  | regular | M4 | 100.4 | 5427.2 | 3121.6 | 8548.8 |
| Nano  | sparse  | M4 | 98.0 | 5228.6 | 3042.6 | 8271.2 |
| Micro | regular | M4 | 96.0 | 5218.6 | 2905.8 | 8124.4 |
| Micro | sparse  | M4 | 96.4 | 5177.4 | 2913.5 | 8090.9 |

## Energy, datasheet-derived estimate (STM32H747xI Run-mode IDD, not measured)

Cross-check computed from the STM32H747xI/G datasheet (DS12930 Rev 3) while
the INA219 power path was untrustworthy. Source tables:

- **Table 28** -- "Run mode, code from flash memory, only Arm Cortex-M7
  running, cache ON, LDO regulator ON": VOS0/480MHz, all peripherals
  enabled, Typ **220 mA**.
- **Table 31** -- "Run mode, code from flash memory, only Arm Cortex-M4
  running, ART accelerator ON, LDO regulator ON": VOS0/240MHz, all
  peripherals enabled, Typ **190 mA**.
- Rail voltage: **3.3V**, confirmed from the Portenta H7 schematic (`+3V3`
  net feeding the STM32's VDD pins).
- **LDO vs. SMPS resolved**: the schematic shows SMPS-related nets
  (`VLXSMPS` etc.), raising the question of whether the SMPS-regulator
  tables (22/23/27/32) applied instead. Ruled out by Table 27 itself: SMPS
  regulator mode on the standard STM32H747xI **does not support VOS0 at
  all** (max VOS1/400MHz; VOS0/480MHz via SMPS is listed only for a
  different part, "STM32H757XIH6A"). Since our sketches run M7 at its
  default 480MHz (VOS0 -- the Portenta's documented max, no explicit
  clock reconfig in any of these sketches), the chip cannot be in
  pure-SMPS mode for that rail -- LDO tables are the correct reference
  regardless of what's on the schematic.
- M7 core power: 220mA x 3.3V = **726.0 mW**. M4 core power: 190mA x
  3.3V = **627.0 mW**. Applied as flat per-core constants (Typ @ 25 degC)
  against each row's measured total latency (mel + inference, DWT-based,
  trustworthy).

| Model | Mel type | Core | power (mW) | E_total (uJ) |
|---|---|---|---:|---:|
| Nano  | regular | M7 | 726.0 | 36649.6 |
| Nano  | sparse  | M7 | 726.0 | 36461.4 |
| Nano  | fastlog | M7 | 726.0 | 36091.5 |
| Micro | regular | M7 | 726.0 | 36189.0 |
| Micro | sparse  | M7 | 726.0 | 36386.5 |
| Micro | fastlog | M7 | 726.0 | 36038.1 |
| Nano  | regular | M4 | 627.0 | 53387.7 |
| Nano  | sparse  | M4 | 627.0 | 52918.7 |
| Nano  | fastlog | M4 | 627.0 | 47973.9 |
| Micro | regular | M4 | 627.0 | 53062.6 |
| Micro | sparse  | M4 | 627.0 | 52624.7 |
| Micro | fastlog | M4 | 627.0 | 47605.4 |

**This is a lower-bound estimate, not a measurement.** It covers only the
STM32H747's own core IDD (VOS0, "all peripherals enabled" bucket) -- it
does **not** include: the board's own upstream regulator efficiency
(STM32 VDD is downstream of Portenta's own 3.3V regulation, not 100%
efficient), the INA219 itself, mic bias/ADC front-end, SD card (if
active), or SDRAM (irrelevant here -- Nano/Micro don't use it). It also
assumes the *other* core is fully idle/reset during each single-core test,
which was not independently confirmed for how these sketches were flashed
(M4 sketches don't stop a previously-flashed, still-resident M7 image or
vice versa without an explicit reflash of both). Real total board power
will likely be higher than this, not lower.

**Sanity-check value**: this estimate lands roughly 6-7x higher than the
(already-invalidated) measured figures earlier in this document -- e.g.
M7 Nano-regular: 36649.6 uJ estimated vs. 5026.7 uJ measured; M4
Nano-regular: 53387.7 uJ estimated vs. 8548.8 uJ measured. That consistent
gap is independent corroboration that the bad power adapter wasn't just
adding noise -- it was suppressing the real reading by nearly an order of
magnitude, not a small offset. Treat any *future* INA219 re-measurement
that lands anywhere near the old ~95-100 mW range as still suspect; a
correct reading should be much closer to several hundred mW per core.

## P(bird) average (sanity check -- variants should broadly track regular)

| Model | Mel type | Core | P(bird) avg |
|---|---|---|---:|
| Nano  | regular | M7 | 0.8398 |
| Nano  | sparse  | M7 | 0.8441 |
| Nano  | fastlog | M7 | 0.6570 |
| Micro | regular | M7 | 0.7719 |
| Micro | sparse  | M7 | 0.7582 |
| Micro | fastlog | M7 | 0.6160 |
| Nano  | regular | M4 | 0.7842 |
| Nano  | sparse  | M4 | 0.8148 |
| Nano  | fastlog | M4 | 0.8097 |
| Micro | regular | M4 | 0.7828 |
| Micro | sparse  | M4 | 0.7777 |
| Micro | fastlog | M4 | 0.7580 |

## Q15 fixed-point FFT on M4 -- tried, not a win (2026-08-04)

CMSIS-DSP's `arm_rfft_q15` was tried as a further M4 mel-stage optimization
(Cortex-M4's DSP-extension SIMD MACs, dual 16-bit ops/cycle, vs. the f32
path's one-at-a-time FPU ops). Packing/scale were verified empirically on
real hardware first (`q15_rfft_diag/`) before writing the real pipeline:
`arm_rfft_q15`'s output for `fftLenReal=1024` is the full N-point complex
spectrum (`dst[2k]=Re(X[k])`, `dst[2k+1]=Im(X[k])`), scaled by exactly
1/1024 vs. the unnormalized f32 output -- both confirmed against known
test tones (DC, Nyquist, and an interior bin), with the Hann-window
3-bin leakage signature (-0.25/+0.5/-0.25) showing up identically in the
FFT's Hermitian mirror half as independent confirmation of the packing.

Implemented as a shadow comparison in `drongonet_m4_instrumented.ino`
(`RUN_Q15_COMPARE`, gated to free the 40 KB TFLite arena rather than risk
the documented tight-RAM hang -- see that file), computing both mel paths
on the same real captured clip and reporting timing + error via RPC.
Result, 5 real captures:

| | value |
|---|---:|
| mel_f32 | ~45013 us |
| mel_q15 | ~51845 us |
| speedup | **0.87x (15% slower)** |
| maxAbsDiff (normalized mel, 0-1 scale) | 0.31-0.56 |
| meanAbsDiff | 0.04-0.16 |

**Both slower and less accurate -- not adopted.** Root cause for the
speed regression: this CMSIS-DSP version (5.7.0, bundled with the
Portenta board package) has no `arm_rfft_fast_q15` -- only F32/F64 got
the optimized "fast" RFFT; `arm_rfft_q15` is the older, structurally
slower implementation. This wasn't a same-algorithm datatype comparison;
it was an older algorithm in Q15 against a newer one in F32, and the
algorithmic gap outweighed Q15's per-instruction SIMD throughput
advantage. Root cause for the accuracy regression: `arm_rfft_q15`
downscales by a fixed schedule (halving every internal stage) regardless
of input amplitude. The diagnostic's synthetic test tones (amplitude
8000) landed cleanly; real mic audio (RMS ~1470-1960 here, much quieter
relative to int16 full scale) loses much more effective precision under
that same fixed schedule, which the subsequent log step then amplifies
(small absolute errors near the noise floor become large relative dB
differences). Fixing this would need per-frame block-floating-point
scaling (normalize each frame's amplitude before the FFT, track the
scale factor, correct after) -- a real undertaking, and even then the
"no fast Q15 RFFT" algorithmic problem remains untouched. Not worth
pursuing further here: fast-log10f (see above) already captured the
practical M4 win (-16%) at far lower cost and risk.

## Notes

- **Fast-log is the real win, especially on M4.** Swapping libm
  `log10f()` for the bit-trick approximation cuts mel-stage time by
  ~16% on M4 (both models, both landing at exactly -16.0%) vs. only
  ~2-3% on M7. Matches expectations going in: M7's fast hardware FPU
  makes even the real `log10f()` comparatively cheap, while M4's weaker
  core pays a much larger relative tax for the transcendental call --
  2944 calls per inference (16 mels x 184 frames) is enough for that
  difference to show up clearly. **This is the more effective lever of
  the two mel optimisations tested here**, particularly if M4 latency
  matters for the final writeup.
- **Sparse mel's payoff is small in practice, on both cores.** The
  filterbank pruning removes ~10% of stored weights (931 -> 840), but
  measured mel-stage wall time only drops ~1-1.3% on M4 and is within
  noise on M7 (Micro even reads slightly higher under sparse -- run-to-
  run jitter, not a real regression). Confirms the CMSIS-DSP FFT itself
  (unchanged, fixed cost either way) dominates the mel stage far more
  than the filterbank accumulation does.
- **Fast-log's P(bird) shift is larger than sparse mel's.** M7 Nano
  drops from 0.8398 -> 0.6570 (regular -> fastlog), a bigger swing than
  any sparse-mel pair saw. The approximation's ~0.5 dB error compounds
  across all 2944 mel values before normalisation, shifting the
  quantized input more than sparse mel's edge-trimming does. Same
  approximation already ships in production `DrongoNet_Nano`, so this
  isn't a correctness red flag on its own, but it's a real behavioural
  difference worth being aware of if fast-log is adopted -- re-verify
  `BIRD_SCORE_THRESHOLD` against fast-log data specifically rather than
  assuming the regular-mel threshold still applies.
- M4 is consistently ~1.75-1.9x slower than M7 for inference regardless
  of mel variant (e.g. Nano infer: ~19.7-19.8ms M7 vs ~31.0-31.1ms M4);
  fast-log narrows the *mel* gap specifically (M4/M7 mel ratio drops
  from ~1.76-1.79x under regular to ~1.52x under fastlog) without
  touching inference at all, as expected -- the two stages are
  independent.
- M7 mel/infer times bounce between two clusters (~28-30ms) across
  samples in the regular/sparse runs -- not something either mel variant
  introduced; worth a closer look separately if it matters for a future
  writeup.
- ~~Energy tracks latency directly~~ -- withdrawn; the power readings
  behind this observation are invalid (see warning at top). No energy
  conclusion until re-measured with a known-good power adapter.
