# Goertzel vs. DrongoNet Micro, dual-core, under concurrent MP3+SD load

> **DrongoNet Micro re-run at N=100 windows**: the "Results" table below
> is this doc's original 30s/10-window run. Re-run at
> `BENCHMARK_DURATION_MS=300000` (300s, `m5_drongonet_micro_bench.ino`)
> to get ~100 windows (each window is real-time-paced by live mic capture
> at ~1 per 3s, so 10x the duration was needed for 10x the windows):
> mel 314,424.1us (min 297,852 max 317,767) + infer 87,069.9us (min
> 83,207 max 87,804) = 401,494.0us — only ~0.1% higher than the N=10
> figure below (401,111.6us), well within noise. Clean 100/100 windows,
> zero `qDetector`/`qMp3` drops, zero SD write failures (8,333 MP3
> frames, 292 SD writes, 1,199,952 bytes — both scale linearly with the
> 10x longer run, as expected). Canonical N=100 figure lives in
> `GOERTZEL_VS_DRONGONET_LATENCY.md`; Goertzel's own figure in that
> comparison is a separate extrapolation from this doc's chunk-level
> data, not re-measured directly at N=100 windows.

Benchmark comparing the cheap single-bin Goertzel filter against
DrongoNet Micro's full mel+CNN pipeline as the Core-0 "bird audio
detector", while Core 1 continuously encodes captured audio to MP3
(vendored Shine, `libshine`) and writes it to SD, on M5Stack Core2
(ESP32-D0WDQ6-V3). This adds the realistic concurrent I/O load none of
the earlier idle-chip characterizations
(`M5STACK_INFERENCE_LATENCY.md`) had. Design plan:
`/home/muneim/.claude/plans/inherited-singing-peach.md`. Sketches:
`m5_goertzel_bench/`, `m5_drongonet_micro_bench/`.

30s fixed runs, real captured audio (not synthetic -- unlike the earlier
inference/mel-only characterizations, this benchmark exercises the live
mic + real detection + real MP3/SD pipeline together).

## Results

| | Goertzel | DrongoNet Micro |
|---|---:|---:|
| Per-pass unit | 256-sample (16ms) chunk | 48000-sample (3s) window |
| Passes completed | 1876 chunks | 10 windows |
| Compute avg | 61.6 us/chunk | mel 314546.0 us + infer 86565.6 us = 401111.6 us/window |
| Compute min/max | 37 / 235 us | mel 299900-316357 us, infer 83190-87412 us |
| Duty cycle (compute / real-time budget) | 0.39% | 13.4% |
| Compute per 1s of audio | ~3.85 ms | ~133.7 ms |
| `qDetector` overflow | 0 | 0 |
| Detections | 2 (threshold uncalibrated, see below) | 0 (threshold uncalibrated, see below) |

**DrongoNet Micro is ~35x more compute-intensive per second of audio
than Goertzel** (133.7ms vs. 3.85ms), as expected given a full mel-
spectrogram + CNN pipeline vs. a single 16-multiply-add Goertzel block
-- but **both comfortably sustain real-time** even with MP3 encode + SD
write running concurrently on the other core: zero `qDetector` drops
for either, meaning neither detector ever fell behind the live audio
stream during this run.

## Concurrent-load cost on DrongoNet Micro (vs. idle-chip baseline)

| Stage | Idle-chip (M5STACK_INFERENCE_LATENCY.md) | Under this benchmark's concurrent load | Delta |
|---|---:|---:|---:|
| Mel | ~294.0 ms | ~314.5-316.4 ms | ~+7% |
| Inference | ~82.7 ms | ~86.3-87.4 ms | ~+5% |

A real, measurable but modest contention cost from Core 1's MP3/SD
activity -- not the dominant factor in either stage's latency. Given
only Core 0 touches PSRAM in this design (Core 1's MP3/SD buffers stay
in internal SRAM), this slowdown is more likely scheduler-level (FreeRTOS
inter-core signalling, cache effects) than PSRAM bus contention, though
that wasn't independently isolated.

## Core-1 MP3 + SD results (both sketches, same code, near-identical numbers)

| | Goertzel run | DrongoNet run |
|---|---:|---:|
| MP3 frames (of 833 expected @ 30s/576-sample-frames) | 833 | 833 |
| MP3 encode avg | 7884.6-8115.5 us | 8417.2 us |
| `qMp3` overflow | 0 | 0 |
| SD writes | 29-30 | 29 |
| SD write avg/max | 8785.1 / 113571 us | 11197.5 / 22092 us |
| Total MP3 bytes | 119952 (matches 32kbps x 30s almost exactly) | 119952 |

Real-time MP3+SD throughput confirmed stable regardless of which
detector is running on Core 0 -- the two cores' workloads are genuinely
independent in this design.

## A real bug found and fixed along the way: task-priority starvation

First `m5_goertzel_bench` run measured MP3 encoding at **1.57 seconds
per 576-sample (36ms of audio) frame** -- physically implausible for
Shine's lightweight fixed-point encoder, and only 15 of an expected ~833
frames completed in 30s. Root cause: `fanoutTask` (audio capture +
fan-out) and `mp3SdTask` (encode + SD write) are both pinned to Core 1;
`fanoutTask` was given higher FreeRTOS priority and never yields (tight
busy-poll on `M5.Mic.isRecording()`, no `delay()`/blocking call), so it
never dropped out of the "ready" state and `mp3SdTask` was almost
completely starved of CPU time -- not a Shine performance problem.
**Fix: equal priority for both tasks**, letting FreeRTOS's round-robin
time-slicing share Core 1 fairly. After the fix: 833/833 frames, 0
drops, ~8ms/frame (comfortably within the 36ms budget) -- confirmed
stable across three separate runs (two Goertzel, one DrongoNet). Kept in
the code as an explicit comment at the `xTaskCreatePinnedToCore` call
site in both sketches, since this exact bug shape (same-core producer
starving a same-core consumer via priority + no yield point) will recur
if either sketch is modified without re-reading that comment.

## MAG_THRESHOLD / BIRD_SCORE_THRESHOLD -- still uncalibrated

- **Goertzel's `MAG_THRESHOLD`**: updated from the Portenta's
  meaningless-here placeholder (150) to 2000, informed by this session's
  own idle-noise data (steady-state 1s peak magnitudes clustered
  800-1300 across 29 windows). Detections dropped from ~5000+ (with 150)
  to 2 (with 2000) -- clearing the ambient noise floor as intended, but
  **this is still not a signal-calibrated threshold** -- no real bird
  call or playback has been run against this hardware. One reproducible
  startup outlier (44003.4, first 1s window only, consistent across two
  separate runs) is likely an I2S PDM decimation-filter warm-up
  transient, not investigated further since it's a one-off, not
  steady-state behavior.
- **DrongoNet Micro's `BIRD_SCORE_THRESHOLD`**: left at the Portenta
  Nano's calibrated value (0.8) as a placeholder -- Micro itself was
  never independently calibrated on any hardware in this repo, Portenta
  included. All 10 windows this run scored 0.30-0.53 (0 detections),
  plausible for a quiet room but not confirmed against a known bird-call
  positive.

**Before trusting any detection *count* from either sketch** (as
opposed to the latency/throughput numbers above, which don't depend on
threshold calibration at all), run a real bird-call/playback calibration
pass on this hardware for both thresholds.
