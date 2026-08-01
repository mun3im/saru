# h7_goertzel

A classic-DSP, non-ML "pretend it's a bird" chirp detector, split across
the Portenta H7's two cores and wired together over `RPC`. Exploratory
sketch pair — not part of the production `src/ARGUS` cascade.

## What it does

- **`goertzel_m4/`** samples the analog mic on `A0` at 16 kHz and runs a
  single [Goertzel filter](https://en.wikipedia.org/wiki/Goertzel_algorithm)
  (a single-bin DFT — cheap, fixed-frequency energy detector, no FFT or
  model needed) centred at 3 kHz with a 16-sample window. On a rising edge
  past `MAG_THRESHOLD`, it does `RPC.call("handleBird", "BIRD")`. It also
  reports the peak filter magnitude once a second via
  `RPC.call("reportMagnitude", ...)` for live threshold calibration.
- **`goertzel_m7/`** binds both RPC calls: `handleBird` prints `BIRD` to
  `Serial` (115200) and flashes the red LED; `reportMagnitude` prints the
  peak magnitude. Both are visible over USB serial/tty — no IDE required,
  `picocom -b 115200 /dev/ttyACM0` (or any other terminal) works.

## Findings

The filter went through two rounds of tuning, both driven by live
measurement rather than guesswork:

1. **Hardware sanity check first.** Before touching the DSP, a separate
   `adc_diag_rpc/` pair (min/max ADC swing reported once a second) was
   used to confirm the mic itself was producing a real signal — idle swing
   ~54 (511–565) vs. a healthy margin when making noise. Worth doing this
   before debugging any filter: a filter can't fix a dead/unwired mic.
2. **Started from AudioMoth's actual defaults.** [AudioMoth](https://www.openacousticdevices.info/)'s
   own frequency-trigger Goertzel filter defaults to a 64-sample window
   centred at Fs/4 (confirmed from `OpenAcousticDevices/AudioMoth-Configuration-App`'s
   source). At our 16 kHz, that's a 4 kHz centre with a narrow ~250 Hz-wide
   passband (Q≈16). It reliably detected a loud ebird.com playback
   (idle magnitude ~65–99, signal up to 6063) — but a live magnitude
   readout showed content around 2 kHz barely moved the needle
   (peak ~200), because the passband was too narrow/high-Q for broader
   species coverage.
3. **Widened for coverage, not just loudness.** Dropped to a 16-sample
   window (AudioMoth's own narrowest/widest-passband option) re-centred at
   3 kHz — Fs/N = 1000 Hz per bin (Q≈3) — trading selectivity for a much
   wider response. Re-measured: idle ~42–96, real signal ~249–550.
   `MAG_THRESHOLD = 400` sits with ~4x margin above the idle floor and
   reliably triggers `BIRD` on real audio.

Current working parameters (`goertzel_m4/goertzel_m4.ino`):

| Parameter | Value | Why |
|---|---|---|
| `SAMPLE_RATE_HZ` | 16000 | requested Fs |
| `BLOCK_SIZE` (N) | 16 | AudioMoth's narrowest/widest-passband window option |
| `TARGET_FREQ_HZ` | 3000 | re-centred to cover ~2 kHz content the narrower filter missed |
| `MAG_THRESHOLD` | 400 | measured: idle ~42-96, signal ~249-550 |
| `ADC_MIDPOINT` | 512 | this board's measured idle ADC centre (10-bit `analogRead`) |

If detection quality regresses again (new mic, new enclosure, different
species), re-add live measurement rather than re-guessing constants — see
the `reportMagnitude` RPC call already wired in above.

## Relevance to ARGUS's dual-tier orchestration

`src/ARGUS/ARGUS.ino` (the production sketch) currently runs its entire
DrongoNet → MynaNet cascade single-threaded on the M7 core; the M4 core is
idle. But the design docs (`HOW_ARGUS_WORKS.md`) describe the *intended*
architecture as a Cortex-M4 gatekeeper that RPC-triggers a wake on the
Cortex-M7 classifier when it detects activity — the same asymmetric,
recall-biased duty-cycling idea DrongoNet/MynaNet embody in software, just
split across physical cores instead of run sequentially on one.

This sketch pair is a minimal, working proof of that exact mechanism: a
cheap, always-on Tier-1-style detector on M4 (Goertzel instead of a CNN)
that stays quiet until it sees something interesting, then wakes M7 over
`RPC` — not a candidate replacement for DrongoNet's spectrogram/CNN
approach (a fixed-frequency Goertzel filter has none of a CNN's
generalization across call shapes/species), but a validated reference for
the M4→M7 RPC wake-up plumbing itself, decoupled from any model code.
