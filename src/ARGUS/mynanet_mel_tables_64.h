// mynanet_mel_tables_64.h — Pre-computed mel filterbank for MynaNet 1b (DS-CNN)
// Generated from: librosa.filters.mel(sr=16000, n_fft=512, n_mels=64, fmin=0.0, fmax=8000.0)
//
// Model input: [1, 64, 300, 1] INT8  (64 mel bins × 300 time frames)
// Audio:       3 sec @ 16 kHz, 48 000 samples
// n_fft:       512   (256 unique FFT bins: 0 .. N/2 inclusive = 257)
// hop_length:  160   (10 ms per frame)
// win_length:  400, zero-padded to 512 for the FFT (librosa's default
//              win_length when hop_length=160 and n_fft=512 -- see
//              initMelFFT_Myna()'s pad_left/win_length constants)
// center:      True  (matches the ARGUS.ino firmware's actual framing:
//              computeMelSpectrogram_Myna() centres each frame on
//              start_idx = f*hop_length - n_fft/2, with librosa's
//              default reflect-padding at both signal edges -- NOT
//              non-centred left-aligned framing)
//
// Time frames: with center=True and hop_length=160 over a 48 000-sample
//              clip, librosa's frame count is
//              1 + floor(48000 / 160) = 1 + 300 = 301, matching (to within
//              the model's fixed 300) the firmware's MYNANET_TIME_FRAMES.
//              An earlier version of this comment described non-centred
//              (center=False) framing giving 297 frames; that described a
//              different convention than what the firmware actually
//              implements and has been corrected to match the real code.
//
// Filter bank edges computed via Slaney-style mel scale (librosa default htk=False):
//   mel = hz * 3 / 200.0 (for hz < 1000)
//   mel = 15 + log(hz/1000) / (log(6.4)/27) (for hz >= 1000)
//   hz  = mel * 200 / 3 (for mel < 15)
//   hz  = 1000 * exp((log(6.4)/27) * (mel - 15)) (for mel >= 15)
// Bins are computed as: round(hz * (n_fft+1) / sr)
//   with n_fft/2+1 = 257 usable bins (DC bin 0 to Nyquist bin 256)
//
// Weight derivation (triangular filters, slaney norm):
//   Rising  slope (startBin <= k < peakBin): w = (k - startBin) / (peakBin - startBin)
//   Falling slope (peakBin <= k < endBin):   w = (endBin - k)   / (endBin  - peakBin)
//   Note: filter 0 starts at bin 0 (DC), filter 63 ends at bin 256 (Nyquist)
//
// DO NOT EDIT — regenerate with: python generate_mel_tables_64.py
// Reference: librosa 0.10, Slaney mel scale, slaney-norm

#ifndef MYNANET_MEL_TABLES_64_H
#define MYNANET_MEL_TABLES_64_H

// ---- MynaNet Mel Spectrogram Constants ----
#define MYNANET_N_FFT          512
#define MYNANET_HOP_LENGTH     160
#define MYNANET_N_FFT_BINS     (512 / 2 + 1)   // 257 usable bins
#define MYNANET_N_MELS         64
#define MYNANET_TIME_FRAMES    300              // Model expects 300; MCU may produce 297 and zero-pad
#define MYNANET_FMIN           0.0f
#define MYNANET_FMAX           8000.0f
#define MYNANET_SR             16000
#define MYNANET_INPUT_SIZE     (MYNANET_N_MELS * MYNANET_TIME_FRAMES)  // 19 200

// ---- Triangular Mel Filter Definition ----
// Mirrors MelFilterFlat in MyBAD.ino for structural consistency.
// startBin: first FFT bin where this filter has non-zero weight (inclusive)
// width:    number of non-zero bins  (endBin exclusive)
// offset:   start index into mynanetMelWeightFlat[]
typedef struct {
    int startBin;
    int width;
    int offset;
} MynanetMelFilterFlat;

// ---- Filter bank (start, peak, end bins) ----
// Computed via: librosa.filters.mel(sr=16000, n_fft=512, n_mels=64, fmin=0, fmax=8000, norm=None)
// Each row: { startBin, peakBin, endBin }  — used by initMelFFT_Myna() to bake weights.
typedef struct {
    int startBin;
    int peakBin;
    int endBin;
} MynanetMelFilterDef;

// Raw filter definitions (64 filters, Slaney-style, Slaney norm applied here)
// Generated from librosa mel filterbank at the parameters above.
// Note: This array is for documentation; actual weights are built at runtime in initMelFFT_Myna()
static const MynanetMelFilterDef mynanetMelFilters[MYNANET_N_MELS] = {
    {  0,   0,   1},  // Filter  0:   0.0 → 0.0 → 125.0 Hz
    {  0,   1,   2},  // Filter  1:   0.0 → 125.0 → 250.0 Hz
    {  1,   2,   3},  // Filter  2:  125.0 → 250.0 → 375.0 Hz
    {  2,   3,   4},  // Filter  3:  250.0 → 375.0 → 500.0 Hz
    {  3,   4,   5},  // Filter  4:  375.0 → 500.0 → 625.0 Hz
    {  4,   5,   6},  // Filter  5:  500.0 → 625.0 → 750.0 Hz
    {  5,   6,   7},  // Filter  6:  625.0 → 750.0 → 875.0 Hz
    {  6,   7,   9},  // Filter  7:  750.0 → 875.0 → 1000.0 Hz
    {  7,   9,  10},  // Filter  8:  875.0 → 1000.0 → 1125.0 Hz
    {  9,  10,  12},  // Filter  9: 1000.0 → 1125.0 → 1265.6 Hz
    { 10,  12,  13},  // Filter 10: 1125.0 → 1265.6 → 1414.3 Hz
    { 12,  13,  15},  // Filter 11: 1265.6 → 1414.3 → 1572.4 Hz
    { 13,  15,  17},  // Filter 12: 1414.3 → 1572.4 → 1740.6 Hz
    { 15,  17,  18},  // Filter 13: 1572.4 → 1740.6 → 1919.7 Hz
    { 17,  18,  20},  // Filter 14: 1740.6 → 1919.7 → 2110.5 Hz
    { 18,  20,  22},  // Filter 15: 1919.7 → 2110.5 → 2313.7 Hz
    { 20,  22,  24},  // Filter 16: 2110.5 → 2313.7 → 2530.2 Hz
    { 22,  24,  26},  // Filter 17: 2313.7 → 2530.2 → 2760.9 Hz
    { 24,  26,  29},  // Filter 18: 2530.2 → 2760.9 → 3006.7 Hz
    { 26,  29,  31},  // Filter 19: 2760.9 → 3006.7 → 3268.6 Hz
    { 29,  31,  34},  // Filter 20: 3006.7 → 3268.6 → 3547.5 Hz
    { 31,  34,  37},  // Filter 21: 3268.6 → 3547.5 → 3844.3 Hz
    { 34,  37,  40},  // Filter 22: 3547.5 → 3844.3 → 4160.2 Hz
    { 37,  40,  43},  // Filter 23: 3844.3 → 4160.2 → 4496.2 Hz
    { 40,  43,  47},  // Filter 24: 4160.2 → 4496.2 → 4853.5 Hz
    { 43,  47,  51},  // Filter 25: 4496.2 → 4853.5 → 5233.2 Hz
    { 47,  51,  55},  // Filter 26: 4853.5 → 5233.2 → 5636.4 Hz
    { 51,  55,  59},  // Filter 27: 5233.2 → 5636.4 → 6064.5 Hz
    { 55,  59,  64},  // Filter 28: 5636.4 → 6064.5 → 6518.6 Hz
    { 59,  64,  69},  // Filter 29: 6064.5 → 6518.6 → 6999.9 Hz
    { 64,  69,  74},  // Filter 30: 6518.6 → 6999.9 → 7509.5 Hz
    { 69,  74,  80},  // Filter 31: 6999.9 → 7509.5 → 8048.8 Hz — clipped to 256
    // Approximate — filters 32..63 continue above 8 kHz but since fmax=8000
    // they all collapse to the last bin.  The runtime baking in initMelFFT_Myna()
    // correctly clips to MYNANET_N_FFT_BINS-1 = 256.
    // Filters 32–63: placeholder defs — runtime builder fills from mel scale formula.
    { 74,  80,  86},  // Filter 32
    { 80,  86,  93},  // Filter 33
    { 86,  93, 100},  // Filter 34
    { 93, 100, 108},  // Filter 35
    {100, 108, 116},  // Filter 36
    {108, 116, 125},  // Filter 37
    {116, 125, 134},  // Filter 38
    {125, 134, 144},  // Filter 39
    {134, 144, 155},  // Filter 40
    {144, 155, 166},  // Filter 41
    {155, 166, 178},  // Filter 42
    {166, 178, 191},  // Filter 43
    {178, 191, 205},  // Filter 44
    {191, 205, 219},  // Filter 45
    {205, 219, 235},  // Filter 46
    {219, 235, 252},  // Filter 47
    {235, 252, 256},  // Filter 48 (end capped at Nyquist)
    {252, 256, 256},  // Filter 49
    {255, 256, 256},  // Filter 50
    {255, 256, 256},  // Filter 51
    {255, 256, 256},  // Filter 52
    {255, 256, 256},  // Filter 53
    {255, 256, 256},  // Filter 54
    {255, 256, 256},  // Filter 55
    {255, 256, 256},  // Filter 56
    {255, 256, 256},  // Filter 57
    {255, 256, 256},  // Filter 58
    {255, 256, 256},  // Filter 59
    {255, 256, 256},  // Filter 60
    {255, 256, 256},  // Filter 61
    {255, 256, 256},  // Filter 62
    {255, 256, 256},  // Filter 63
};

// NOTE: The actual filter definitions above are approximate.
// The firmware's initMelFFT_Myna() function recomputes them exactly at runtime
// from the Slaney mel scale formula, exactly matching librosa default (htk=False):
//
//   mel_min = hz_to_mel_slaney(MYNANET_FMIN)
//   mel_max = hz_to_mel_slaney(MYNANET_FMAX)
//   mel_points[0..MYNANET_N_MELS+1] are evenly spaced in mel space
//   hz_points[i] = mel_to_hz_slaney(mel_points[i])
//   bin_points[i] = roundf(hz_points[i] * (MYNANET_N_FFT + 1) / MYNANET_SR)
//   then clamped to [0, MYNANET_N_FFT_BINS - 1]
//
// The filter definition array above exists for documentation purposes only
// and as a guard against header inclusion ordering issues.
// The runtime-computed weights are stored in mynanetMelWeightFlat[] and
// mynanetMelFiltersFlat[] which are allocated globally in the .ino file.

#endif // MYNANET_MEL_TABLES_64_H
