// mel_tables.h � Auto-generated mel filterbank tables for Portenta H7
// Generated: 2026-06-04 00:12:31
// Parameters: sr=16000, n_fft=1024, n_mels=16, fmin=100.0, fmax=8000.0
//
// Matches: librosa.filters.mel(sr=16000, n_fft=1024, n_mels=16, fmin=100.0, fmax=8000.0)
// Used in: seabadnet-micro training (6b_micro_final.py)
//
// DO NOT EDIT � regenerate with: python generate_mel_tables.py

#ifndef MEL_TABLES_H
#define MEL_TABLES_H

// ---- Mel Spectrogram Constants ----
#define MEL_N_FFT      1024
#define MEL_HOP_LENGTH  256
#define MEL_N_FFT_BINS  (1024 / 2 + 1)  // 513
#define MEL_N_MELS      16
#define MEL_FMIN        100.0f
#define MEL_FMAX        8000.0f
#define MEL_SR          16000

// ---- Triangular Mel Filter Definition ----
// Each filter is a triangle: ramp up from startBin to peakBin,
//                            ramp down from peakBin to endBin.
// Weight at startBin = 0, at peakBin = 1, at endBin = 0.
// Runtime weight computation:
//   Rising:  w = (k - start) / (peak - start)
//   Falling: w = (end - k) / (end - peak)

typedef struct {
    int startBin;  // First non-zero FFT bin (inclusive)
    int peakBin;   // Peak of triangular filter
    int endBin;    // Last non-zero FFT bin (inclusive)
} MelFilterDef;

static const MelFilterDef melFilters[MEL_N_MELS] = {
    {   6,   14,   23},  // Filter  0:   100.0 �   220.6 �   359.3 Hz  (width=17 bins)
    {  14,   23,   33},  // Filter  1:   220.6 �   359.3 �   519.0 Hz  (width=19 bins)
    {  23,   33,   45},  // Filter  2:   359.3 �   519.0 �   702.7 Hz  (width=22 bins)
    {  33,   45,   58},  // Filter  3:   519.0 �   702.7 �   914.1 Hz  (width=25 bins)
    {  45,   58,   74},  // Filter  4:   702.7 �   914.1 �  1157.3 Hz  (width=29 bins)
    {  58,   74,   92},  // Filter  5:   914.1 �  1157.3 �  1437.2 Hz  (width=34 bins)
    {  74,   92,  112},  // Filter  6:  1157.3 �  1437.2 �  1759.4 Hz  (width=38 bins)
    {  92,  112,  136},  // Filter  7:  1437.2 �  1759.4 �  2130.0 Hz  (width=44 bins)
    { 112,  136,  163},  // Filter  8:  1759.4 �  2130.0 �  2556.5 Hz  (width=51 bins)
    { 136,  163,  195},  // Filter  9:  2130.0 �  2556.5 �  3047.3 Hz  (width=59 bins)
    { 163,  195,  231},  // Filter 10:  2556.5 �  3047.3 �  3612.1 Hz  (width=68 bins)
    { 195,  231,  273},  // Filter 11:  3047.3 �  3612.1 �  4262.0 Hz  (width=78 bins)
    { 231,  273,  320},  // Filter 12:  3612.1 �  4262.0 �  5009.8 Hz  (width=89 bins)
    { 273,  320,  376},  // Filter 13:  4262.0 �  5009.8 �  5870.3 Hz  (width=103 bins)
    { 320,  376,  439},  // Filter 14:  5009.8 �  5870.3 �  6860.5 Hz  (width=119 bins)
    { 376,  439,  512},  // Filter 15:  5870.3 �  6860.5 �  8000.0 Hz  (width=136 bins)
};

// Raw bin edge points (18 points for 16 filters):
// bins = [6, 14, 23, 33, 45, 58, 74, 92, 112, 136, 163, 195, 231, 273, 320, 376, 439, 512]
// Hz   = [100.0, 220.6, 359.3, 519.0, 702.7, 914.1, 1157.3, 1437.2, 1759.4, 2130.0, 2556.5, 3047.3, 3612.1, 4262.0, 5009.8, 5870.3, 6860.5, 8000.0]

#endif // MEL_TABLES_H
