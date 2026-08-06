// mel_tables_80_edge.h -- 80-mel filterbank for DrongoNet-edge, renamed from
// src/DrongoNet/DrongoNet_Edge/mel_tables_80.h to coexist in the same
// translation unit as mel_tables.h (16-mel, shared by nano+micro): both
// upstream files define MEL_N_MELS/MelFilterDef/melFilters with different
// values, which would collide if included together unmodified. FFT/hop/sr
// (MEL_N_FFT, MEL_HOP_LENGTH, MEL_N_FFT_BINS, MEL_FMIN/FMAX/SR) are
// identical across all three DrongoNet variants and already come from
// mel_tables.h -- not redefined here.
//
// Generated: 2026-06-11 16:07:39
// Parameters: sr=16000, n_fft=1024, n_mels=80, fmin=100.0, fmax=8000.0
// DO NOT EDIT -- regenerate with: python generate_mel_tables.py --n_mels 80

#ifndef EDGE_MEL_TABLES_80_H
#define EDGE_MEL_TABLES_80_H

#define EDGE_MEL_N_MELS 80

typedef struct {
    int startBin;
    int peakBin;
    int endBin;
} EdgeMelFilterDef;

static const EdgeMelFilterDef edgeMelFilters[EDGE_MEL_N_MELS] = {
    {   6,    7,    9},
    {   7,    9,   11},
    {   9,   11,   12},
    {  11,   12,   14},
    {  12,   14,   16},
    {  14,   16,   18},
    {  16,   18,   20},
    {  18,   20,   21},
    {  20,   21,   23},
    {  21,   23,   26},
    {  23,   26,   28},
    {  26,   28,   30},
    {  28,   30,   32},
    {  30,   32,   34},
    {  32,   34,   37},
    {  34,   37,   39},
    {  37,   39,   42},
    {  39,   42,   44},
    {  42,   44,   47},
    {  44,   47,   50},
    {  47,   50,   53},
    {  50,   53,   56},
    {  53,   56,   59},
    {  56,   59,   62},
    {  59,   62,   65},
    {  62,   65,   68},
    {  65,   68,   72},
    {  68,   72,   75},
    {  72,   75,   79},
    {  75,   79,   82},
    {  79,   82,   86},
    {  82,   86,   90},
    {  86,   90,   94},
    {  90,   94,   98},
    {  94,   98,  103},
    {  98,  103,  107},
    { 103,  107,  112},
    { 107,  112,  116},
    { 112,  116,  121},
    { 116,  121,  126},
    { 121,  126,  131},
    { 126,  131,  137},
    { 131,  137,  142},
    { 137,  142,  148},
    { 142,  148,  153},
    { 148,  153,  159},
    { 153,  159,  165},
    { 159,  165,  172},
    { 165,  172,  178},
    { 172,  178,  185},
    { 178,  185,  192},
    { 185,  192,  199},
    { 192,  199,  206},
    { 199,  206,  214},
    { 206,  214,  221},
    { 214,  221,  229},
    { 221,  229,  238},
    { 229,  238,  246},
    { 238,  246,  255},
    { 246,  255,  264},
    { 255,  264,  273},
    { 264,  273,  283},
    { 273,  283,  292},
    { 283,  292,  303},
    { 292,  303,  313},
    { 303,  313,  324},
    { 313,  324,  335},
    { 324,  335,  346},
    { 335,  346,  358},
    { 346,  358,  370},
    { 358,  370,  382},
    { 370,  382,  395},
    { 382,  395,  408},
    { 395,  408,  422},
    { 408,  422,  436},
    { 422,  436,  450},
    { 436,  450,  465},
    { 450,  465,  480},
    { 465,  480,  496},
    { 480,  496,  512},
};

#endif // EDGE_MEL_TABLES_80_H
