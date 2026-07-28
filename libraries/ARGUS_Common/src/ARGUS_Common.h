// ARGUS_Common.h — shared timing, math, and memory-profiling helpers.
//
// Factored out of ARGUS.ino, DrongoNet_Micro/Drongonet.ino,
// DrongoNet_Nano/DrongoNet_Nano.ino, and MynaNet/MynaNet.ino, which each
// carried a byte-for-byte (or whitespace-only) copy of this code. Header-only
// and `inline` throughout so it is safe to include from more than one sketch
// translation unit without multiple-definition link errors.
//
// Requires: Arduino core for Portenta H7 (CMSIS DWT registers, Mbed OS
// mbed_stats_*), so it is written for CORE_CM7 / CORE_CM4 targets and
// mirrors the architecture list in library.properties.

#ifndef ARGUS_COMMON_H
#define ARGUS_COMMON_H

#include <Arduino.h>
#include <mbed.h>
#include <mbed_stats.h>
#include <cstring>

// ─────────────────────────────────────────────────────────────────────────
// DWT Cycle Counter — µs-accurate timing without Serial overhead
// ─────────────────────────────────────────────────────────────────────────
// Cortex-M7 DWT.CYCCNT is a 32-bit free-running counter at the core clock.
// On Portenta H7 M7 core @ 480 MHz: 1 cycle = 2.083 ns -> divide by 480 for us.
// Overflow every ~8.95 s; subtraction is safe via uint32 modular arithmetic
// for windows < 8 s (all pipeline stages in this project are << 1 s).
//
// WHY DWT instead of micros():
//   micros() has ~4 us resolution and includes Mbed OS scheduler overhead.
//   Serial.print() calls add 5-15 ms per line if they straddle a timed region.
//   DWT captures cycles BEFORE any print, eliminating UART-induced inflation.
#define DWT_ENABLE()  do { CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk; \
                           DWT->CYCCNT = 0; DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk; } while(0)
#define DWT_RESET()   do { DWT->CYCCNT = 0; } while(0)
#define DWT_CYCLES()  (DWT->CYCCNT)
#define DWT_US(cyc)   ((float)(cyc) / 480.0f)   // us at 480 MHz

// ─────────────────────────────────────────────────────────────────────────
// fast_log10f — IEEE-754 exponent trick (~10 cycles vs ~150 for libm log10f)
// ─────────────────────────────────────────────────────────────────────────
// Error < 0.5 dB — negligible after min-max normalisation.
// Reference: Schraudolph (1999), Neural Computation.
inline float argus_fast_log10f(float x) {
  if (x <= 0.0f) return -100.0f;  // guard: return floor dB
  uint32_t bits;
  memcpy(&bits, &x, sizeof(bits));  // type-safe bit reinterpret
  int   exp  = (int)((bits >> 23) & 0xFF) - 127;
  float mant = (float)(bits & 0x7FFFFF) * (1.0f / (float)0x800000);
  // log2(x) ~ exp + mant  (linear interpolation of mantissa)
  return (exp + mant) * 0.30103f;  // log10 = log2 x log10(2)
}

// ─────────────────────────────────────────────────────────────────────────
// StageRAM — per-stage RAM snapshot for transient-memory profiling
// ─────────────────────────────────────────────────────────────────────────
struct ArgusStageRAM {
  uint32_t heap_current;   // live heap bytes at snapshot time
  uint32_t heap_peak;      // heap high-water mark (max_size)
  uint32_t stack_hwm;      // main-thread stack high-water mark
  uint32_t stack_reserved; // main-thread reserved stack bytes
};

// Snapshot Mbed RAM stats for the main (loop) thread. The main thread is
// identified as the one with the largest reserved_size, which is the
// Arduino sketch thread.
inline ArgusStageRAM argus_snapshot_ram() {
  ArgusStageRAM s = {0, 0, 0, 0};

  mbed_stats_heap_t heap;
  mbed_stats_heap_get(&heap);
  s.heap_current = (uint32_t)heap.current_size;
  s.heap_peak    = (uint32_t)heap.max_size;

  int cnt = osThreadGetCount();
  if (cnt <= 0 || cnt > 32)
    return s; // sanity gate

  mbed_stats_stack_t *stk =
      (mbed_stats_stack_t *)malloc(cnt * sizeof(mbed_stats_stack_t));
  if (!stk)
    return s;

  int got = mbed_stats_stack_get_each(stk, cnt);
  uint32_t best_reserved = 0;
  for (int i = 0; i < got; i++) {
    if (stk[i].reserved_size > best_reserved) {
      best_reserved    = stk[i].reserved_size;
      s.stack_hwm      = stk[i].max_size;
      s.stack_reserved = stk[i].reserved_size;
    }
  }
  free(stk);
  return s;
}

// Per-stage transient RAM delta logger. Call before and after each
// pipeline stage; prints the ADDITIONAL stack/heap depth that stage used.
inline void argus_print_stage_delta(const char *label, const ArgusStageRAM &before,
                                    const ArgusStageRAM &after) {
  Serial.print(F("  Stage ["));
  Serial.print(label);
  Serial.print(F("]  stack HWM delta = +"));
  int32_t delta = (int32_t)after.stack_hwm - (int32_t)before.stack_hwm;
  Serial.print(delta >= 0 ? delta : 0);
  Serial.print(F(" B   heap peak delta = +"));
  int32_t hd = (int32_t)after.heap_peak - (int32_t)before.heap_peak;
  Serial.println(hd >= 0 ? hd : 0);
}

// ─────────────────────────────────────────────────────────────────────────
// System hardware storage breakdown (Flash & RAM totals + linker-symbol
// flash usage estimate)
// ─────────────────────────────────────────────────────────────────────────
// Mbed OS system statistics provide the total physical limits. GCC linker
// symbols provide the actual compiled footprint.
//
// NOTE: __etext, __data_start__, __data_end__ are provided by the Mbed OS
// bootstrap linker script for this target but are not declared in any
// public Arduino/Mbed header, so C++ requires an explicit extern
// declaration before use -- omitting it is a compile error
// ("'__etext' was not declared in this scope"), not a silent no-op. This
// was the actual bug in the pre-refactor copies of this function in
// ARGUS.ino, Drongonet.ino, and DrongoNet_Nano.ino.
extern "C" char __etext;
extern "C" char __data_start__;
extern "C" char __data_end__;

inline void argus_print_system_storage() {
  Serial.println(F("\n╔════════════════════════════════════════╗"));
  Serial.println(F("║      SYSTEM STORAGE (FLASH & RAM)        ║"));
  Serial.println(F("╚════════════════════════════════════════╝"));

  mbed_stats_sys_t sys_stats;
  mbed_stats_sys_get(&sys_stats);

  // Mbed OS supports multiple memory regions. Sum them for total capacity.
  uint32_t total_flash = 0;
  uint32_t total_ram   = 0;
  for (int i = 0; i < MBED_MAX_MEM_REGIONS; i++) {
    total_flash += sys_stats.rom_size[i];
    total_ram   += sys_stats.ram_size[i];
  }

  Serial.println(F("  [HARDWARE TOTALS]"));
  Serial.print(F("    Total ROM (Flash) : "));
  Serial.print(total_flash / 1024.0f, 2); Serial.println(F(" KB"));
  Serial.print(F("    Total RAM         : "));
  Serial.print(total_ram / 1024.0f, 2); Serial.println(F(" KB"));

#if defined(__GNUC__)
  // Used Flash = Text Section (Code/Const) + Data Section (Initialised vars
  // copied to RAM). sys_stats.rom_start[0] gives the primary flash base.
  uint32_t text_size  = (uint32_t)&__etext - sys_stats.rom_start[0];
  uint32_t data_size  = (uint32_t)&__data_end__ - (uint32_t)&__data_start__;
  uint32_t used_flash = text_size + data_size;
  uint32_t free_flash = total_flash > used_flash ? total_flash - used_flash : 0;

  Serial.println(F("\n  [FLASH USAGE ESTIMATE (from Linker Symbols)]"));
  Serial.print(F("    Code/Const (.text): ")); Serial.print(text_size); Serial.println(F(" B"));
  Serial.print(F("    Init Data (.data) : ")); Serial.print(data_size); Serial.println(F(" B"));
  Serial.print(F("    Total Used Flash  : ")); Serial.print(used_flash); Serial.println(F(" B"));
  Serial.print(F("    Remaining Flash   : ")); Serial.print(free_flash); Serial.print(F(" B ("));
  Serial.print((float)free_flash / 1024.0f, 2); Serial.println(F(" KB)"));
  if (total_flash > 0) {
    Serial.print(F("    Flash Utilization : ")); Serial.print(((float)used_flash / total_flash) * 100.0f, 1); Serial.println(F(" %"));
  }
#endif
}

#endif // ARGUS_COMMON_H
