# ARGUS

**Asymmetric Responsive Guardian for Uninterrupted Species Monitoring**

ARGUS is an uninterrupted, two-tier acoustic biologger designed for
long-term wildlife monitoring in remote environments.

Inspired by *Argus Panoptes* --- the guardian who never slept --- ARGUS
listens continuously at low power and activates high-intelligence
processing only when a biological event occurs.

**Goal:** >98% recall of bird activity while enabling months of
deployment on a single battery.

------------------------------------------------------------------------

## Why ARGUS?

Field monitoring systems face a difficult trade-off:

-   Continuous listening drains batteries
-   Aggressive power saving risks missing rare events
-   Storing everything creates massive data volume

ARGUS solves this with asymmetric intelligence:

-   A lightweight **Sentinel** listens continuously
-   A powerful **Classifier** wakes only when needed

No missed birds. No wasted power.

------------------------------------------------------------------------

## System Architecture

ARGUS runs on a dual-core microcontroller (Cortex-M4 + Cortex-M7).

### Tier 1 --- Sentinel (Cortex-M4)

-   Always listening at ultra-low power
-   Lightweight bird activity detection
-   3-second wake cycles via RTC
-   Activates Tier 2 only when bird presence is detected

**Priority:** Maximize recall.

------------------------------------------------------------------------

### Tier 2 --- Classifier (Cortex-M7)

-   Wakes only upon gatekeeper trigger
-   Performs species-level classification
-   Logs metadata or stores audio
-   Returns system to low-power state

**Priority:** Accuracy and efficiency.

------------------------------------------------------------------------

![ARGUS System Architecture](./argus_arch.svg)


------------------------------------------------------------------------

## Core Components

### 1. SEABAD Dataset

-   50,000 labeled 3-second clips (16 kHz)
-   Binary: Bird / No Bird
-   Optimized to prevent false negatives

https://github.com/mun3im/seabad

------------------------------------------------------------------------

### 2. DrongoNet (Tier 1 Bird Activity Detector)

-   Custom 4-layer CNN (<8 KB INT8)
-   Input: 16×184 mel spectrogram (`n_fft=512`)
-   <8 ms inference on Cortex-M4
-   ≥99% recall target

------------------------------------------------------------------------

### 3. MyGardenBird Dataset

-   7,200 labeled clips
-   12 Southeast Asian species
-   Ornithologist-verified

https://github.com/mun3im/mygardenbird

------------------------------------------------------------------------

### 4. MynaNet (Tier 2 Classifier)

-   Depthwise Separable CNN + SE + Attention
-   Input: 64×300 mel spectrogram (`n_fft=1024`)
-   92% top-1 accuracy
-   <45 ms inference on Cortex-M7
-   Unknown species → audio saved for review

https://github.com/mun3im/mynanet

------------------------------------------------------------------------

### 5. Runtime System

-   Platform: Arduino Portenta H7
-   FreeRTOS-based scheduling
-   RPC inter-core communication
-   DMA audio pipeline
-   Average current: <85 µA (99.9% sleep duty cycle)

------------------------------------------------------------------------

## Deployment Context

Designed for Southeast Asian rainforest monitoring where:

-   Power is limited
-   Long-term autonomy is required
-   High recall is critical

ARGUS performs edge inference directly on embedded hardware --- no cloud
required.

------------------------------------------------------------------------

## Advantages

-   Two-tier ultra-low-power design
-   ≥99% bird activity recall
-   On-demand species classification
-   Unknown species capture for dataset growth
-   Hardware-aware ML architecture

------------------------------------------------------------------------

## License

MIT

------------------------------------------------------------------------

## Citation

Publication forthcoming.
