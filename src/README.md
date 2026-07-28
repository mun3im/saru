# Source Modules

| Module | Description | Guide |
|--------|-------------|-------|
| [ARGUS](./ARGUS/) | Full production cascade system | Root README |
| [DrongoNet](./DrongoNet/) | Tier 1 Bird Activity Detector | Guide 03 |
| [MynaNet](./MynaNet/) | Tier 2 Species Classifier | Guide 04 |
| [analog_mic_test](./analog_mic_test/) | MAX4466 mic smoke test | Guide 01 |
| [ina219](./ina219/) | INA219 power monitor | Guide 02 |

## Shared library

`ARGUS.ino`, `DrongoNet/DrongoNet_Micro`, `DrongoNet/DrongoNet_Nano`,
`DrongoNet/DrongoNet_Edge`, and `MynaNet.ino` all `#include <ARGUS_Common.h>`
for their DWT timing macros, `fast_log10f`, and RAM/flash profiling helpers,
which previously existed as separate copy-pasted code in every sketch.
Before compiling any of these, copy `../libraries/ARGUS_Common` into your
Arduino `libraries/` folder (or open the sketch from inside a full checkout
of this repo, which the Arduino IDE will also search).
