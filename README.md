# Neural Amp Modeler (NAM) modified by sskalnik @ S3 Sound
All content created by sskalnik @ S3 Sound is the intellectual property of sskalnik @ S3 Sound.

Other content is the intellectual property of its original authors, and subject to its respective license terms as noted.

See LICENSE file for the original NAM license (MIT).

## What's different?
* "Hyper Accuracy" training, using the COMPLEX Architecture invented by [Slammin Mofo](https://www.tone3000.com/slamminmofo)
    * Check out his shops and free NAM checkpoint packs:
        * https://slammincaptures.bigcartel.com
        * https://ko-fi.com/slammincaptures/shop
        * https://www.thegearpage.net/board/index.php?threads/slammin-tonocracy-tonex-nam-captures.2444935
* REVxSTD, REVySTD, and REVyHi Architectures invented by [38](https://www.thegearpage.net/board/index.php?threads/probably-the-best-nam-quality-in-the-world-revxstd.2727799/)
* "Super Input" created by François NEURALNET (uploaded here with their explicit written permission).
* Pre-emphasis A-weighting + low-pass filters (just like AIDA-X), which can produce superior results.
* Additional pre-emphasis filter options, including ITU-468, which is a more modern standard than A-weighting.
* Force both matrix operations and convolutions to use full 32-bit float Tensor Cores.
* Fully customizable arbitrary precision, including full 32-bit FP, bf16-mixed, 16-bit, and more.
* Architecture, precision, `NY`, and many other options are customizable in the GUI trainer.
* GUI trainer no longer silently blanks the user metadata fields. 
* V5 dry input audio file = S3 Sound's "acid test". This is meant for TB-303-style acid synths, not guitars!
* Load any arbitrary file format, sample rate, and bit depth, using `soundfile` instead of `wavio`.
* ReduceLROnPlateau instead of ExponentialLR.
* Loss metrics are embedded in the checkpoint JSON.
* Final validation ESR is labeled separately from training ESR.
* Lightning TensorBoardLogger stores data in a sensible, clear, and logical folder hierarchy.
* Additional callbacks log and monitor more data, including system resource utilization.
* Lots of informative comments.
* Variable names are more descriptive and "Pythonic" (as opposed to `x`, `y`, `i`, `j`, etc.).

## See also:
* https://www.thegearpage.net/board/index.php?threads/nam-hyper-accuracy-captures.2543142
* https://www.thegearpage.net/board/index.php?threads/probably-the-best-nam-quality-in-the-world-revxstd.2727799
* https://thegearforum.com/threads/nam-neural-amp-modeler.1698

## Explanatory notes
### Dry input audio file standards and requirements
The dry input audio file must have the following sections, in this order:
1. Silent Padding
2. Validation 1
3. Silent Padding + Blips + Silent Padding
4. Training data
5. Silent Padding
6. Validation 2
Silent padding at the end of the file is recommended, but not required.

(SP = silence padding):
```
    | SP | Validation 1 | SP Blips SP | Training data | SP | Validation 2 | SP |
    | 0  | v1_start     | blips_start | train_start   |    | v2_start     | -1 |

If SP is the same length in all instances, then you can just make all splits be:
    silence_padded_start_of_each_section = absolute_start_of_each section - (1/2 * len(SP))
    silence_padded_end_of_each_section = absolute_end_of_each section + (1/2 * len(SP))
```
Original NAM README.md contents below:
---
# NAM: Neural Amp Modeler

[![Build](https://github.com/sdatkinson/neural-amp-modeler/actions/workflows/python-package.yml/badge.svg)](https://github.com/sdatkinson/neural-amp-modeler/actions/workflows/python-package.yml)

This repository handles training models and exporting them to .nam files.
For playing trained models in real time in a standalone application or plugin, see the partner repo,
[NeuralAmpModelerPlugin](https://github.com/sdatkinson/NeuralAmpModelerPlugin).

For more information about the NAM ecosystem please check out https://www.neuralampmodeler.com/.

## Documentation
Online documentation can be found here: 
https://neural-amp-modeler.readthedocs.io

To build the documentation locally on a Linux system:
```bash
cd docs
make html
```

Or on Windows,
```
cd docs
make.bat html
```
