# File: core.py
# Created Date: Tuesday December 20th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
The core of the "simplified trainer"

Used by the GUI and Colab trainers.
"""

import hashlib as _hashlib
import time
import tkinter as _tk
from copy import deepcopy as _deepcopy
from enum import Enum as _Enum
from functools import partial as _partial
from pathlib import Path as _Path
from time import time as _time
from typing import (
    Dict as _Dict,
    NamedTuple as _NamedTuple,
    Optional as _Optional,
    Sequence as _Sequence,
    Tuple as _Tuple,
    Union as _Union,
)

import matplotlib.pyplot as _plt
import numpy as _np
import pytorch_lightning as _pl
import torch as _torch
from pydantic import BaseModel as _BaseModel
from pytorch_lightning.utilities.warnings import (
    PossibleUserWarning as _PossibleUserWarning,
)
from torch.utils.data import DataLoader as _DataLoader

from ..data import (
    AbstractDataset as _AbstractDataset,
    DataError as _DataError,
    Split as _Split,
    init_dataset as _init_dataset,
    wav_to_np as _wav_to_np,
    wav_to_tensor as _wav_to_tensor,
)
from ..models.exportable import Exportable as _Exportable
from ..models.losses import esr as _ESR
from ..models.metadata import UserMetadata as _UserMetadata
from ..util import filter_warnings as _filter_warnings
from ._version import PROTEUS_VERSION as _PROTEUS_VERSION, Version as _Version
from .lightning_module import LightningModule as _LightningModule
from . import metadata as _metadata
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.pytorch.callbacks import DeviceStatsMonitor


# Force both matrix operations and convolutions to use full 32-bit float Tensor Cores:
# https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html#tensorfloat-32-tf32-on-nvidia-ampere-and-later-devices
_torch.backends.cudnn.conv.fp32_precision = "ieee"


# Training using the simplified trainers in NAM is done at 48k.
STANDARD_SAMPLE_RATE = 48_000.0

# Default number of samples per (mini)batch datum to pull from the model output.
# Lots more info here: https://www.thegearpage.net/board/index.php?threads/nam-hyper-accuracy-captures.2543142/post-43368443
#
# I can't remember if these figures were for 32-true or bf16-mixed...
# Also, I can't remember if these figures were with stock Adam or AdamW8bit...
# TODO: Collect more data and publish stats!
# NY = 64436 (2^16) instead of 8192 (2^13), so 8x larger
# batch_size = 32
# Number of steps per epoch becomes 3.
# ULTRA uses about 15-16 GB of VRAM.
# Complex uses about 21 GB of VRAM.
# Standard uses about 8 GB of VRAM.
#
# NY = 32768 (2^15) instead of 8192 (2^13), so 4x larger
# batch_size = 32
# Number of steps per epoch becomes 7.
# ULTRA uses about 8 GB of VRAM.
# Complex uses about 12 GB of VRAM.
# REVyHI uses about 5 GB of VRAM.
# REVxSTD uses about 4 GB of VRAM.
# Standard uses about 4 GB of VRAM.
_NY_DEFAULT = 8192

_CAB_MRSTFT_PRE_EMPH_WEIGHT = 2.0e-4
_CAB_MRSTFT_PRE_EMPH_COEF = 0.85

_DELAY_CALIBRATION_ABS_THRESHOLD = 0.0003
_DELAY_CALIBRATION_REL_THRESHOLD = 0.001
_DELAY_CALIBRATION_SAFETY_FACTOR = 0


class Architecture(_Enum):
    COMPLEX = "complex"
    REVYHI = "revyhi"
    REVYSTD = 'revystd'
    REVXSTD = "revxstd"
    STANDARD = "standard"
    LITE = "lite"
    FEATHER = "feather"
    NANO = "nano"


class _InputValidationError(ValueError):
    pass


def _detect_input_version(input_path) -> _Tuple[_Version, bool]:
    """
    Check to see if the input matches any of the known inputs

    :return: version, strong match
    """

    def detect_strong(input_path) -> _Optional[_Version]:
        def assign_hash(path):
            # Use this to create hashes for new files
            md5 = _hashlib.md5()
            buffer_size = 65536
            with open(path, "rb") as f:
                while True:
                    data = f.read(buffer_size)
                    if not data:
                        break
                    md5.update(data)
            file_hash = md5.hexdigest()
            return file_hash

        file_hash = assign_hash(input_path)
        print(f"Strong hash: {file_hash}")

        version = {
            "4d54a958861bf720ec4637f43d44a7ef": _Version(1, 0, 0),
            "7c3b6119c74465f79d96c761a0e27370": _Version(1, 1, 1),
            "ede3b9d82135ce10c7ace3bb27469422": _Version(2, 0, 0),
            "36cd1af62985c2fac3e654333e36431e": _Version(3, 0, 0),
            "80e224bd5622fd6153ff1fd9f34cb3bd": _PROTEUS_VERSION,
            "3a126f4bee0627f0941a6e20ec3ea20d": _Version(5, 0, 0),
        }.get(file_hash)
        if version is None:
            print(
                f"Provided input file {input_path} does not strong-match any known "
                "standard input files."
            )
        return version

    def detect_weak(input_path) -> _Optional[_Version]:
        def assign_hash(path):
            Hash = _Optional[str]
            Hashes = _Tuple[Hash, Hash]

            def _hash(x: _np.ndarray) -> str:
                return _hashlib.md5(x).hexdigest()

            def assign_hashes_v1(path) -> Hashes:
                # Use this to create recognized hashes for new files
                x, info = _wav_to_np(path, info=True)
                rate = info.rate
                if rate != _V1_DATA_INFO.rate:
                    return None, None
                # Times of intervals, in seconds
                t_blips = _V1_DATA_INFO.t_blips
                t_sweep = 3 * rate
                t_white = 3 * rate
                t_validation = _V1_DATA_INFO.t_validate
                # v1 and v2 start with 1 blips, sine sweeps, and white noise
                start_hash = _hash(x[: t_blips + t_sweep + t_white])
                # v1 ends with validation signal
                end_hash = _hash(x[-t_validation:])
                return start_hash, end_hash

            def assign_hashes_v2(path) -> Hashes:
                # Use this to create recognized hashes for new files
                x, info = _wav_to_np(path, info=True)
                rate = info.rate
                if rate != _V2_DATA_INFO.rate:
                    return None, None
                # Times of intervals, in seconds
                t_blips = _V2_DATA_INFO.t_blips
                t_sweep = 3 * rate
                t_white = 3 * rate
                t_validation = _V1_DATA_INFO.t_validate
                # v1 and v2 start with 1 blips, sine sweeps, and white noise
                start_hash = _hash(x[: (t_blips + t_sweep + t_white)])
                # v2 ends with 2x validation & blips
                end_hash = _hash(x[-(2 * t_validation + t_blips) :])
                return start_hash, end_hash

            def assign_hashes_v3(path) -> Hashes:
                # Use this to create recognized hashes for new files
                x, info = _wav_to_np(path, info=True)
                rate = info.rate
                if rate != _V3_DATA_INFO.rate:
                    return None, None
                # Times of intervals, in seconds
                # See below.
                end_of_start_interval = 17 * rate  # Start at 0
                start_of_end_interval = -9 * rate
                start_hash = _hash(x[:end_of_start_interval])
                end_hash = _hash(x[start_of_end_interval:])
                return start_hash, end_hash

            def assign_hash_v4(path) -> Hash:
                # Use this to create recognized hashes for new files
                x, info = _wav_to_np(path, info=True)
                rate = info.rate
                if rate != _V4_DATA_INFO.rate:
                    return None
                # I don't care about anything in the file except the starting blip and
                start_hash = _hash(x[: int(1 * _V4_DATA_INFO.rate)])
                return start_hash

            # TODO: I don't know exactly what is going on here lol
            def assign_hashes_v5(path) -> Hashes:
                # Use this to create recognized hashes for new files
                x, info = _wav_to_np(path, info=True)
                rate = info.rate
                if rate != _V5_DATA_INFO.rate:
                    return None, None
                # Times of intervals, in seconds
                # See below.
                end_of_start_interval = 17 * rate  # Start at 0
                start_of_end_interval = -9 * rate
                start_hash = _hash(x[:end_of_start_interval])
                end_hash = _hash(x[start_of_end_interval:])
                return start_hash, end_hash

            start_hash_v1, end_hash_v1 = assign_hashes_v1(path)
            start_hash_v2, end_hash_v2 = assign_hashes_v2(path)
            start_hash_v3, end_hash_v3 = assign_hashes_v3(path)
            hash_v4 = assign_hash_v4(path)
            start_hash_v5, end_hash_v5 = assign_hashes_v5(path)

            return (
                start_hash_v1,
                end_hash_v1,
                start_hash_v2,
                end_hash_v2,
                start_hash_v3,
                end_hash_v3,
                hash_v4,
                start_hash_v5,
                end_hash_v5,
            )

        (
            start_hash_v1,
            end_hash_v1,
            start_hash_v2,
            end_hash_v2,
            start_hash_v3,
            end_hash_v3,
            hash_v4,
            start_hash_v5,
            end_hash_v5,
        ) = assign_hash(input_path)

        print(
            "Weak hashes:\n"
            f" Start (v1) : {start_hash_v1}\n"
            f" End (v1)   : {end_hash_v1}\n"
            f" Start (v2) : {start_hash_v2}\n"
            f" End (v2)   : {end_hash_v2}\n"
            f" Start (v3) : {start_hash_v3}\n"
            f" End (v3)   : {end_hash_v3}\n"
            f" Proteus    : {hash_v4}\n"
            f" Start (v5) : {start_hash_v5}\n"
            f" End (v5)   : {end_hash_v5}\n"
        )

        # Check for matches, starting with most recent. Proteus last since its match is
        # the most permissive.
        version = {
            (
                "dadb5d62f6c3973a59bf01439799809b",
                "8458126969a3f9d8e19a53554eb1fd52",
            ): _Version(3, 0, 0)
        }.get((start_hash_v3, end_hash_v3))
        if version is not None:
            return version
        version = {
            (
                "1c4d94fbcb47e4d820bef611c1d4ae65",
                "28694e7bf9ab3f8ae6ef86e9545d4663",
            ): _Version(2, 0, 0)
        }.get((start_hash_v2, end_hash_v2))
        if version is not None:
            return version
        version = {
            (
                "bb4e140c9299bae67560d280917eb52b",
                "9b2468fcb6e9460a399fc5f64389d353",
            ): _Version(
                1, 0, 0
            ),  # FIXME!
            (
                "9f20c6b5f7fef68dd88307625a573a14",
                "8458126969a3f9d8e19a53554eb1fd52",
            ): _Version(1, 1, 1),
        }.get((start_hash_v1, end_hash_v1))
        if version is not None:
            return version
        version = {"46151c8030798081acc00a725325a07d": _PROTEUS_VERSION}.get(hash_v4)
        return version

    version = detect_strong(input_path)
    if version is not None:
        strong_match = True
        return version, strong_match
    print("Falling back to weak-matching...")
    version = detect_weak(input_path)
    if version is None:
        raise _InputValidationError(
            f"Input file at {input_path} cannot be recognized as any known version!"
        )
    strong_match = False

    return version, strong_match


class _DataInfo(_BaseModel):
    """:param major_version: Data major version"""
    major_version: int
    rate: _Optional[float]
    t_blips: int
    first_blips_start: int
    t_validate: int
    train_start: int
    validation_start: int
    noise_interval: _Tuple[int, int]
    blip_locations: _Sequence[_Sequence[int]]


# ESR will be plotted from 10_000 to 101_000
_V1_DATA_INFO = _DataInfo(
    major_version=1,
    rate=STANDARD_SAMPLE_RATE,
    t_blips=48_000,
    first_blips_start=0,
    t_validate=432_000,
    train_start=0,
    validation_start=-432_000,
    noise_interval=(0, 6000),
    blip_locations=((12_000, 36_000),),
)

# ESR will be plotted from 10_000 to 101_000
# V2:
# (0:00-0:02) Blips at 0:00.5 and 0:01.5
# (0:02-0:05) Chirps
# (0:05-0:07) Noise
# (0:07-2:50.5) General training data
# (2:50.5-2:51) Silence
# (2:51-3:00) Validation 1
# (3:00-3:09) Validation 2
# (3:09-3:11) Blips at 3:09.5 and 3:10.5
_V2_DATA_INFO = _DataInfo(
    major_version=2,
    rate=STANDARD_SAMPLE_RATE,
    t_blips=96_000,
    first_blips_start=0,
    t_validate=432_000,
    train_start=0,
    validation_start=-960_000,  # 96_000 + 2 * 432_000
    noise_interval=(12_000, 18_000),
    blip_locations=((24_000, 72_000), (-72_000, -24_000)),
)

# ESR will be plotted from 10_000 to 101_000
# V3:
# (0:00-0:09) Validation 1
# (0:09-0:10) Silence
# (0:10-0:12) Blips at 0:10.5 and 0:11.5
# (0:12-0:15) Chirps
# (0:15-0:17) Noise
# (0:17-3:00.5) General training data
# (3:00.5-3:01) Silence
# (3:01-3:10) Validation 2
_V3_DATA_INFO = _DataInfo(
    major_version=3,
    rate=STANDARD_SAMPLE_RATE,
    t_blips=96_000,            # time window in which to find the two blips: 2 seconds, with one blip in the middle of each second
    first_blips_start=480_000, # ((first blip position in seconds - .5 seconds) * sample rate)
    t_validate=432_000,        # first 9 seconds are the validation, which is repeated at the end
    train_start=480_000,
    validation_start=-432_000, # first 9 seconds are the validation, which is repeated at the end (note the negative index!)
    noise_interval=(492_000, 498_000), # silence for 125ms in length, 250ms before the first blip, establishing the wet "output" file's noise floor
    blip_locations=((504_000, 552_000),), # Actual locations of each of the two blips
)
# FYI, this is how the validation slicing is done in _check_v3:
#    y = _wav_to_tensor(output_path, rate=rate)
#    n = len(_wav_to_tensor(input_path)) # to End-crop output
#    y_val_1 = y[: _V3_DATA_INFO.t_validate]
#       = 0 : 432_000
#    y_val_2 = y[n - _V3_DATA_INFO.t_validate : n]
#       = (len(input.wav)  - 432_000) : len(input.wav)
#
# FYI, from _get_data_config:
# validation_start = data_info.validation_start
# train_stop = validation_start
# train_kwargs = {"start_samples": data_info.train_start, "stop_samples": train_stop}
# validation_kwargs = {"start_samples": validation_start}

# ESR will be plotted from 10_000 to 101_000
# V4 (aka GuitarML Proteus)
# https://github.com/GuitarML/Releases/releases/download/v1.0.0/Proteus_Capture_Utility.zip
# * 44.1k
# * Odd length...
# * There's a blip on sample zero. This has to be ignored or else over-compensated
#   latencies will come out wrong!
# (0:00-0:01) Blips at 0:00.0 and 0:00.5
# (0:01-0:09) Sine sweeps
# (0:09-0:17) White noise
# (0:17:0.20) Rising white noise (to 0:20.333 appx)
# (0:20-3:30.858) General training data (ends on sample 9,298,872)
# I'm arbitrarily assigning the last 10 seconds as validation data.
_V4_DATA_INFO = _DataInfo(
    major_version=4,
    rate=44_100.0,
    t_blips=44_099,  # Need to ignore the first blip!
    first_blips_start=1,  # Need to ignore the first blip!
    t_validate=441_000,
    # Blips are problematic for training because they don't have preceding silence
    train_start=44_100,
    validation_start=-441_000,
    noise_interval=(6_000, 12_000),
    blip_locations=((22_050,),),
)

# V5 = S3 Sound's "acid test". This is meant for TB-303-style acid synths, not guitars!
#
# ESR will be plotted from 96_000 to 108_000.
# 120 BPM = 2 seconds per bar = 500 ms per 1/4 note = 125 ms per 1/16 note.
# 1/4 = 24_000, 1/8 = 12_000, 1/16 = 6_000 samples @ 48_000 Hz sample rate.
#
# Minutes:Seconds.Samples
# (00:00.000 - 00:02.000) = Silence [0:96_000]
# (00:02.000 - 01:10.000) = Validation 1 [96_000:3_360_000] 1:08 length, 70 * 48000 = 3360000
# (01:10.000 - 01:10.499) = Silence before Blip 1 [3_360_000:3_384_000]
# (01:10.500)             = Blip 1 @ 3_384_000
# (01:10.501 - 01:11:499) = Silence between Blip 1 and Blip 2
# (01:11.500)             = Blip 2 @ 3_432_000
# (01:11.501 - 01:12.000) = Silence between Blip 2 and Operator white noise C3 (discontinuity?)
# (01:12.000 - 09:04.000) = Training data (Technically the final sine sweep ends at 09:04.000) [3_456_000:26_112_000]
# (09:04.000 - 09:04:500) = Silence between sine sweep audio end and Validation 2 audio start
# (09:04.500 - 10:12.500) = Validation 2 [26_136_000:29_400_000]
# (10:12.500 - 10:14.000) = Silence
# (10:14.000)             = EOF @ 29_472_000
_V5_DATA_INFO = _DataInfo(
    major_version=5,
    rate=STANDARD_SAMPLE_RATE,
    t_blips=96_000,               # time window in which to find the two blips: 2 seconds, with one blip in the middle of each second
    first_blips_start=3_360_000, # ((first blip position in seconds - .5 seconds) * sample rate)
    t_validate=3_264_000,         # 01:08.000 = 68 * 48000 = 3_264_000
    train_start=3_456_000,        # 01:12.000 = 72 * 48000 = 3_456_000
    validation_start=-3_360_000,  # -01:09.500 = 1:08.000 + 1.5 seconds trailing silence. ((9 * 60) + 4.5) * 48000 = 26_136_000 absolute position. (29_472_000 - (((9 * 60) + 4.5) * 48000)) / 48000 = 69.5
    noise_interval=(3_372_000, 3_378_000), # silence for 125ms in length, 250ms before the first blip, establishing the wet "output" file's noise floor.
    blip_locations=((3_384_000, 3_432_000),), # Actual locations of each of the two blips
)
# FYI, from _get_data_config:
# validation_start = data_info.validation_start
# train_stop = validation_start
# train_kwargs = {"start_samples": data_info.train_start, "stop_samples": train_stop}
# validation_kwargs = {"start_samples": validation_start}


def _warn_lookaheads(indices: _Sequence[int]) -> str:
    return (
        f"WARNING: delays from some blips ({','.join([str(i) for i in indices])}) are "
        "at the minimum value possible. This usually means that something is "
        "wrong with your data. Check if training ends with a poor result!"
    )


def _calibrate_latency_v_all(
    data_info: _DataInfo,
    y,
    manual_available: bool,
    show_plots: bool,
    abs_threshold=_DELAY_CALIBRATION_ABS_THRESHOLD,
    rel_threshold=_DELAY_CALIBRATION_REL_THRESHOLD,
    safety_factor=_DELAY_CALIBRATION_SAFETY_FACTOR,
    _override_suppress_plots: bool = False,
) -> _metadata.LatencyCalibration:
    """
    Calibrate the delay in teh input-output pair based on blips.
    This only uses the blips in the first set of blip locations!

    :param y: The output audio, in complete.
    """

    def report_any_latency_warnings(
        delays: _Sequence[int],
    ) -> _metadata.LatencyCalibrationWarnings:
        # Warnings associated with any single delay:

        if len(delays) == 0:
            return _metadata.LatencyCalibrationWarnings(
                matches_lookahead=False,
                disagreement_too_high=False,
                not_detected=True,
            )

        # "Lookahead warning": if the delay is equal to the lookahead, then it's
        # probably an error.
        lookahead_warnings = [i for i, d in enumerate(delays, 1) if d == -lookahead]
        matches_lookahead = len(lookahead_warnings) > 0
        if matches_lookahead:
            print(_warn_lookaheads(lookahead_warnings))

        # Ensemble warnings

        # If they're _really_ different, then something might be wrong.
        max_disagreement_threshold = 20
        max_disagreement_too_high = (
            _np.max(delays) - _np.min(delays) >= max_disagreement_threshold
        )
        if max_disagreement_too_high:
            print(
                "WARNING: Latencies are anomalously different from each other (more "
                f"than {max_disagreement_threshold} samples). If this model turns out "
                "badly, then you might need to provide the latency manually."
            )

        return _metadata.LatencyCalibrationWarnings(
            matches_lookahead=matches_lookahead,
            disagreement_too_high=max_disagreement_too_high,
            not_detected=False,
        )

    lookahead = 1_000
    lookback = 10_000
    # Calibrate the level for the trigger:
    end_of_blip_search_range = data_info.first_blips_start + data_info.t_blips
    print(f"Scanning for two dirac pulse 'blips' in sample range {data_info.first_blips_start} to {end_of_blip_search_range}...")
    y = y[data_info.first_blips_start:end_of_blip_search_range]
    # Calculate the max amplitude of the background noise within data_info.noise_interval
    noise_interval_before_dirac_pulse_blip = data_info.noise_interval[0] - data_info.first_blips_start
    noise_interval_after_dirac_pulse_blip = data_info.noise_interval[1] - data_info.first_blips_start
    background_level = _np.max(
        _np.abs(
            y[noise_interval_before_dirac_pulse_blip:noise_interval_after_dirac_pulse_blip]
        )
    )
    print(f"background_level = {background_level} based on a scan from {noise_interval_before_dirac_pulse_blip} to {noise_interval_after_dirac_pulse_blip}")
    print(f"rel_threshold is {rel_threshold}")
    # Which is larger? The background noise from the previous step + abs_threshold, or (1.0 + rel_threshold) * the background noise?
    trigger_threshold = max(
        background_level + abs_threshold,
        (1.0 + rel_threshold) * background_level,
    )
    print(f"Trigger threshold is {trigger_threshold}")

    y_scans = []
    for blip_index, i_abs in enumerate(data_info.blip_locations[0], 1):
        # Relative to start of the data
        i_rel = i_abs - data_info.first_blips_start
        # Start at 1_000 samples before the blip location
        start_looking = i_rel - lookahead
        # Stop at 10_000 samples after the blip location
        stop_looking = i_rel + lookback
        y_scans.append(y[start_looking:stop_looking])
    y_scan_average = _np.mean(_np.stack(y_scans), axis=0)
    triggered = _np.where(_np.abs(y_scan_average) > trigger_threshold)[0]
    if len(triggered) == 0:  # No impulse responses were detected; can't calibrate
        msg = (
            "No response activated the trigger in response to input spikes. "
            "Is something wrong with the reamp? "
            "Check for background noise just before the first blip!"
        )
        if (show_plots or not manual_available) and not _override_suppress_plots:
            print(msg)
            print("SHARE THIS PLOT IF YOU ASK FOR HELP")
            _plt.figure()
            _plt.plot(
                _np.arange(-lookahead, lookback),
                y_scan_average,
                color="C0",
                label="Signal average",
            )
            for y_scan in y_scans:
                _plt.plot(
                    _np.arange(-lookahead, lookback), y_scan, color="C0", alpha=0.2
                )
            _plt.axvline(x=0, color="C1", linestyle="--", label="Trigger")
            _plt.axhline(
                y=-trigger_threshold, color="k", linestyle="--", label="Threshold"
            )
            _plt.axhline(y=trigger_threshold, color="k", linestyle="--")
            _plt.xlim((-lookahead, lookback))
            _plt.xlabel("Samples")
            _plt.ylabel("Response")
            _plt.legend()
            _plt.title("SHARE THIS PLOT IF YOU ASK FOR HELP")
            _plt.show()
        delays = []
        recommended = None

    else:
        delay = triggered[0] + start_looking - i_rel
        delays = [delay]
        recommended = delay - safety_factor
        print(f"Delay based on average is {delay}")
        print(
            f"After aplying safety factor of {safety_factor}, the final delay is "
            f"{recommended}"
        )

    warnings = report_any_latency_warnings(delays)

    return _metadata.LatencyCalibration(
        algorithm_version=1,
        delays=delays,
        safety_factor=safety_factor,
        recommended=recommended,
        warnings=warnings,
    )


_calibrate_latency_v1 = _partial(_calibrate_latency_v_all, _V1_DATA_INFO)
_calibrate_latency_v2 = _partial(_calibrate_latency_v_all, _V2_DATA_INFO)
_calibrate_latency_v3 = _partial(_calibrate_latency_v_all, _V3_DATA_INFO)
_calibrate_latency_v4 = _partial(_calibrate_latency_v_all, _V4_DATA_INFO)
_calibrate_latency_v5 = _partial(_calibrate_latency_v_all, _V5_DATA_INFO)


def _plot_latency_v_all(
    data_info: _DataInfo, latency: int, input_path: str, output_path: str, _nofail=True
):
    print("Plotting the latency for manual inspection...")
    x = _wav_to_np(input_path)[data_info.first_blips_start : data_info.first_blips_start + data_info.t_blips]
    y = _wav_to_np(output_path)[data_info.first_blips_start : data_info.first_blips_start + data_info.t_blips]
    # Only get the blips we really want.
    i = _np.where(_np.abs(x) > 0.5 * _np.abs(x).max())[0]
    if len(i) == 0:
        print("Failed to find the spike in the input file.")
        print(
            "Plotting the input and output; there should be spikes at around the "
            "marked locations."
        )
        t = _np.arange(data_info.first_blips_start, data_info.first_blips_start + data_info.t_blips)
        expected_spikes = data_info.blip_locations[0]  # For v1 specifically
        fig, axs = _plt.subplots(len((x, y)), 1)
        for ax, curve in zip(axs, (x, y)):
            ax.plot(t, curve)
            [ax.axvline(x=es, color="C1", linestyle="--") for es in expected_spikes]
        _plt.show()
        if _nofail:
            raise RuntimeError("Failed to plot delay")
    else:
        _plt.figure()
        di = 20
        # V1's got not a spike but a longer plateau; take the front of it.
        if data_info.major_version == 1:
            i = [i[0]]
        for e, ii in enumerate(i, 1):
            _plt.plot(
                _np.arange(-di, di),
                y[ii - di + latency : ii + di + latency],
                ".-",
                label=f"Output {e}",
            )
        _plt.axvline(x=0, linestyle="--", color="k")
        _plt.legend()
        _plt.show()  # This doesn't freeze the notebook


_plot_latency_v1 = _partial(_plot_latency_v_all, _V1_DATA_INFO)
_plot_latency_v2 = _partial(_plot_latency_v_all, _V2_DATA_INFO)
_plot_latency_v3 = _partial(_plot_latency_v_all, _V3_DATA_INFO)
_plot_latency_v4 = _partial(_plot_latency_v_all, _V4_DATA_INFO)
_plot_latency_v5 = _partial(_plot_latency_v_all, _V5_DATA_INFO)


class _AnalyzeLatencyError(RuntimeError):
    """Raised when the latency analysis fails."""
    pass


def _analyze_latency(
    user_latency: _Optional[int],
    input_version: _Version,
    input_path: str,
    output_path: str,
    silent: bool = False,
    _override_suppress_plots: bool = False,
) -> _metadata.Latency:
    """
    Use an automatic algorithm to calibrate the latency of the output audio.

    Return alongside the manual latency that the user provided if applicable.
    """
    if input_version.major == 1:
        calibrate, plot = _calibrate_latency_v1, _plot_latency_v1
    elif input_version.major == 2:
        calibrate, plot = _calibrate_latency_v2, _plot_latency_v2
    elif input_version.major == 3:
        calibrate, plot = _calibrate_latency_v3, _plot_latency_v3
    elif input_version.major == 4:
        calibrate, plot = _calibrate_latency_v4, _plot_latency_v4
    elif input_version.major == 5:
        calibrate, plot = _calibrate_latency_v5, _plot_latency_v5
    else:
        raise NotImplementedError(
            f"Input calibration not implemented for input version {input_version}"
        )
    if user_latency is not None:
        print(f"Delay is specified as {user_latency}")
    calibration_output = calibrate(
        _wav_to_np(output_path),
        manual_available=user_latency is not None,
        show_plots=not silent,
        _override_suppress_plots=_override_suppress_plots,
    )
    if not silent and calibration_output.recommended is not None:
        plot(calibration_output.recommended, input_path, output_path)
    return _metadata.Latency(manual=user_latency, calibration=calibration_output)


def get_lstm_config(architecture):
    return {
        Architecture.STANDARD: {
            "num_layers": 1,
            "hidden_size": 24,
            "train_burn_in": 4096,
            "train_truncate": 512,
        },
        Architecture.LITE: {
            "num_layers": 2,
            "hidden_size": 8,
            "train_burn_in": 4096,
            "train_truncate": 512,
        },
        Architecture.FEATHER: {
            "num_layers": 1,
            "hidden_size": 16,
            "train_burn_in": 4096,
            "train_truncate": 512,
        },
        Architecture.NANO: {
            "num_layers": 1,
            "hidden_size": 12,
            "train_burn_in": 4096,
            "train_truncate": 512,
        },
    }[architecture]


def _esr_validation_replicate_msg(threshold: float) -> str:
    return (
        f"Validation replicates have a self-ESR of over {threshold}. "
        "Your gear doesn't sound like itself when played twice!\n\n"
        "Possible causes:\n"
        " * Your signal chain is too noisy.\n"
        " * There's a time-based effect (chorus, delay, reverb) turned on.\n"
        " * Some knob got moved while reamping.\n"
        " * You started reamping before the amp had time to warm up fully."
    )


def _check_v1(*args, **kwargs) -> _metadata.DataChecks:
    return _metadata.DataChecks(version=1, passed=True)


def _check_v2(
    input_path, output_path, delay: int, silent: bool
) -> _metadata.DataChecks:
    with _torch.no_grad():
        print("V2 checks...")
        rate = _V2_DATA_INFO.rate
        y = _wav_to_tensor(output_path, rate=rate)
        t_blips = _V2_DATA_INFO.t_blips
        t_validate = _V2_DATA_INFO.t_validate
        y_val_1 = y[-(t_blips + 2 * t_validate) : -(t_blips + t_validate)]
        y_val_2 = y[-(t_blips + t_validate) : -t_blips]
        esr_replicate = _ESR(y_val_1, y_val_2).item()
        print(f"Replicate ESR is {esr_replicate:.8f}.")
        esr_replicate_threshold = 0.01
        if esr_replicate > esr_replicate_threshold:
            print(_esr_validation_replicate_msg(esr_replicate_threshold))

        # Do the blips line up?
        # If the ESR is too bad, then flag it.
        print("Checking blips...")

        def get_blips(y):
            """
            :return: [start/end,replicate]
            """
            i0, i1 = _V2_DATA_INFO.blip_locations[0]
            j0, j1 = _V2_DATA_INFO.blip_locations[1]

            i0, i1, j0, j1 = [i + delay for i in (i0, i1, j0, j1)]
            start = -10
            end = 1000
            blips = _torch.stack(
                [
                    _torch.stack([y[i0 + start : i0 + end], y[i1 + start : i1 + end]]),
                    _torch.stack([y[j0 + start : j0 + end], y[j1 + start : j1 + end]]),
                ]
            )
            return blips

        blips = get_blips(y)
        esr_0 = _ESR(blips[0][0], blips[0][1]).item()  # Within start
        esr_1 = _ESR(blips[1][0], blips[1][1]).item()  # Within end
        esr_cross_0 = _ESR(blips[0][0], blips[1][0]).item()  # 1st repeat, start vs end
        esr_cross_1 = _ESR(blips[0][1], blips[1][1]).item()  # 2nd repeat, start vs end

        print("  ESRs:")
        print(f"    Start     : {esr_0}")
        print(f"    End       : {esr_1}")
        print(f"    Cross (1) : {esr_cross_0}")
        print(f"    Cross (2) : {esr_cross_1}")

        esr_threshold = 1.0e-2

        def plot_esr_blip_error(
            show_plot: bool,
            msg: str,
            arrays: _Sequence[_Sequence[float]],
            labels: _Sequence[str],
        ):
            """
            :param silent: Whether to make and show a plot about it
            """
            if show_plot:
                _plt.figure()
                [_plt.plot(array, label=label) for array, label in zip(arrays, labels)]
                _plt.xlabel("Sample")
                _plt.ylabel("Output")
                _plt.legend()
                _plt.grid()
            print(msg)
            if show_plot:
                _plt.show()
            print(
                "This is known to be a very sensitive test, so training will continue. "
                "If the model doesn't look good, then this may be why!"
            )

        # Check consecutive blips
        show_blip_plots = False
        for e, blip_pair, when in zip((esr_0, esr_1), blips, ("start", "end")):
            if e >= esr_threshold:
                plot_esr_blip_error(
                    show_blip_plots,
                    f"Failed consecutive blip check at {when} of training signal. The "
                    "target tone doesn't seem to be replicable over short timespans."
                    "\n\n"
                    "  Possible causes:\n\n"
                    "    * Your recording setup is really noisy.\n"
                    "    * There's a noise gate that's messing things up.\n"
                    "    * There's a time-based effect (chorus, delay, reverb) in "
                    "the signal chain",
                    blip_pair,
                    ("Replicate 1", "Replicate 2"),
                )
                return _metadata.DataChecks(version=2, passed=False)
        # Check blips between start & end of train signal
        for e, blip_pair, replicate in zip(
            (esr_cross_0, esr_cross_1), blips.permute(1, 0, 2), (1, 2)
        ):
            if e >= esr_threshold:
                plot_esr_blip_error(
                    show_blip_plots,
                    f"Failed start-to-end blip check for blip replicate {replicate}. "
                    "The target tone doesn't seem to be same at the end of the reamp "
                    "as it was at the start. Did some setting change during reamping?",
                    blip_pair,
                    (f"Start, replicate {replicate}", f"End, replicate {replicate}"),
                )
                return _metadata.DataChecks(version=2, passed=False)
        return _metadata.DataChecks(version=2, passed=True)


def _check_v3(
    input_path, output_path, silent: bool, *args, **kwargs
) -> _metadata.DataChecks:
    with _torch.no_grad():
        print("V3 checks...")
        rate = _V3_DATA_INFO.rate
        y = _wav_to_tensor(output_path, rate=rate)
        n = len(_wav_to_tensor(input_path)) # to End-crop output
        # first 9 seconds
        y_val_1 = y[: _V3_DATA_INFO.t_validate]
        # last 9 seconds
        y_val_2 = y[n - _V3_DATA_INFO.t_validate : n]
        esr_replicate = _ESR(y_val_1, y_val_2).item()
        print(f"Replicate ESR is {esr_replicate:.8f}.")
        esr_replicate_threshold = 0.01
        if esr_replicate > esr_replicate_threshold:
            print(_esr_validation_replicate_msg(esr_replicate_threshold))
            if not silent:
                _plt.figure()
                t = _np.arange(len(y_val_1)) / rate
                _plt.plot(t, y_val_1, label="Validation 1")
                _plt.plot(t, y_val_2, label="Validation 2")
                _plt.xlabel("Time (sec)")
                _plt.legend()
                _plt.title("V3 check: Validation replicate FAILURE")
                _plt.show()
            return _metadata.DataChecks(version=3, passed=False)
    return _metadata.DataChecks(version=3, passed=True)


def _check_v4(
    input_path, output_path, silent: bool, *args, **kwargs
) -> _metadata.DataChecks:
    # Things we can't check:
    # Latency compensation agreement
    # Data replicability
    print("Using Proteus audio file. Standard data checks aren't possible!")
    signal, info = _wav_to_np(output_path, info=True)
    passed = True
    if info.rate != _V4_DATA_INFO.rate:
        print(
            f"Output signal has sample rate {info.rate}; expected {_V4_DATA_INFO.rate}!"
        )
        passed = False
    # I don't care what's in the files except that they're long enough to hold the blip
    # and the last 10 seconds I decided to use as validation
    required_length = int((1.0 + 10.0) * _V4_DATA_INFO.rate)
    if len(signal) < required_length:
        print(
            "File doesn't meet the minimum length requirements for latency compensation and validation signal!"
        )
        passed = False
    return _metadata.DataChecks(version=4, passed=passed)


def _check_v5(
    input_path, output_path, silent: bool, *args, **kwargs
) -> _metadata.DataChecks:
    with _torch.no_grad():
        print("V5 checks...")
        rate = _V5_DATA_INFO.rate

        wet_audio_tensor = _wav_to_tensor(output_path, rate=rate)
        # Minutes:Seconds.Samples
        # (00:00.000 - 00:02.000) = Silence [0:96_000]
        # (00:02.000 - 01:10.000) = Validation 1 [96_000:3_360_000] 1:08 length, 70 * 48000 = 3360000
        # (01:10.000 - 01:10.499) = Silence before Blip 1 [3_360_000:3_384_000]
        # (01:10.500)             = Blip 1 @ 3_384_000
        # (01:10.501 - 01:11:499) = Silence between Blip 1 and Blip 2
        # (01:11.500)             = Blip 2 @ 3_432_000
        # (01:11.501 - 01:12.000) = Silence between Blip 2 and Operator white noise C3 (discontinuity?)
        # (01:12.000 - 09:04.000) = Training data (Technically the final sine sweep ends at 09:04.000) [3_456_000:26_112_000]
        # (09:04.000 - 09:04:500) = Silence between sine sweep audio end and Validation 2 audio start
        # (09:04.500 - 10:12.500) = Validation 2 [26_136_000:29_400_000]
        # (10:12.500 - 10:14.000) = Silence
        # (10:14.000)             = EOF @ 29_472_000
        #
        # t_blips=96_000
        # first_blips_start = 3_360_000
        # t_validate=3_264_000,         # 01:08.000 = 68 * 48000 = 3_264_000
        # train_start=3_456_000,        # 01:12.000 = 72 * 48000 = 3_456_000
        # validation_start=-3_360_000,  # ((9 * 60) + 4.5) * 48000 = 26_136_000 absolute position.
        # noise_interval=(3_372_000, 3_378_000), # silence for 125ms in length, 250ms before the first blip.
        # blip_locations=((3_384_000, 3_432_000),), # Actual locations of each of the two blips.
        wet_audio_validation_segment_1 = wet_audio_tensor[: _V5_DATA_INFO.t_validate]

        dry_audio_tensor_len = len(_wav_to_tensor(input_path))
        print(f"Dry audio input Tensor length = {dry_audio_tensor_len}")
        wet_audio_validation_segment_2 = wet_audio_tensor[(dry_audio_tensor_len + _V5_DATA_INFO.validation_start) : (dry_audio_tensor_len - (1.5 * rate))]
        print("First wet audio segment = [96_000:3_360_000]")
        print(f"Second wet audio segment = [{(dry_audio_tensor_len + _V5_DATA_INFO.validation_start)}:{(dry_audio_tensor_len - (1.5 * rate))}]")
        # Validation (replicate) ESR =
        #   * how much does wet_audio_validation_segment_1 differ from wet_audio_validation_segment_2?
        #   * does either segment drift off-grid from their expected positions relative to the start and end timings of the dry/wet audio files?
        #
        # If there is a big difference, it's because these two copies of the same exact input audio segment run through the physical signal chain at two different times produces inconsistent results.
        # Or it's because there's something mucking up the timing, like an echo or delay effect.
        # There is a threshold for this because obviously analog gear is not deterministic, and maybe your tubes warmed up a bit between the two takes.
        esr_replicate = _ESR(wet_audio_validation_segment_1, wet_audio_validation_segment_2).item()
        print(f"Replicate ESR is {esr_replicate:.8f}.")
        esr_replicate_threshold = 0.01
        if esr_replicate > esr_replicate_threshold:
            print(_esr_validation_replicate_msg(esr_replicate_threshold))
            if not silent:
                _plt.figure()
                t = _np.arange(len(wet_audio_validation_segment_1)) / rate
                _plt.plot(t, wet_audio_validation_segment_1, label="Validation 1")
                _plt.plot(t, wet_audio_validation_segment_2, label="Validation 2")
                _plt.xlabel("Time (sec)")
                _plt.legend()
                _plt.title("V5 check: Validation replicate FAILURE")
                _plt.show()
            return _metadata.DataChecks(version=5, passed=False)
    return _metadata.DataChecks(version=5, passed=True)


def _check_data(
    input_path: str, output_path: str, input_version: _Version, delay: int, silent: bool
) -> _Optional[_metadata.DataChecks]:
    """
    Ensure that everything should go smoothly

    :return: True if looks good
    """
    if input_version.major == 1:
        f = _check_v1
    elif input_version.major == 2:
        f = _check_v2
    elif input_version.major == 3:
        f = _check_v3
    elif input_version.major == 4:
        f = _check_v4
    elif input_version.major == 5:
        f = _check_v5
    else:
        print(f"Checks not implemented for input version {input_version}; skip")
        return None
    out = f(input_path, output_path, delay, silent)
    # Issue 442: Deprecate inputs
    if input_version.major in [1, 2]:
        print(
            f"Input version {input_version} is deprecated and will be removed in "
            "version 0.11 of the trainer. To continue using it, you must ignore checks."
        )
        out.passed = False
    return out


def get_wavenet_config(architecture):
    return {
        # head_size -> channels
        # channels  -> input_size
        #
        # head 3 -> channels 3 -> input 3
        Architecture.COMPLEX: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 32,
                    "head_size": 8,
                    "kernel_size": 3,
                    "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "input_size": 32,
                    "condition_size": 1,
                    "channels": 8,
                    "head_size": 1,
                    "kernel_size": 3,
                    "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True,
                },
            ],
            "head_scale": 0.02,
        },
        Architecture.REVYHI: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 10,
                    "head_size": 10,
                    "kernel_size": 6,
                    "dilations": [1024, 256, 64, 16, 4, 1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "input_size": 10,
                    "condition_size": 1,
                    "channels": 10,
                    "head_size": 10,
                    "kernel_size": 6,
                    "dilations": [1024, 256, 64, 16, 4, 1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "input_size": 10,
                    "condition_size": 1,
                    "channels": 10,
                    "head_size": 10,
                    "kernel_size": 6,
                    "dilations": [1024, 256, 64, 16, 4, 1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "input_size": 10,
                    "condition_size": 1,
                    "channels": 10,
                    "head_size": 10,
                    "kernel_size": 6,
                    "dilations": [1024, 256, 64, 16, 4, 1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "input_size": 10,
                    "condition_size": 1,
                    "channels": 10,
                    "head_size": 1,
                    "kernel_size": 6,
                    "dilations": [1024, 256, 64, 16, 4, 1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True,
                }
            ],
            "head_scale": 0.99
        },
        # Recommended settings: http://coginthemachine.ddns.net/mnt/_namhtml/wnet/namconfig.html
        # "pre_emph_mrstft_weight": 0.0002 # Same as the default of _CAB_MRSTFT_PRE_EMPH_WEIGHT = 2.0e-4
        # "pre_emph_mrstft_coef": 0.85     # Same as the default of _CAB_MRSTFT_PRE_EMPH_COEF = 0.85
        # "lr_scheduler": {{"gamma": 0.9985}
        Architecture.REVYSTD: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 8,
                    "head_size": 8,
                    "kernel_size": 5,
                    "dilations": [1024, 256, 64, 16, 4, 1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False
                },
                {
                    "input_size": 8,
                    "condition_size": 1,
                    "channels": 8,
                    "head_size": 8,
                    "kernel_size": 5,
                    "dilations": [1024, 256, 64, 16, 4, 1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False
                },
                {
                    "input_size": 8,
                    "condition_size": 1,
                    "channels": 8,
                    "head_size": 8,
                    "kernel_size": 5,
                    "dilations": [1024, 256, 64, 16, 4, 1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False
                },
                {
                    "input_size": 8,
                    "condition_size": 1,
                    "channels": 8,
                    "head_size": 8,
                    "kernel_size": 5,
                    "dilations": [1024, 256, 64, 16, 4, 1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False
                },
                {
                    "input_size": 8,
                    "condition_size": 1,
                    "channels": 8,
                    "head_size": 1,
                    "kernel_size": 5,
                    "dilations": [1024, 256, 64, 16, 4, 1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True
                }
            ],
            "head_scale": 0.99
        },
        # Original uses different settings:
        # https://github.com/sdatkinson/neural-amp-modeler/compare/main...38github:neural-amp-modeler:main#diff-2ca4268576a15c724ac7227c5ae5e17ed1e70edd2a86ce943497e82973f91030R921
        # "head_scale": 0.50,
        # _CAB_MRSTFT_PRE_EMPH_WEIGHT = 1.5e-3 # Different from the default of _CAB_MRSTFT_PRE_EMPH_WEIGHT = 2.0e-4
        # _CAB_MRSTFT_PRE_EMPH_COEF = 0.15     # Different from the default of _CAB_MRSTFT_PRE_EMPH_COEF = 0.85
        # "loss": {
        #     "type": "combined_loss",
        #     "loss_weights": {"spectral_mse": 0.9, "perceptual_loss": 0.1, "low_freq_weight": 2.0, "high_freq_weight": 1.0},
        # },
        Architecture.REVXSTD: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 8,
                    "head_size": 8,
                    "kernel_size": 6,
                    "dilations": [729, 243, 81, 27, 9, 3, 1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False
                },
                {
                    "condition_size": 1,
                    "input_size": 8,
                    "channels": 8,
                    "head_size": 8,
                    "kernel_size": 6,
                    "dilations": [729, 243, 81, 27, 9, 3, 1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False
                },
                {
                    "condition_size": 1,
                    "input_size": 8,
                    "channels": 8,
                    "head_size": 8,
                    "kernel_size": 6,
                    "dilations": [729, 243, 81, 27, 9, 3, 1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False
                },
                {
                    "condition_size": 1,
                    "input_size": 8,
                    "channels": 8,
                    "head_size": 1,
                    "kernel_size": 6,
                    "dilations": [729, 243, 81, 27, 9, 3, 1],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True
                }
            ],
            "head_scale": 0.99
        },
        Architecture.STANDARD: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 16,
                    "head_size": 8,
                    "kernel_size": 3,
                    "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                    "activation": "Tanh",
                    "head_bias": False,
                },
                {
                    "condition_size": 1,
                    "input_size": 16,
                    "channels": 8,
                    "head_size": 1,
                    "kernel_size": 3,
                    "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                    "activation": "Tanh",
                    "head_bias": True,
                },
            ],
            "head_scale": 0.02,
        },
        Architecture.LITE: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 12,
                    "head_size": 6,
                    "kernel_size": 3,
                    "dilations": [1, 2, 4, 8, 16, 32, 64],
                    "activation": "Tanh",
                    "head_bias": False,
                },
                {
                    "condition_size": 1,
                    "input_size": 12,
                    "channels": 6,
                    "head_size": 1,
                    "kernel_size": 3,
                    "dilations": [128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                    "activation": "Tanh",
                    "head_bias": True,
                },
            ],
            "head_scale": 0.02,
        },
        Architecture.FEATHER: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 8,
                    "head_size": 4,
                    "kernel_size": 3,
                    "dilations": [1, 2, 4, 8, 16, 32, 64],
                    "activation": "Tanh",
                    "head_bias": False,
                },
                {
                    "condition_size": 1,
                    "input_size": 8,
                    "channels": 4,
                    "head_size": 1,
                    "kernel_size": 3,
                    "dilations": [128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                    "activation": "Tanh",
                    "head_bias": True,
                },
            ],
            "head_scale": 0.02,
        },
        Architecture.NANO: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 4,
                    "head_size": 2,
                    "kernel_size": 3,
                    "dilations": [1, 2, 4, 8, 16, 32, 64],
                    "activation": "Tanh",
                    "head_bias": False,
                },
                {
                    "condition_size": 1,
                    "input_size": 4,
                    "channels": 2,
                    "head_size": 1,
                    "kernel_size": 3,
                    "dilations": [128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                    "activation": "Tanh",
                    "head_bias": True,
                },
            ],
            "head_scale": 0.02,
        },
    }[architecture]


def _get_data_config(
    input_version: _Version,
    input_path: _Path,
    output_path: _Path,
    ny: int,
    latency: int,
) -> dict:
    def get_split_kwargs(data_info: _DataInfo):
        if data_info.major_version == 1:
            train_val_split = data_info.validation_start
            train_kwargs = {"stop_samples": train_val_split}
            validation_kwargs = {"start_samples": train_val_split}
        elif data_info.major_version == 2:
            validation_start = data_info.validation_start
            train_stop = validation_start
            validation_stop = validation_start + data_info.t_validate
            train_kwargs = {"stop_samples": train_stop}
            validation_kwargs = {
                "start_samples": validation_start,
                "stop_samples": validation_stop,
            }
        elif data_info.major_version == 3:
            validation_start = data_info.validation_start
            train_stop = validation_start
            train_kwargs = {"start_samples": 480_000, "stop_samples": train_stop}
            validation_kwargs = {"start_samples": validation_start}
        elif data_info.major_version == 4:
            validation_start = data_info.validation_start
            train_stop = validation_start
            train_kwargs = {"stop_samples": train_stop}
            # Proteus doesn't have silence to get a clean split. Bite the bullet.
            print(
                "Using Proteus files:\n"
                " * There isn't a silent point to split the validation set, so some of "
                "your gear's response from the train set will leak into the start of "
                "the validation set and impact validation accuracy (Bypassing data "
                "quality check)\n"
                " * Since the validation set is different, the ESRs reported for this "
                "model aren't comparable to those from the other 'NAM' training files."
            )
            validation_kwargs = {
                "start_samples": validation_start,
                "require_input_pre_silence": False,
            }
        elif data_info.major_version == 5:
            validation_start = data_info.validation_start
            train_stop = validation_start
            # validation_start=-3_360_000,  # -01:09.500 = 1:08.000 + 1.5 seconds trailing silence.
            # ((9 * 60) + 4.5) * 48000 = 26_136_000 absolute position.
            # (29_472_000 - (((9 * 60) + 4.5) * 48000)) / 48000 = 69.5
            #
            # train_start=3_456_000,        # 01:12.000 = 72 * 48000 = 3_456_000
            #
            # Minutes:Seconds.Samples
            # (01:12.000 - 09:04.000) = Training data (Technically the final sine sweep ends at 09:04.000) [3_456_000:26_112_000]
            # (09:04.000 - 09:04:500) = Silence between sine sweep audio end and Validation 2 audio start
            # (09:04.500 - 10:12.500) = Validation 2 [26_136_000:29_400_000]
            train_kwargs = {"start_samples": data_info.train_start, "stop_samples": 26_136_000}
            validation_kwargs = {"start_samples": data_info.validation_start}
        else:
            raise NotImplementedError(f"kwargs for input version {input_version}")
        return train_kwargs, validation_kwargs

    data_info = {
        1: _V1_DATA_INFO,
        2: _V2_DATA_INFO,
        3: _V3_DATA_INFO,
        4: _V4_DATA_INFO,
        5: _V5_DATA_INFO,
    }[input_version.major]

    train_kwargs, validation_kwargs = get_split_kwargs(data_info)
    data_config = {
        "train": {"ny": ny, **train_kwargs},
        "validation": {"ny": None, **validation_kwargs},
        "common": {
            "x_path": input_path,
            "y_path": output_path,
            "delay": latency,
            "allow_unequal_lengths": True,
        },
    }
    return data_config


def _get_configs(
    input_version: _Version,
    input_path: str,
    output_path: str,
    latency: int,
    epochs: int,
    model_type: str,
    architecture: Architecture,
    ny: int,
    lr: float,
    lr_decay: float,
    batch_size: int,
    fit_mrstft: bool,
):
    # DATA CONFIG
    data_config = _get_data_config(
        input_version=input_version,
        input_path=input_path,
        output_path=output_path,
        ny=ny,
        latency=latency,
    )

    # MODEL CONFIG
    if model_type == "WaveNet":
        model_config = {
            "net": {
                "name": "WaveNet",
                # This should do decently. If you really want a nice model, try turning up
                # "channels" in the first block and "input_size" in the second from 12 to 16.
                "config": get_wavenet_config(architecture),
            },
            "loss": {"val_loss": "esr"},
            "optimizer": {"lr": lr},
            "lr_scheduler": {
                'class': 'ReduceLROnPlateau',
                    'kwargs': {
                        'mode': 'min',
                        'factor': lr_decay,
                        'patience': 5
                    },
                'monitor': 'ESRPREEMPH'
            },
        }
    else:
        model_config = {
            "net": {
                "name": "LSTM",
                "config": get_lstm_config(architecture),
            },
            "loss": {
                "val_loss": "mse",
                "mask_first": 4096,
                "pre_emph_weight": 1.0,
                "pre_emph_coef": 0.85,
            },
            "optimizer": {"lr": 0.01},
            "lr_scheduler": {"class": "ExponentialLR", "kwargs": {"gamma": 0.995}},
        }
    if fit_mrstft:
        model_config["loss"]["pre_emph_mrstft_weight"] = _CAB_MRSTFT_PRE_EMPH_WEIGHT
        model_config["loss"]["pre_emph_mrstft_coef"] = _CAB_MRSTFT_PRE_EMPH_COEF

    # DEVICE CONFIG (part of learning_config)
    if _torch.cuda.is_available():
        device_config = {"accelerator": "gpu", "devices": 1}
    elif _torch.backends.mps.is_available():
        device_config = {"accelerator": "mps", "devices": 1}
    else:
        print("WARNING: No GPU was found. Training will be very slow!")
        device_config = {}
    # LEARNING CONFIG
    # Set `num_workers` to the same number of P-cores in your computer.
    learning_config = {
        "train_dataloader": {
            "batch_size": batch_size,
            "shuffle": True,
            "pin_memory": True,
            "drop_last": True,
            "num_workers": 6,
            "persistent_workers": True
        },
        "val_dataloader": {},
        "trainer": {"max_epochs": epochs, **device_config},
    }
    return data_config, model_config, learning_config


def _get_dataloaders(
    data_config: _Dict, learning_config: _Dict, model: _LightningModule
) -> _Tuple[_DataLoader, _DataLoader]:
    data_config, learning_config = [
        _deepcopy(c) for c in (data_config, learning_config)
    ]
    data_config["common"]["nx"] = model.net.receptive_field
    dataset_train = _init_dataset(data_config, _Split.TRAIN)
    dataset_validation = _init_dataset(data_config, _Split.VALIDATION)
    train_dataloader = _DataLoader(dataset_train, **learning_config["train_dataloader"])
    val_dataloader = _DataLoader(dataset_validation, **learning_config["val_dataloader"])
    return train_dataloader, val_dataloader


def _esr(pred: _torch.Tensor, target: _torch.Tensor) -> float:
    return (_torch.mean(_torch.square(pred - target)).item() / _torch.mean(_torch.square(target)).item())


def _plot(
    model,
    ds,
    window_start: _Optional[int] = None,
    window_end: _Optional[int] = None,
    filepath: _Optional[str] = None,
    silent: bool = False,
) -> float:
    """:return: The ESR"""

    print("Plotting a comparison of your model with the target output...")
    with _torch.no_grad():
        print(f"Dataset x size in samples = {len(ds.x)}")
        print(f"Dataset y size in samples = {len(ds.y)}")
        dataset_x_size_in_seconds = len(ds.x) / 48_000
        print(f"Dataset x size in seconds = {dataset_x_size_in_seconds:.2f}")
        t0 = _time()
        output = model(ds.x).flatten().cpu().numpy()
        print(f"len(model(ds.x).flatten().cpu().numpy()) = {len(output)}")
        t1 = _time()
        print(f"Took {t1 - t0:.2f} sec ({dataset_x_size_in_seconds / (t1 - t0):.2f}x)")

    esr = _esr(_torch.Tensor(output), ds.y)
    # Trying my best to put numbers to it...
    if esr < 0.001:
        esr_comment = "HOLY SHIT BRO!!! Niiiiiiice! 🤘😎"
    elif esr < 0.01:
        esr_comment = "Great!"
    elif esr < 0.035:
        esr_comment = "Not bad!"
    elif esr < 0.1:
        esr_comment = "...This *might* sound ok!"
    elif esr < 0.3:
        esr_comment = "...This probably won't sound great :("
    else:
        esr_comment = "...Something seems to have gone wrong."
    print(f"Error-signal ratio = {esr:.4g}")
    print(esr_comment)

    # figsize in inches * dpi default of 100
    #   (16, 5)      = 1600 x 500 pixels
    #   (19.2, 10.8) = 1920 x 1080 pixels
    #   (9.6, 5.4)   = 960 x 540 pixels
    _plt.figure(figsize=(9.6, 5.4), edgecolor='black')
    _plt.plot(output[window_start:window_end], color='xkcd:orange', linestyle='--', label='Prediction')
    _plt.plot(ds.y[window_start:window_end], color='xkcd:blue', linestyle='solid', label='Target')
    _plt.title(f"ESR={esr:.4g}")
    _plt.xlabel('Time in samples')
    _plt.ylabel('Amplitude normalized to range [1, -1]')
    _plt.legend(fancybox=True, shadow=True)
    _plt.grid()
    if filepath is not None:
        # E.g.:
        #   ./RLROP_ESRPREEMPH/2026_Feb_21_NAM_120_AcidInput_v4_JRAT_nam_48kHz_24int_mono_1773770799.png
        # Where ./RLROP_ESRPREEMPH is the trainer default output path
        # and 2026_Feb_21_NAM_120_AcidInput_v4_JRAT_nam_48kHz_24int_mono is the model name
        # and 2026_Feb_21_NAM_120_AcidInput_v4_JRAT_nam_48kHz_24int_mono_1773770799 is the checkpoint basename
        # defined in:
        #   def _save_checkpoint(self, trainer: _pl.Trainer, filepath: str):
        _plt.savefig(f"{filepath}_{int(time.time())}.png")
    if not silent:
        _plt.show()
    return esr


def _print_nasty_checks_warning():
    print(
        "\n"
        "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
        "X                                                                          X\n"
        "X                                WARNING:                                  X\n"
        "X                                                                          X\n"
        "X       You are ignoring the checks! Your model might turn out bad!        X\n"
        "X                                                                          X\n"
        "X                              I warned you!                               X\n"
        "X                                                                          X\n"
        "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
    )


def _nasty_checks_modal():
    msg = "You are ignoring the checks!\nYour model might turn out bad!"

    root = _tk.Tk()
    root.withdraw()  # hide the root window
    modal = _tk.Toplevel(root)
    modal.geometry("300x100")
    modal.title("Warning!")
    label = _tk.Label(modal, text=msg)
    label.pack(pady=10)
    ok_button = _tk.Button(
        modal,
        text="I can only blame myself!",
        command=lambda: [modal.destroy(), root.quit()],
    )
    ok_button.pack()
    modal.grab_set()  # disable interaction with root window while modal is open
    modal.mainloop()


class _ValidationStopping(_pl.callbacks.EarlyStopping):
    """
    Callback to indicate to stop training if the validation metric is good enough,
    without the other conditions that EarlyStopping usually forces like patience.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = _np.inf


class _ModelCheckpoint(_pl.callbacks.model_checkpoint.ModelCheckpoint):
    """
    Extension to model checkpoint to save a .nam file as well as the .ckpt file.
    """

    def __init__(
        self,
        *args,
        user_metadata: _Optional[_UserMetadata] = None,
        settings_metadata: _Optional[_metadata.Settings] = None,
        data_metadata: _Optional[_metadata.Data] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._user_metadata = user_metadata
        self._settings_metadata = settings_metadata
        self._data_metadata = data_metadata

    _NAM_FILE_EXTENSION = _Exportable.FILE_EXTENSION

    @classmethod
    def _get_nam_filepath(cls, filepath: str) -> _Path:
        """
        Given a .ckpt filepath, figure out a .nam for it.
        """
        if not filepath.endswith(cls.FILE_EXTENSION):
            raise ValueError(
                f"Checkpoint filepath {filepath} doesn't end in expected extension "
                f"{cls.FILE_EXTENSION}"
            )
        return _Path(filepath[: -len(cls.FILE_EXTENSION)] + cls._NAM_FILE_EXTENSION)

    @property
    def _include_other_metadata(self) -> bool:
        return self._settings_metadata is not None and self._data_metadata is not None

    def _save_checkpoint(self, trainer: _pl.Trainer, filepath: str):
        # Save the .ckpt:
        super()._save_checkpoint(trainer, filepath)
        # Save the .nam:
        nam_filepath = self._get_nam_filepath(filepath)
        pl_model: _LightningModule = trainer.model
        nam_model = pl_model.net
        outdir = nam_filepath.parent
        # HACK: Assume the extension
        basename = f"{nam_filepath.name[: -len(self._NAM_FILE_EXTENSION)]}_{int(time.time())}"
        other_metadata = (
            None
            if not self._include_other_metadata
            else {
                _metadata.TRAINING_KEY: _metadata.TrainingMetadata(
                    settings=self._settings_metadata,
                    data=self._data_metadata,
                    validation_esr=None,  # TODO how to get this?
                ).model_dump()
            }
        )
        nam_model.export(
            outdir,
            basename=basename,
            user_metadata=self._user_metadata,
            other_metadata=other_metadata,
        )

    def _remove_checkpoint(self, trainer: _pl.Trainer, filepath: str) -> None:
        super()._remove_checkpoint(trainer, filepath)
        nam_path = self._get_nam_filepath(filepath)
        if nam_path.exists():
            nam_path.unlink()


def get_callbacks(
    threshold_esr: _Optional[float],
    dirpath: _Optional[str],
    user_metadata: _Optional[_UserMetadata] = None,
    settings_metadata: _Optional[_metadata.Settings] = None,
    data_metadata: _Optional[_metadata.Data] = None,
):
    callbacks = [
        _ModelCheckpoint(
            dirpath=dirpath,
            filename="checkpoint_best_{epoch:04d}_{step}_{ESR:.4g}_{ESRPREEMPH:.4g}_{MSE:.3e}",
            save_top_k=3,
            monitor="val_loss",
            every_n_epochs=1,
            user_metadata=user_metadata,
            settings_metadata=settings_metadata,
            data_metadata=data_metadata,
        ),
        _ModelCheckpoint(
            dirpath=dirpath,
            filename="checkpoint_last_{epoch:04d}_{step}",
            every_n_epochs=1,
            user_metadata=user_metadata,
            settings_metadata=settings_metadata,
            data_metadata=data_metadata,
        ),
        DeviceStatsMonitor(cpu_stats=True),
    ]
    if threshold_esr is not None:
        callbacks.append(
            _ValidationStopping(monitor="ESR", stopping_threshold=threshold_esr)
        )
    return callbacks


class TrainOutput(_NamedTuple):
    """
    :param model: The trained model
    :param simplified_trainer_metadata: The metadata summarizing training with the
        simplified trainer.
    """
    model: _Optional[_LightningModule]
    metadata: _metadata.TrainingMetadata


class _FinalLatencyError(ValueError):
    """Raised when the final latency cannot be determined."""
    pass


def _get_final_latency(latency_analysis: _metadata.Latency) -> int:
    """
    Make a decision based on automatic and manual latency values what will be
    used for training.
    """
    user = latency_analysis.manual
    analyzed = latency_analysis.calibration.recommended

    if user is not None:
        if analyzed is not None:
            if user == analyzed:
                print(f"The user latency is same as the analyzed latency ({user}).")
            else:
                print(
                    f"The user latency is different from the analyzed latency ({user} vs {analyzed})."
                )
                print(f"Override the analyzed latency with the user latency.")
        else:
            print(
                f"Cannot automatically analyze the latency. Use the user latency ({user})."
            )

        return user

    if analyzed is not None:
        print(f"Cannot use the user latency. Use the analyzed latency ({analyzed}).")
        return analyzed

    raise _FinalLatencyError(
        "No latency provided and cannot automatically analyze the latency."
    )


def train(
    input_path: str,
    output_path: str,
    train_path: str,
    epochs=300,
    latency: _Optional[int] = None,
    model_type: str = "WaveNet",
    architecture: _Union[Architecture, str] = Architecture.STANDARD,
    precision: str = '32-true',
    batch_size: int = 16,
    ny: int = _NY_DEFAULT,
    lr=0.002,
    lr_decay=0.004,
    seed: _Optional[int] = 0,
    save_plot: bool = True,
    silent: bool = False,
    modelname: str = "model",
    ignore_checks: bool = False,
    local: bool = False,
    fit_mrstft: bool = True,
    threshold_esr: _Optional[bool] = None,
    user_metadata: _Optional[_UserMetadata] = None,
    fast_dev_run: _Union[bool, int] = False,
) -> _Optional[TrainOutput]:
    """
    :param input_path: Full path to input file
    :param output_path: Full path to output file
    :param lr_decay: = 1-gamma for Exponential learning rate decay.
    :param threshold_esr: Stop training if ESR is better than this. Ignore if `None`.
    :param fast_dev_run: One-step training, used for tests.
    """

    if seed is not None:
        _torch.manual_seed(seed)

    # HACK: We need to check the sample rates and lengths of the audio here or else
    # It will look like a bad self-ESR (Issue 473)
    # Can move this into the "v3 checks" once the others are deprecated.
    # And honestly remake this whole thing as a data processing pipeline.
    sample_rate_validation = _check_audio_sample_rates(input_path, output_path)
    if not sample_rate_validation.passed:
        raise ValueError(
            "Different sample rates detected for input "
            f"({sample_rate_validation.input}) and output "
            f"({sample_rate_validation.output}) audio!"
        )
    length_validation = _check_audio_lengths(input_path, output_path)
    if not length_validation.passed:
        raise ValueError(
            "Your recording differs in length from the input file by "
            f"{length_validation.delta_seconds:.2f} seconds. Check your reamp "
            "in your DAW and ensure that they are the same length."
        )

    input_version, strong_match = _detect_input_version(input_path)

    user_latency = latency
    latency_analysis = _analyze_latency(user_latency, input_version, input_path, output_path, silent=silent)
    final_latency = _get_final_latency(latency_analysis)

    data_check_output = _check_data(input_path, output_path, input_version, final_latency, silent)

    if data_check_output is not None:
        if data_check_output.passed:
            print("-Checks passed")
        else:
            print("Failed checks!")
            if ignore_checks:
                if local and not silent:
                    _nasty_checks_modal()
                else:
                    _print_nasty_checks_warning()
            elif not local:  # And not ignore_checks
                print(
                    "(To disable this check, run AT YOUR OWN RISK with "
                    "`ignore_checks=True`.)"
                )
            if not ignore_checks:
                print("Exiting core training...")
                return TrainOutput(
                    model=None,
                    metadata=_metadata.TrainingMetadata(
                        settings=_metadata.Settings(ignore_checks=ignore_checks),
                        data=_metadata.Data(
                            latency=latency_analysis, checks=data_check_output
                        ),
                        validation_esr=None,
                    ),
                )

    data_config, model_config, learning_config = _get_configs(
        input_version,
        input_path,
        output_path,
        final_latency,
        epochs,
        model_type,
        Architecture(architecture),
        ny,
        lr,
        lr_decay,
        batch_size,
        fit_mrstft,
    )
    
    assert ("fast_dev_run" not in learning_config), "fast_dev_run is set as a kwarg to train()"

    print("Starting training. It's time to kick ass and chew bubblegum!")
    # Issue:
    # * Model needs sample rate from data, but data set needs nx from model.
    # * Model is re-instantiated after training anyways.
    # (Hacky) solution: set sample rate in model from dataloader after second
    # instantiation from final checkpoint.
    model = _LightningModule.init_from_config(model_config)
    train_dataloader, val_dataloader = _get_dataloaders(data_config, learning_config, model)
    if train_dataloader.dataset.sample_rate != val_dataloader.dataset.sample_rate:
        raise RuntimeError(
            "Train and validation data loaders have different data set sample rates: "
            f"{train_dataloader.dataset.sample_rate}, "
            f"{val_dataloader.dataset.sample_rate}"
        )
    sample_rate = train_dataloader.dataset.sample_rate
    model.net.sample_rate = sample_rate

    # Put together the metadata that's needed in checkpoints:
    settings_metadata = _metadata.Settings(ignore_checks=ignore_checks)
    data_metadata = _metadata.Data(latency=latency_analysis, checks=data_check_output)

    # /save_dir/name/version/sub_dir/
    # if sub_dir is None:
    #     /save_dir/name/version/
    save_dir = _Path(train_path).joinpath(modelname)
    str_epochs = str(epochs)
    str_hparams = f"{precision}_ny{str(ny)}_batch{str(batch_size)}_lr{str(lr)}_lrdecay{str(lr_decay)}"
    str_architecture = architecture.value
    logger = TensorBoardLogger(save_dir=save_dir, name=str_architecture, version=str_epochs, sub_dir=str_hparams)

    # Save checkpoints here:
    dirpath = _Path(logger.log_dir).joinpath('checkpoints')

    trainer = _pl.Trainer(
        callbacks=get_callbacks(
            threshold_esr,
            user_metadata=user_metadata,
            settings_metadata=settings_metadata,
            data_metadata=data_metadata,
            dirpath=dirpath,
        ),
        default_root_dir=train_path,
        fast_dev_run=fast_dev_run,
        logger=logger,
        precision=precision,
        # https://lightning.ai/docs/pytorch/stable/common/trainer.html#benchmark
        benchmark=True,
        # Enable this for debugging and optimization experiments:
        # https://lightning.ai/docs/pytorch/stable/tuning/profiler_basic.html
        # Also consider: https://lightning.ai/docs/pytorch/stable/api_references.html#profiler
        #profiler="advanced",
        **learning_config["trainer"],
    )
    # Suppress the PossibleUserWarning about num_workers (Issue 345)
    with _filter_warnings("ignore", category=_PossibleUserWarning):
        # THIS IS WHERE TRAINING ACTUALLY OCCURS!
        trainer.fit(model, train_dataloader, val_dataloader)

    # Go to best checkpoint
    best_checkpoint = trainer.checkpoint_callback.best_model_path
    if best_checkpoint != "":
        model = _LightningModule.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,
            **_LightningModule.parse_config(model_config),
        )
    model.cpu()
    model.eval()
    model.net.sample_rate = sample_rate  # Hack, part 2

    def window_kwargs(version: _Version):
        if version.major == 1:
            return dict(
                window_start=100_000,  # Start of the plotting window, in samples
                window_end=101_000,  # End of the plotting window, in samples
            )
        elif version.major == 2:
            # Same validation set even though it's a different spot in the reamp file
            return dict(
                window_start=100_000,  # Start of the plotting window, in samples
                window_end=101_000,  # End of the plotting window, in samples
            )
        elif version.major == 5:
            return dict(
                window_start=96_000,  # Start of the plotting window, in samples
                window_end=120_000,  # End of the plotting window, in samples
            )# Fallback:
        return dict(
            window_start=100_000,  # Start of the plotting window, in samples
            window_end=101_000,  # End of the plotting window, in samples
        )

    try:
        # Suppress the PossibleUserWarning about num_workers (Issue 345)
        with _filter_warnings("ignore", category=_PossibleUserWarning):
            trainer.fit(model, train_dataloader, val_dataloader)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # Always try to export a model, even if training was interrupted
        # Go to best checkpoint
        best_checkpoint = trainer.checkpoint_callback.best_model_path
        if best_checkpoint != "":
            model = _LightningModule.load_from_checkpoint(
                trainer.checkpoint_callback.best_model_path,
                **_LightningModule.parse_config(model_config),
            )
        model.cpu()
        model.eval()
        model.net.sample_rate = sample_rate  # Hack, part 2

        validation_esr = _plot(
            model,
            val_dataloader.dataset,
            filepath=train_path + "/" + f"{modelname}_{str_architecture}_{str_epochs}_{str_hparams}" if save_plot else None,
            silent=silent,
            **window_kwargs(input_version),
        )

        for dl in (train_dataloader, val_dataloader):
            assert isinstance(dl.dataset, _AbstractDataset)
            dl.dataset.teardown()

        return TrainOutput(
            model=model,
            metadata=_metadata.TrainingMetadata(
                settings=settings_metadata,
                data=data_metadata,
                validation_esr=validation_esr,
            ),
        )


class DataInputValidation(_BaseModel):
    passed: bool


def validate_input(input_path) -> DataInputValidation:
    """:return: Could it be validated?"""
    try:
        _detect_input_version(input_path)
        # succeeded...
        return DataInputValidation(passed=True)
    except _InputValidationError as e:
        print(f"Input validation failed!\n\n{e}")
        return DataInputValidation(passed=False)


class _PyTorchDataSplitValidation(_BaseModel):
    """:param msg: On exception, catch and assign. Otherwise None"""
    passed: bool
    msg: _Optional[str]


class _PyTorchDataValidation(_BaseModel):
    passed: bool
    train: _PyTorchDataSplitValidation  # cf Split.TRAIN
    validation: _PyTorchDataSplitValidation  # Split.VALIDATION


class _SampleRateValidation(_BaseModel):
    passed: bool
    input: int
    output: int


class _LengthValidation(_BaseModel):
    passed: bool
    delta_seconds: float


class DataValidationOutput(_BaseModel):
    passed: bool
    passed_critical: bool
    sample_rate: _SampleRateValidation
    length: _LengthValidation
    input_version: str
    latency: _metadata.Latency
    checks: _metadata.DataChecks
    pytorch: _PyTorchDataValidation


def _check_audio_sample_rates(
    input_path: _Path,
    output_path: _Path,
) -> _SampleRateValidation:
    _, x_info = _wav_to_np(input_path, info=True)
    _, y_info = _wav_to_np(output_path, info=True)

    return _SampleRateValidation(
        passed=x_info.rate == y_info.rate,
        input=x_info.rate,
        output=y_info.rate,
    )


def _check_audio_lengths(
    input_path: _Path,
    output_path: _Path,
    max_under_seconds: _Optional[float] = 0.0,
    max_over_seconds: _Optional[float] = 1.0,
) -> _LengthValidation:
    """
    Check that the input and output have the right lengths compared to each
    other.

    :param input_path: Path to input audio
    :param output_path: Path to output audio
    :param max_under_seconds: If not None, the maximum amount by which the
        output can be shorter than the input. Should be non-negative i.e. a
        value of 1.0 means that the output can't be more than a second shorter
        than the input.
    :param max_over_seconds: If not None, the maximum amount by which the
        output can be longer than the input. Should be non-negative i.e. a
        value of 1.0 means that the output can't be more than a second longer
        than the input.
    """
    x, x_info = _wav_to_np(input_path, info=True)
    y, y_info = _wav_to_np(output_path, info=True)

    length_input = len(x) / x_info.rate
    length_output = len(y) / y_info.rate
    delta_seconds = length_output - length_input

    passed = True
    if max_under_seconds is not None and delta_seconds < -max_under_seconds:
        passed = False
    if max_over_seconds is not None and delta_seconds > max_over_seconds:
        passed = False

    return _LengthValidation(passed=passed, delta_seconds=delta_seconds)


def validate_data(
    input_path: _Path,
    output_path: _Path,
    user_latency: _Optional[int],
    num_output_samples_per_datum: int = _NY_DEFAULT,
):
    """
    Just do the checks to make sure that the data are ok.

    * Version identification
    * Latency calibration
    * Other checks
    """
    print("Validating data...")
    passed = True  # Until proven otherwise
    passed_critical = True  # These can't be ignored

    sample_rate_validation = _check_audio_sample_rates(input_path, output_path)
    passed = passed and sample_rate_validation.passed
    passed_critical = passed_critical and sample_rate_validation.passed

    length_validation = _check_audio_lengths(input_path, output_path)
    passed = passed and length_validation.passed
    passed_critical = passed_critical and length_validation.passed

    # Data version ID
    input_version, strong_match = _detect_input_version(input_path)

    # Latency analysis
    latency_analysis = _analyze_latency(
        user_latency, input_version, input_path, output_path, silent=True
    )
    if latency_analysis.manual is None and any(
        val for val in latency_analysis.calibration.warnings.model_dump().values()
    ):
        passed = False
    final_latency = _get_final_latency(latency_analysis)

    # Other data checks based on input file version
    data_checks = _check_data(
        input_path,
        output_path,
        input_version,
        latency_analysis.calibration.recommended,
        silent=True,
    )
    passed = passed and data_checks.passed

    # Finally, try to make the PyTorch Dataset objects and note any failures:
    data_config = _get_data_config(
        input_version=input_version,
        input_path=input_path,
        output_path=output_path,
        ny=num_output_samples_per_datum,
        latency=final_latency,
    )
    # HACK this should depend on the model that's going to be used, but I think it will
    # be unlikely to make a difference. Still, would be nice to fix.
    data_config["common"]["nx"] = 4096

    pytorch_data_split_validation_dict: _Dict[str, _PyTorchDataSplitValidation] = {}
    for split in _Split:
        try:
            ds = _init_dataset(data_config, split)
            ds.teardown()
            pytorch_data_split_validation_dict[split.value] = (
                _PyTorchDataSplitValidation(passed=True, msg=None)
            )
        except _DataError as e:
            pytorch_data_split_validation_dict[split.value] = (
                _PyTorchDataSplitValidation(passed=False, msg=str(e))
            )
    pytorch_data_validation = _PyTorchDataValidation(
        passed=all(v.passed for v in pytorch_data_split_validation_dict.values()),
        **pytorch_data_split_validation_dict,
    )
    passed = passed and pytorch_data_validation.passed
    passed_critical = passed_critical and pytorch_data_validation.passed

    return DataValidationOutput(
        passed=passed,
        passed_critical=passed_critical,
        sample_rate=sample_rate_validation,
        length=length_validation,
        input_version=str(input_version),
        latency=latency_analysis,
        checks=data_checks,
        pytorch=pytorch_data_validation,
    )
