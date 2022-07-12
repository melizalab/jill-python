# -*- coding: utf-8 -*-
"""plays very long audio files through JACK

- stimuli are loaded from arf files
- all datasets must be 1-D with sampling rates that match the jack daemon
- buffers are large, because latency is not considered to be an issue

Requires the python packages h5py and JACK-Client.

"""
import os
import queue
import threading
import logging
import numpy as np
import h5py as h5
import jack

log = logging.getLogger("jbigstim")  # root logger
event = threading.Event()


def setup_log(log, debug=False):
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s : %(message)s")
    loglevel = logging.DEBUG if debug else logging.INFO
    log.setLevel(loglevel)
    ch.setLevel(loglevel)
    ch.setFormatter(formatter)
    log.addHandler(ch)


def split_hdf5_path(path):
    """given path='/path/to/file/entry/dset', splits into '/path/to/file' and 'entry/dset'"""
    path = os.path.abspath(path.rstrip("/"))
    if os.access(path, os.F_OK):
        raise ValueError("dataset path must point to a dataset within an hdf5 file")
    head, tail = os.path.split(path)
    while not os.access(head, os.F_OK):
        head, x = os.path.split(head)
        tail = os.path.join(x, tail)
    return (head, tail)


def dset_amplitude(dset):
    n = dset.chunks[0]
    sse = 0
    peak = 0
    for i in range(0, dset.size, n):
        signal = dset[i : i + n]
        sse += np.sum(signal * signal)
        peak = max(peak, np.abs(signal).max())
    rms = np.sqrt(sse / dset.size)
    return {"rms": 20 * np.log10(rms) + 3.0103, "peak": 20 * np.log10(peak) + 3.0103}


def check_datasets(datasets, sampling_rate, check_amplitude=False):
    """check all datasets to ensure that they exist and have consistent sampling rates"""
    log.info("- checking datasets:")
    for dset_path in datasets:
        path, dset_name = split_hdf5_path(dset_path)
        with h5.File(path, "r") as fp:
            dset = fp[dset_name]
            dset_rate = dset.attrs["sampling_rate"]
            if dset.ndim != 1:
                raise ValueError(f"{dset_path} has more than one dimension")
            if dset_rate != sampling_rate:
                raise ValueError(
                    f"{dset_path} has the wrong sampling rate ({dset_rate}, needs to be {sampling_rate})"
                )
            log.info(
                "  ✓ %s is %d samples (%.2f s)",
                dset_path,
                dset.size,
                dset.size / dset.attrs["sampling_rate"],
            )
            if check_amplitude:
                amplitude = dset_amplitude(dset)
                log.info("    - RMS=%(rms).2f dBFS; peak=%(peak).2f dBFS" % amplitude)


def get_sync_start(dset, at_time):
    """Determines the sample offset in dset corresponding to the time of day `at_time`

    If `at_time` is before the time of day when the recording started, then the
    offset is calculated for `at_time` in the subsequent day.
    """
    from datetime import datetime, timedelta

    dset_timestamp = dset.parent.attrs["timestamp"]
    sampling_rate = dset.attrs["sampling_rate"]
    dset_start = datetime.fromtimestamp(dset_timestamp[0]) + timedelta(
        microseconds=int(dset_timestamp[1])
    )
    delta = at_time - dset_start
    sample = int(
        delta.seconds * sampling_rate + delta.microseconds * sampling_rate / 1000
    )
    if sample > dset.size:
        raise ValueError(
            f"current time corresponds to sample {sample} but dataset is only {dset.size} long"
        )
    return sample


def iter_datasets(
    datasets, block_size, gap_samples, loop=False, scale=1.0, clock_sync=False
):
    """Iterate through the datasets, yielding blocks of samples

    block_size: size of the block (should match audio playback period size)
    gap_samples: the number of silent samples to play between each dataset
    loop: if True, loop through the datasets endlessly
    scale: rescale the output by this factor

    clock_sync: if True, starts at the sample corresponding to the current time
    of day in the recording (i.e. if the recording started at midnight and the
    current time is 6:00 AM, playback would start 6 * 60 * 60 seconds into the
    recording.

    """
    from datetime import datetime, timedelta
    from itertools import cycle

    zeros = np.zeros(block_size, dtype="float32")
    if loop:
        datasets = cycle(datasets)
    for dset_path in datasets:
        path, dset_name = split_hdf5_path(dset_path)
        with h5.File(path, "r") as fp:
            dset = fp[dset_name]
            start_sample = 0
            if clock_sync:
                start_sample = get_sync_start(dset, datetime.now())
            log.info(
                "- started reading from %s at sample %d (%s)",
                dset_path,
                start_sample,
                timedelta(seconds=start_sample / dset.attrs["sampling_rate"]),
            )

            for i in range(start_sample, dset.size, block_size):
                n = dset.size - i
                if n < block_size:
                    data = np.zeros(block_size, dtype="float32")
                    data[:n] = dset[i : i + n] * scale
                else:
                    data = dset[i : i + block_size] * scale
                if data.max() > 1.0 or data.min() < -1.0:
                    log.warn(" - warning: stimulus will clip around sample %d", i)
                yield data.astype("float32")
            log.info("- finished reading from %s", dset_path)
            for i in range(0, gap_samples, block_size):
                yield zeros


def xrun(delay):
    log.error("An xrun occurred, increase JACK's period size?")


def shutdown(status, reason):
    log.error("JACK shutdown!")
    log.error("status: %s", status)
    log.error("reason: %s", reason)
    event.set()


def main(argv=None):
    import argparse
    from . import __version__

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-v", "--version", action="version", version="%(prog)s " + __version__
    )
    p.add_argument("--debug", help="show verbose log messages", action="store_true")
    p.add_argument(
        "--gap",
        "-g",
        type=float,
        default=0.0,
        help="gap between stimuli (%(default)f s)",
    )
    p.add_argument("--loop", "-l", action="store_true", help="loop the stimulus set")
    p.add_argument(
        "--buffer",
        "-b",
        type=int,
        default=100,
        help="number of periods used for buffering (default: %(default)d)",
    )
    p.add_argument(
        "--check-amplitude", action="store_true", help="check amplitude of stimuli"
    )
    p.add_argument(
        "--scale",
        "-S",
        default=0.0,
        type=float,
        help="scale output (default %(default).1f dB)",
    )
    p.add_argument(
        "--sync-start",
        action="store_true",
        help="start playback at sample corresponding to current time",
    )
    p.add_argument("--output", "-o", help="create connection to output audio port")
    p.add_argument("--name", "-n", default="jbigstim", help="set jack client name")
    p.add_argument("--server", "-s", help="connect to specific jack daemon")
    p.add_argument(
        "datasets",
        nargs="+",
        help="hdf5 datasets to play (e.g. file.arf/entry_0001/pcm_001)",
    )

    args = p.parse_args()
    if args.buffer < 1:
        p.error("buffer must be at least 1 period")
    setup_log(log, args.debug)

    q = queue.Queue(maxsize=args.buffer)
    log.info("- this is jbigstim, version %s", __version__)

    log.debug("- connecting to JACK server %s", args.server or "")
    client = jack.Client(args.name, servername=args.server)
    if client.status.server_started:
        log.info("- JACK server started")
    if client.status.name_not_unique:
        log.info("  - unique name %s assigned to client", client.name)

    jack_sampling_rate = client.samplerate
    jack_period_size = client.blocksize
    q_timeout = jack_period_size * args.buffer / jack_sampling_rate
    log.info("  - sample rate: %.1f Hz", jack_sampling_rate)
    log.info(
        "  - period size: %d samples (%.1f ms)",
        jack_period_size,
        jack_period_size / jack_sampling_rate * 1000,
    )
    log.info("  - buffer size: %d periods (%.1f ms)", args.buffer, q_timeout * 1000)
    client.set_xrun_callback(xrun)
    client.set_shutdown_callback(shutdown)
    client.outports.register("out")

    def stop_callback(msg=""):
        if msg:
            log.error(msg)
        for port in client.outports:
            port.get_array().fill(0)
        event.set()
        raise jack.CallbackExit

    @client.set_process_callback
    def process(frames):
        if frames != jack_period_size:
            stop_callback("period size changed; aborting")
        try:
            data = q.get_nowait()
        except queue.Empty:
            stop_callback("buffer underrun")
        if data is None:
            stop_callback()
        port = client.outports[0]
        port.get_array()[:] = data

    check_datasets(args.datasets, jack_sampling_rate, args.check_amplitude)

    scale_x = 10 ** (args.scale / 20)
    log.info("- stimulus amplitude will be scaled by %.3fx", scale_x)
    block_g = iter_datasets(
        args.datasets,
        jack_period_size,
        int(args.gap * jack_sampling_rate),
        loop=args.loop,
        scale=scale_x,
        clock_sync=args.sync_start,
    )
    log.debug("- prefilling queue")
    for _, data in zip(range(args.buffer), block_g):
        q.put_nowait(data)

    try:
        with client:
            if args.output:
                log.info("- connecting output port to %s", args.output)
                client.outports[0].connect(args.output)
            for data in block_g:
                q.put(data, timeout=q_timeout)
            q.put(None, timeout=q_timeout)
            event.wait()
    except KeyboardInterrupt:
        p.exit("\nInterrupted by user")
    except (queue.Full):
        log.error("buffer overrun!")
        p.exit(1)
