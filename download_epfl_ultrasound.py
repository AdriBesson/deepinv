#!/usr/bin/env python3
"""Download individual .npz acquisitions from the EPFL ultrafast ultrasound dataset.

https://www.epfl.ch/labs/lts5/research/us/epfl-ultrafast-ultrasound-datasets/

The dataset is published only as very large zip archives (60-330 GB each, one per
volunteer, volunteer 1 split by region, plus one for the in-vitro CIRS 054GS
phantom). This script never downloads a whole
archive: it reads the zip central directory with HTTP range requests, then
fetches and inflates only the selected members, one at a time.

Files are stored as
    <out>/epfl_ultrafast_ultrasound/volunteer_<NNN>/<region>/invivo_<XXXXX>.npz
    <out>/epfl_ultrafast_ultrasound/invitro_CIRS054GS/<region>/invitro_<XXXXX>.npz
where the in-vitro regions are the phantom sections "054GS" and "FullSpeckle".

Examples
    python download_epfl_ultrasound.py --list
    python download_epfl_ultrasound.py --volunteer 5 --region abdomen --limit 10
    python download_epfl_ultrasound.py --volunteer 1 --region carotid shoulders
    python download_epfl_ultrasound.py --volunteer 2 --ids 6520 6521
    python download_epfl_ultrasound.py --volunteer 2 5 7 --region abdomen --limit 100
    python download_epfl_ultrasound.py --invitro --ids 2162
    python download_epfl_ultrasound.py --invitro --region FullSpeckle --limit 10
"""

import argparse
import collections
import io
import os
import struct
import sys
import time
import urllib.request
import zipfile
import zlib

BASE_URL = "https://datasets.epfl.ch/epfl_ultrafast_ultrasound"
DEFAULT_OUT = "/media/bobby/SSK SSD/"
# Beamformed cache written by examples/self-supervised-learning/precompute_n2i_ultrasound.py
# (CACHE_ROOT there): <cache>/<dataset_dir>/<region>/<stem>.npy
DEFAULT_CACHE = "/data/n2i_das_cache"

# Dataset -> archives holding its data: volunteer numbers for the in-vivo data,
# INVITRO for the phantom.
INVITRO = "invitro"
ARCHIVES = {
    1: [
        "volunteer_1_abdomen",
        "volunteer_1_arm_back",
        "volunteer_1_breast_carotid_shoulders",
        "volunteer_1_legs",
    ],
    **{v: [f"volunteer_{v:03d}"] for v in range(2, 10)},
    INVITRO: ["invitro_CIRS054GS"],
}
VOLUNTEERS = [k for k in ARCHIVES if k != INVITRO]


def dataset_dir(dataset):
    """Output folder of a dataset: volunteer_<NNN> or invitro_CIRS054GS."""
    return ARCHIVES[INVITRO][0] if dataset == INVITRO else f"volunteer_{dataset:03d}"


def dataset_name(dataset):
    return "in-vitro phantom" if dataset == INVITRO else f"volunteer {dataset}"


def file_name(dataset, acq):
    return f"{'invitro' if dataset == INVITRO else 'invivo'}_{acq:05d}.npz"

CHUNK = 1 << 20
RETRIES = 5


class HttpFile(io.RawIOBase):
    """Minimal seekable read-only file over HTTP range requests (for zipfile)."""

    def __init__(self, url):
        self.url = url
        self.pos = 0
        # HEAD returns Content-Length: 0 on this server; use a 1-byte range instead.
        r = _open(url, 0, 0)
        self.size = int(r.headers["Content-Range"].split("/")[1])
        r.close()

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        with _open(self.url, self.pos, end) as r:
            data = r.read()
        self.pos += len(data)
        return data


def _open(url, start, end=None):
    rng = f"bytes={start}-" if end is None else f"bytes={start}-{end}"
    req = urllib.request.Request(url, headers={"Range": rng, "User-Agent": "epfl-us-dl/1.0"})
    return urllib.request.urlopen(req, timeout=60)


def list_members(archive):
    """Return {(region, acquisition_id): ZipInfo} for all .npz files in an archive."""
    url = f"{BASE_URL}/{archive}.zip"
    zf = zipfile.ZipFile(HttpFile(url))
    members = {}
    for info in zf.infolist():
        if info.is_dir() or not info.filename.endswith(".npz"):
            continue
        parts = info.filename.split("/")
        # volunteer_1_*.zip: "<region>/invivo_X.npz"; others: "volunteer_NNN/<region>/invivo_X.npz";
        # invitro_CIRS054GS.zip: "invitro_CIRS054GS/<region>/invitro_X.npz"
        region, fname = parts[-2], parts[-1]
        acq = int(fname[fname.index("_") + 1:-len(".npz")])
        members[(region, acq)] = info
    return url, members


def index_dataset(dataset):
    """Return list of (url, region, acq_id, ZipInfo) for a dataset, sorted by acq_id."""
    entries = []
    for archive in ARCHIVES[dataset]:
        url, members = list_members(archive)
        for (region, acq), info in members.items():
            entries.append((url, region, acq, info))
    entries.sort(key=lambda e: e[2])
    return entries


def download_member(url, info, dest, quiet=False):
    """Fetch one zip member via a single ranged GET, inflate it, verify CRC, write to dest."""
    tmp = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(1, RETRIES + 1):
        try:
            _download_member_once(url, info, tmp, quiet)
            os.replace(tmp, dest)
            return
        except Exception as e:  # noqa: BLE001
            if attempt == RETRIES:
                raise
            print(f"\n  retry {attempt}/{RETRIES - 1} after error: {e}", file=sys.stderr)
            time.sleep(2 * attempt)


def _download_member_once(url, info, tmp, quiet):
    # Local file header: sig(4) ver(2) flag(2) method(2) time(2) date(2) crc(4)
    #                    csize(4) usize(4) nlen(2) elen(2) + name + extra + data
    with _open(url, info.header_offset) as r:
        hdr = r.read(30)
        if hdr[:4] != b"PK\x03\x04":
            raise IOError("bad local file header")
        nlen, elen = struct.unpack("<HH", hdr[26:30])
        r.read(nlen + elen)

        if info.compress_type == zipfile.ZIP_DEFLATED:
            inflate = zlib.decompressobj(-15)
        elif info.compress_type == zipfile.ZIP_STORED:
            inflate = None
        else:
            raise IOError(f"unsupported compression {info.compress_type}")

        remaining = info.compress_size
        crc = 0
        written = 0
        t0 = time.time()
        with open(tmp, "wb") as f:
            while remaining > 0:
                buf = r.read(min(CHUNK, remaining))
                if not buf:
                    raise IOError("connection closed early")
                remaining -= len(buf)
                out = inflate.decompress(buf) if inflate else buf
                f.write(out)
                crc = zlib.crc32(out, crc)
                written += len(out)
                if not quiet:
                    done = info.compress_size - remaining
                    rate = done / max(time.time() - t0, 1e-6) / 1e6
                    print(f"\r  {done / 1e6:8.1f}/{info.compress_size / 1e6:.1f} MB  {rate:6.1f} MB/s", end="")
            if inflate:
                out = inflate.flush()
                f.write(out)
                crc = zlib.crc32(out, crc)
                written += len(out)
        if not quiet:
            print()
    if written != info.file_size or (crc & 0xFFFFFFFF) != info.CRC:
        raise IOError("size/CRC mismatch")


def cmd_list(args):
    for d in selected_datasets(args) or list(ARCHIVES):
        counts = collections.Counter()
        ids = collections.defaultdict(list)
        sizes = collections.Counter()
        for _, region, acq, info in index_dataset(d):
            counts[region] += 1
            ids[region].append(acq)
            sizes[region] += info.file_size
        print(f"{dataset_name(d)}:")
        for region in sorted(counts):
            first, last = (file_name(d, f(ids[region]))[:-4] for f in (min, max))
            print(f"  {region:<12} {counts[region]:5d} files  "
                  f"{first}..{last}  "
                  f"{sizes[region] / 1e9:6.1f} GB")


def selected_datasets(args):
    """Datasets named by --volunteer and --invitro, in that order."""
    return (args.volunteer or []) + ([INVITRO] if args.invitro else [])


def select_dataset(dataset, args):
    """Entries of one dataset matching --region/--ids/--limit, as (dataset, url, region, acq, info)."""
    entries = index_dataset(dataset)
    regions = set(args.region or [])
    known = {e[1] for e in entries}
    unknown = regions - known
    if unknown:
        sys.exit(f"unknown region(s) {sorted(unknown)} for {dataset_name(dataset)}; "
                 f"available: {sorted(known)}")
    ids = set(args.ids or [])
    selected = [(dataset, *e) for e in entries
                if (not regions or e[1] in regions) and (not ids or e[2] in ids)]
    if args.limit:
        selected = selected[:args.limit]
    return selected


def cmd_download(args):
    # --limit applies per dataset; --ids is checked against the union of them.
    selected = [e for d in selected_datasets(args) for e in select_dataset(d, args)]
    if args.ids:
        missing = set(args.ids) - {e[3] for e in selected}
        if missing:
            print(f"warning: ids not found for this selection: {sorted(missing)}", file=sys.stderr)
    if not selected:
        sys.exit("nothing selected")

    root = os.path.join(args.out, "epfl_ultrafast_ultrasound")

    def skip_reason(v, region, acq, info):
        dest = os.path.join(root, dataset_dir(v), region, file_name(v, acq))
        if os.path.exists(dest) and os.path.getsize(dest) == info.file_size:
            return "exists"
        cached = os.path.join(args.cache, dataset_dir(v), region, file_name(v, acq)[:-len(".npz")] + ".npy")
        if args.cache and os.path.exists(cached):
            return f"already preprocessed in {args.cache}"
        return None

    skipped = collections.Counter()
    todo = []
    for e in selected:
        reason = skip_reason(e[0], e[2], e[3], e[4])
        if reason:
            skipped[reason] += 1
        else:
            todo.append(e)
    for reason, n in skipped.items():
        print(f"skipping {n} files ({reason})")
    selected = todo
    if not selected:
        print("nothing left to download")
        return

    total = sum(e[4].file_size for e in selected)
    print(f"{len(selected)} files, {total / 1e9:.1f} GB -> {root}")
    if args.dry_run:
        for v, _, region, acq, info in selected:
            print(f"  {dataset_dir(v)}/{region}/{file_name(v, acq)}  {info.file_size / 1e6:.1f} MB")
        return

    for i, (v, url, region, acq, info) in enumerate(selected, 1):
        dest = os.path.join(root, dataset_dir(v), region, file_name(v, acq))
        rel = os.path.relpath(dest, args.out)
        print(f"[{i}/{len(selected)}] {rel}")
        download_member(url, info, dest, quiet=args.quiet)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="list volunteers/regions/acquisition ids and exit")
    p.add_argument("--volunteer", "-v", type=int, nargs="+", choices=VOLUNTEERS, metavar="V",
                   help="volunteer number(s) (1-9)")
    p.add_argument("--invitro", action="store_true",
                   help="the in-vitro CIRS 054GS phantom dataset (regions 054GS, FullSpeckle)")
    p.add_argument("--region", "-r", nargs="+", metavar="REGION",
                   help="body region(s) as named in the archive (see --list); default: all")
    p.add_argument("--ids", nargs="+", type=int, metavar="N",
                   help="only these acquisition numbers (e.g. 14579 for invivo_14579.npz)")
    p.add_argument("--limit", "-n", type=int, help="download at most N files per volunteer / the phantom")
    p.add_argument("--out", "-o", default=DEFAULT_OUT, help=f"dataset root (default: {DEFAULT_OUT})")
    p.add_argument("--cache", default=DEFAULT_CACHE,
                   help="cache of precompute_n2i_ultrasound.py; acquisitions already preprocessed "
                        f"there are not downloaded again ('' to disable) (default: {DEFAULT_CACHE})")
    p.add_argument("--dry-run", action="store_true", help="show what would be downloaded")
    p.add_argument("--quiet", "-q", action="store_true", help="no per-file progress")
    args = p.parse_args()

    if args.list:
        cmd_list(args)
    elif not selected_datasets(args):
        p.error("--volunteer and/or --invitro is required (or use --list)")
    else:
        cmd_download(args)


if __name__ == "__main__":
    main()
