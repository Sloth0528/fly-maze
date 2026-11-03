"""Download and deterministically extract measured FlyWire FAFB v783 connections."""

from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np

SOURCE_URL = "https://storage.googleapis.com/flywire-data/codex/data/fafb/783/connections.csv.gz"
SOURCE_SHA256 = "d49dd692e59e153aa3c83f5257bfc0eff51247b86d7bb183386c6d1622c70fc9"
REGIONS = ("EB", "FB", "PB", "NO")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"pre_root_id", "post_root_id", "neuropil", "syn_count"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"FlyWire CSV must have columns: {sorted(required)}")
        for row in reader:
            if not row or row["pre_root_id"] is None:
                continue
            count = int(row["syn_count"])
            if count <= 0:
                raise ValueError("syn_count must be positive")
            yield row["pre_root_id"], row["post_root_id"], row["neuropil"], count


def prepare(output="data/flywire_cx_512.npz", *, cache="data/raw/connections.csv.gz",
            nodes=512, regions=REGIONS, source=None):
    """Top synapse-strength neurons in specified neuropils; induced directed subgraph.

    Counts are structural, nonnegative weights, NOT physiological synaptic efficacy.
    No synthetic edges, inferred transmitter signs, or biological input/output mapping.
    """
    output = Path(output)
    manifest_path = output.with_suffix(".json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("connectome output exists; choose a new output name")
    if nodes < 2 or not regions:
        raise ValueError("need at least two nodes and at least one neuropil")
    raw = Path(source or cache)
    if source and not raw.is_file():
        raise FileNotFoundError(raw)
    if not raw.exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        temporary = raw.with_suffix(raw.suffix+".partial")
        print("Downloading FlyWire FAFB v783 (~50 MB)...", flush=True)
        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "FlyMaze/2.0 research-teaching"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024*1024):
                handle.write(chunk)
        temporary.replace(raw)
    source_hash = sha256(raw)
    if source is None and source_hash != SOURCE_SHA256:
        raise ValueError("official snapshot SHA256 changed; refusing an unverified download/cache")
    print("Selecting neurons from measured connections...", flush=True)
    region_set = set(regions)
    strength = Counter()
    selected_rows = 0
    for pre, post, region, count in rows(raw):
        if region in region_set:
            strength[pre] += count
            strength[post] += count
            selected_rows += 1
    if len(strength) < nodes:
        raise ValueError(f"only {len(strength)} neurons available in selected regions, requested {nodes}")
    # Deterministic tie-breaking by numeric root id, then fixed numeric node ordering.
    ids = sorted(sorted(strength, key=lambda root: (-strength[root], int(root)))[:nodes], key=int)
    lookup = {root: index for index, root in enumerate(ids)}
    counts = np.zeros((nodes, nodes), dtype=np.int64)  # [post, pre]
    for pre, post, region, count in rows(raw):
        if region in region_set and pre in lookup and post in lookup:
            counts[lookup[post], lookup[pre]] += count
    if not counts.any():
        raise ValueError("selected subgraph has no measured edges")
    metadata = {
        "format_version": 1, "dataset": "FlyWire FAFB v783 adult female",
        "source_url": SOURCE_URL if source is None else None,
        "source_file": raw.name, "source_sha256": source_hash,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "regions": list(regions), "selection": "top total in+out regional synapse count; numeric root-id tie break",
        "matrix_axes": "counts[postsynaptic, presynaptic]",
        "nodes": nodes, "edges": int(np.count_nonzero(counts)), "synapses": int(counts.sum()),
        "regional_source_rows": selected_rows,
        "license": "CC BY-NC 4.0 (FlyWire public-release guidelines)",
        "license_url": "https://flywire.ai/guidelines",
        "citations": ["https://doi.org/10.1038/s41586-024-07558-y",
                      "https://doi.org/10.1038/s41586-024-07686-5"],
        "limitations": ["Subset of adult brain and selected neuropils, not a full brain or nervous system",
                       "Degree/strength selection bias; omitted cells and connections",
                       "Counts are unsigned structural weights, not synaptic efficacy or neurotransmitter effects",
                       "Sensory encoder and motor readout are learned engineering interfaces",
                       "Rate-network dynamics and reinforcement learning are not biological reconstruction"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        np.savez_compressed(handle, counts=counts, root_ids=np.asarray(ids, dtype="U32"),
                            metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)))
    metadata["artifact_sha256"] = sha256(output)
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(f"Prepared {nodes} measured neurons, {metadata['edges']} directed edges, "
          f"{metadata['synapses']} synapses: {output}", flush=True)
    return metadata


def load_graph(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as artifact:
        counts = artifact["counts"]
        ids = artifact["root_ids"].tolist()
        metadata = json.loads(str(artifact["metadata"].item()))
    if (counts.ndim != 2 or counts.shape != (len(ids), len(ids)) or len(set(ids)) != len(ids)
            or len(ids) < 2 or not np.isfinite(counts).all() or (counts < 0).any()
            or not np.any(counts) or not np.equal(counts, np.floor(counts)).all()):
        raise ValueError("invalid measured connectome artifact")
    if metadata.get("format_version") != 1 or metadata.get("matrix_axes") != "counts[postsynaptic, presynaptic]":
        raise ValueError("unsupported connectome format")
    manifest_path = path.with_suffix(".json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("artifact_sha256") != sha256(path):
            raise ValueError("connectome artifact SHA256 does not match manifest")
    return counts, ids, metadata
