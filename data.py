from __future__ import annotations

import mmap
import struct
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

MAGIC = bytes.fromhex("3fa1b496f0405edd")
KIND = {0x00010801: "u", 0x00014000: "i", 0x00012000: "f",
        0x00012002: "f", 0x00014002: "f", 0x00014001: "i"}
DEFAULT_DGL = Path(__file__).resolve().parents[1] / "datasets/FDCompCN_dataset/comp.dgl"

SPLIT_COUNTS = {"train": (2379, 279), "val": (953, 111), "test": (1426, 169)}


def _scan(buf) -> list[dict]:
    u64 = lambda o: struct.unpack_from("<Q", buf, o)[0]
    u32 = lambda o: struct.unpack_from("<I", buf, o)[0]
    recs, pos = [], 0
    while True:
        m = buf.find(MAGIC, pos)
        if m < 0:
            break
        name = ""
        for L in range(1, 25):
            s = m - L
            if s - 8 >= 0 and u64(s - 8) == L and all(32 <= c < 127 for c in buf[s:m]):
                name = buf[s:m].decode()
                break
        o = m + 8
        nd, dt = u32(o + 16), u32(o + 20)
        dims = [u64(o + 24 + 8 * i) for i in range(nd)]
        p = o + 24 + 8 * nd
        nb = u64(p)
        p += 8
        recs.append({"name": name, "ndim": nd, "dt": dt, "dims": dims,
                     "nbytes": nb, "offset": p})
        pos = m + 8
    return recs


def _read(path: Path, rec: dict) -> np.ndarray:
    numel = int(np.prod(rec["dims"])) if rec["dims"] else 0
    itemsize = rec["nbytes"] // max(numel, 1)
    kind = KIND.get(rec["dt"], "i")
    dt = {"f": {4: "<f4", 8: "<f8"}, "i": {1: "<i1", 4: "<i4", 8: "<i8"},
          "u": {1: "<u1"}}[kind].get(itemsize, "<i8")
    a = np.fromfile(path, dtype=dt, count=numel, offset=rec["offset"])
    return a.reshape(rec["dims"]) if rec["ndim"] > 1 else a


def load_fdcompcn(path: str | Path = DEFAULT_DGL) -> dict:
    path = Path(path)
    with open(path, "rb") as f:
        buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        recs = _scan(buf)
        named = {}
        for r in recs:
            if r["name"] and r["name"] not in named:
                named[r["name"]] = r
        x = _read(path, named["feature"]).astype(np.float32)
        y = _read(path, named["label"]).ravel().astype(np.int64)
        chunks = [_read(path, r) for r in recs
                  if not r["name"] and r["dims"] and r["dims"][0] > 100]
        buf.close()
    rels = {}
    for name, (a, b) in zip(("cic", "csc", "cpc"), [(chunks[2], chunks[3]),
                                                    (chunks[4], chunks[5]),
                                                    (chunks[6], chunks[7])]):
        rels[name] = np.stack([a, b], 1).astype(np.int64)
    return {"x": x, "y": y, "relations": rels}


def relation_adjacency(n: int, e: np.ndarray) -> sp.csr_matrix:

    r = np.concatenate([e[:, 0], e[:, 1]])
    c = np.concatenate([e[:, 1], e[:, 0]])
    a = sp.coo_matrix((np.ones(r.size, np.float32), (r, c)), shape=(n, n)).tocsr()
    a.data[:] = 1.0
    a.setdiag(0)
    a.eliminate_zeros()
    return a


def build_hypergraph(n: int, relations: dict) -> dict:

    members = []
    for r, e in relations.items():
        a = relation_adjacency(n, e)
        ncomp, lab = connected_components(a, directed=False)
        for c in range(ncomp):
            mem = np.flatnonzero(lab == c)
            if mem.size >= 2:                     
                members.append(mem)
    node = np.concatenate(members)
    edge = np.concatenate([np.full(m.size, i) for i, m in enumerate(members)])
    m = len(members)
    dv = np.bincount(node, minlength=n).astype(np.float32)
    de = np.bincount(edge, minlength=m).astype(np.float32)
    import torch
    return {"node": torch_long(node), "edge": torch_long(edge), "n_edges": m,
            "dv": dv, "de": de,
            "dv_inv_sqrt": torch.from_numpy(np.power(np.maximum(dv, 1), -0.5).astype(np.float32)),
            "de_inv": torch.from_numpy((1.0 / np.maximum(de, 1)).astype(np.float32))}


def torch_long(a: np.ndarray):
    import torch
    return torch.from_numpy(a.astype(np.int64))

def make_split(y: np.ndarray, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    out = {k: [] for k in SPLIT_COUNTS}
    for cls, j in ((0, 0), (1, 1)):
        idx = rng.permutation(np.flatnonzero(y == cls))
        p = 0
        for part, cnt in SPLIT_COUNTS.items():
            n = cnt[j]
            out[part].append(idx[p:p + n])
            p += n
    return {k: np.sort(np.concatenate(v)).astype(np.int64) for k, v in out.items()}


def standardize(x: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    mu = x[train_idx].mean(0, keepdims=True)
    sd = x[train_idx].std(0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return ((x - mu) / sd).astype(np.float32)
