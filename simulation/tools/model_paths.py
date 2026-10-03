"""Model loading. The XML builders baked absolute asset paths into the models
(C:/Users/Michael/exo-sim/...), so we rewrite that prefix to wherever the repo
is at load time. File on disk isn't touched.

Compiled models get cached in models/.mjb_cache/ because compiling the XML
uses way more memory than loading the .mjb. Cache key = rewritten xml + size
and mtime of every asset + mujoco version. Writes go to a temp file and get
renamed so workers starting at the same time don't read half a file.
Setting EXO_SIM_NO_MJB_CACHE to anything non-empty turns the cache off.
"""
from __future__ import annotations

import hashlib
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

# The prefix the generators baked in.  Both separators appear in the files.
_BAKED_PREFIXES = ("C:/Users/Michael/exo-sim/", "C:\\Users\\Michael\\exo-sim\\")

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "models" / ".mjb_cache"
_FILE_ATTR = re.compile(r'(?:file|meshdir|texturedir)="([^"]+)"')


def configure_memory(xml: str, arena_memory_mb: int = 0,
                     discard_visual: bool = False) -> str:
    # 0 = leave the XML alone. The old njmax=5000 gives every MjData a huge
    # arena, so swap njmax/nstack for `memory` (nconmax stays).
    if (isinstance(arena_memory_mb, bool)
            or int(arena_memory_mb) != arena_memory_mb or arena_memory_mb < 0):
        raise ValueError("arena_memory_mb must be a nonnegative integer (0 keeps XML settings)")
    if not arena_memory_mb and not discard_visual:
        return xml
    tree = ET.fromstring(xml)
    if arena_memory_mb:
        size = tree.find("size")
        if size is None:
            size = ET.SubElement(tree, "size")
        for legacy in ("njmax", "nstack"):
            size.attrib.pop(legacy, None)
        size.set("memory", str(int(arena_memory_mb) * 1024 * 1024))
    if discard_visual:
        # Rangefinders raycast against visual geoms, so stripping changes them.
        if tree.find(".//rangefinder") is not None:
            raise ValueError("discard_visual is incompatible with rangefinder sensors")
        compiler = tree.find("compiler")
        if compiler is None:
            compiler = ET.SubElement(tree, "compiler")
        compiler.set("discardvisual", "true")
    return ET.tostring(tree, encoding="unicode")


def check_arena_capacity(data: mujoco.MjData) -> None:
    # don't silently train on states where mujoco dropped contacts
    if (data.warning[mujoco.mjtWarning.mjWARN_CONTACTFULL].number
            or data.warning[mujoco.mjtWarning.mjWARN_CNSTRFULL].number):
        raise RuntimeError(
            "MuJoCo contact/constraint allocation exhausted; increase "
            "arena_memory_mb (--arena-memory-mb), or use 0 for the original "
            "XML allocation, and check the model's nconmax contact limit")


def rewrite_asset_paths(xml: str, root: Path | None = None) -> str:
    # forward slashes work on windows too
    prefix = (root or REPO_ROOT).as_posix().rstrip("/") + "/"
    for baked in _BAKED_PREFIXES:
        xml = xml.replace(baked, prefix)
    return xml


def _cache_key(path: Path, xml: str) -> str:
    h = hashlib.sha1()
    h.update(mujoco.__version__.encode())
    h.update(xml.encode("utf-8"))
    meshdir = ""
    for m in _FILE_ATTR.finditer(xml):
        ref = m.group(1)
        if m.group(0).startswith("meshdir") or m.group(0).startswith("texturedir"):
            meshdir = ref
            continue
        cand = Path(ref)
        if not cand.is_absolute():
            for base in (Path(meshdir), path.parent):
                if (base / ref).is_file():
                    cand = base / ref
                    break
        try:
            st = cand.stat()
            h.update(f"{cand}|{st.st_size}|{st.st_mtime_ns}".encode())
        except OSError:
            h.update(f"{cand}|missing".encode())
    return h.hexdigest()[:16]


def _compile(path: Path, xml: str, rewritten: str) -> mujoco.MjModel:
    if rewritten == xml:
        return mujoco.MjModel.from_xml_path(str(path))
    # Asset paths are absolute after the rewrite and there are no <include>s,
    # so loading from a string loses nothing.
    return mujoco.MjModel.from_xml_string(rewritten)


def load_model(path: str | Path, root: Path | None = None,
               binary_cache: bool | None = None, *, arena_memory_mb: int = 0,
               discard_visual: bool = False) -> mujoco.MjModel:
    """Load an MJCF, fixing up the baked asset paths. The scripts all use a
    16 MB arena; training and scoring also drop visual geoms, the viewer
    doesn't."""
    path = Path(path)
    xml = path.read_text(encoding="utf-8")
    rewritten = rewrite_asset_paths(xml, root)
    rewritten = configure_memory(rewritten, arena_memory_mb, discard_visual)
    if binary_cache is None:
        binary_cache = os.environ.get("EXO_SIM_NO_MJB_CACHE", "") == ""
    if not binary_cache:
        return _compile(path, xml, rewritten)

    cache = CACHE_DIR / f"{path.stem}.{_cache_key(path, rewritten)}.mjb"
    if cache.is_file():
        try:
            return mujoco.MjModel.from_binary_path(str(cache))
        except Exception:
            pass          # unreadable entry: fall through and rebuild it
    model = _compile(path, xml, rewritten)
    tmp = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        mujoco.mj_saveModel(model, str(tmp), None)
        os.replace(tmp, cache)
    except Exception:
        # A failed cache write shouldn't fail the load.
        try:
            tmp.unlink()
        except OSError:
            pass
    return model
