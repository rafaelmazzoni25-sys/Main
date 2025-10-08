"""Terrain visualization utility.

Ferramenta em Python para visualizar o terreno e objetos estáticos em 3D.
Ela reutiliza os mesmos formatos do cliente legado para gerar uma malha
matricial e sobrepor os objetos com Matplotlib.
"""
from __future__ import annotations

import argparse
import ast
import csv
import functools
import io
import itertools
import json
import math
import re
import struct
import textwrap
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.backend_bases import KeyEvent, PickEvent
from matplotlib.colors import LightSource
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Path3DCollection
from PIL import Image
from tkinter import filedialog, messagebox

try:  # Optional OpenGL stack
    import moderngl
except Exception:  # noqa: BLE001
    moderngl = None  # type: ignore[assignment]

try:  # Optional windowing backend
    import pyglet
    from pyglet import gl as pyglet_gl  # noqa: F401  # used for type checking / side-effects
    from pyglet.window import key as pyglet_key, mouse as pyglet_mouse
except Exception:  # noqa: BLE001
    pyglet = None  # type: ignore[assignment]
    pyglet_gl = None  # type: ignore[assignment]
    pyglet_key = None  # type: ignore[assignment]
    pyglet_mouse = None  # type: ignore[assignment]


def _get_depth_mask(ctx: "moderngl.Context") -> Optional[bool]:
    if moderngl is None:
        return None
    targets = [ctx, getattr(ctx, "screen", None)]
    for target in targets:
        if target is None:
            continue
        try:
            mask = getattr(target, "depth_mask")
        except AttributeError:
            continue
        except Exception:  # noqa: BLE001
            continue
        if isinstance(mask, bool):
            return mask
        if isinstance(mask, (int, np.integer)):
            return bool(mask)
    return None


def _set_depth_mask(ctx: "moderngl.Context", value: bool) -> bool:
    if moderngl is None:
        return False
    updated = False
    targets = [ctx, getattr(ctx, "screen", None)]
    for target in targets:
        if target is None:
            continue
        try:
            setattr(target, "depth_mask", bool(value))
        except AttributeError:
            continue
        except Exception:  # noqa: BLE001
            continue
        else:
            updated = True
    return updated

TERRAIN_SIZE = 256
TERRAIN_SCALE = 100.0
BUX_CODE = (0xFC, 0xCF, 0xAB)
MAP_XOR_KEY = (
    0xD1,
    0x73,
    0x52,
    0xF6,
    0xD2,
    0x9A,
    0xCB,
    0x27,
    0x3E,
    0xAF,
    0x59,
    0x31,
    0x37,
    0xB3,
    0xE7,
    0xA2,
)
G_MIN_HEIGHT = -500.0

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENUM_PATH = REPO_ROOT / "source" / "_enum.h"

MODEL_LINE_RE = re.compile(r"^(MODEL_[A-Z0-9_]+)\s*(?:=\s*([^,]+))?\s*,?$")
CONST_LINE_RE = re.compile(r"^([A-Z0-9_]+)\s*=\s*([^,]+)\s*,?$")

TILE_TEXTURE_CANDIDATES: Dict[int, List[str]] = {
    0: ["TileGrass01", "TileGrass01_R"],
    1: ["TileGrass02"],
    2: ["TileGround01", "AlphaTileGround01", "AlphaTile01"],
    3: ["TileGround02", "AlphaTileGround02"],
    4: ["TileGround03", "AlphaTileGround03"],
    5: ["TileWater01", "Object25/water1", "Object25/water2"],
    6: ["TileWood01"],
    7: ["TileRock01"],
    8: ["TileRock02"],
    9: ["TileRock03"],
    10: ["TileRock04", "AlphaTile01"],
    11: ["TileRock05", "Object64/song_lava1"],
    12: ["TileRock06", "AlphaTile01"],
    13: ["TileRock07"],
}

for ext_index in range(1, 17):
    TILE_TEXTURE_CANDIDATES[13 + ext_index] = [f"ExtTile{ext_index:02d}"]

IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".ozj",
    ".ozt",
    ".jpg",
    ".jpeg",
    ".png",
    ".tga",
    ".bmp",
)

WATER_TILE_IDS = {5, 44, 45, 46}
LAVA_TILE_IDS = {11, 47}
TRANSPARENT_TILE_IDS = {2, 3, 4, 5, 10, 11, 12, 44, 45, 46, 47}

MATERIAL_WATER = 1 << 0
MATERIAL_LAVA = 1 << 1
MATERIAL_TRANSPARENT = 1 << 2
MATERIAL_ADDITIVE = 1 << 3
MATERIAL_EMISSIVE = 1 << 4
MATERIAL_ALPHA_TEST = 1 << 5
MATERIAL_DOUBLE_SIDED = 1 << 6
MATERIAL_NO_SHADOW = 1 << 7
MATERIAL_NORMAL_MAP = 1 << 8

MAX_POINT_LIGHTS = 8


def _eval_int_expression(expr: str, env: Mapping[str, int]) -> int:
    node = ast.parse(expr, mode="eval")

    def _eval(parsed: ast.AST) -> int:
        if isinstance(parsed, ast.Expression):
            return _eval(parsed.body)
        if isinstance(parsed, ast.Constant):
            if isinstance(parsed.value, (int, float)):
                return int(parsed.value)
            raise ValueError(f"Valor constante inesperado: {parsed.value!r}")
        if isinstance(parsed, ast.UnaryOp):
            operand = _eval(parsed.operand)
            if isinstance(parsed.op, ast.UAdd):
                return +operand
            if isinstance(parsed.op, ast.USub):
                return -operand
            raise ValueError(f"Operador unário não suportado: {parsed.op}")
        if isinstance(parsed, ast.BinOp):
            left = _eval(parsed.left)
            right = _eval(parsed.right)
            if isinstance(parsed.op, ast.Add):
                return left + right
            if isinstance(parsed.op, ast.Sub):
                return left - right
            if isinstance(parsed.op, ast.Mult):
                return left * right
            if isinstance(parsed.op, ast.FloorDiv):
                return left // right
            if isinstance(parsed.op, ast.LShift):
                return left << right
            if isinstance(parsed.op, ast.RShift):
                return left >> right
            if isinstance(parsed.op, ast.BitOr):
                return left | right
            if isinstance(parsed.op, ast.BitAnd):
                return left & right
            if isinstance(parsed.op, ast.BitXor):
                return left ^ right
            raise ValueError(f"Operador não suportado: {parsed.op}")
        if isinstance(parsed, ast.Name):
            if parsed.id not in env:
                raise KeyError(parsed.id)
            return int(env[parsed.id])
        raise ValueError(f"Expressão não suportada: {ast.dump(parsed)}")

    return _eval(node)


@functools.lru_cache(maxsize=None)
def load_model_names(enum_path: str) -> Dict[int, str]:
    path = Path(enum_path)
    if not path.exists():
        return {}

    env: Dict[str, int] = {"MAX_CLASS": 7}
    names: Dict[int, str] = {}
    current_value: Optional[int] = None
    inside_world_section = False

    with path.open(encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.split("//", 1)[0].strip()
            if not line:
                continue

            if not inside_world_section:
                if line.startswith("MODEL_WORLD_OBJECT"):
                    inside_world_section = True
                else:
                    continue

            if line.startswith("//skill") or line.startswith("MODEL_SKILL_BEGIN"):
                break

            model_match = MODEL_LINE_RE.match(line)
            if model_match:
                name, expr = model_match.groups()
                if expr is not None:
                    value = _eval_int_expression(expr.strip(), env)
                    current_value = value
                else:
                    if current_value is None:
                        continue
                    current_value += 1
                    value = current_value
                env[name] = value
                names[value] = name
                continue

            const_match = CONST_LINE_RE.match(line)
            if const_match:
                const_name, expr = const_match.groups()
                value = _eval_int_expression(expr.strip(), env)
                env[const_name] = value
                if const_name.startswith("MODEL_"):
                    names[value] = const_name
                current_value = value

    return names


@dataclass
class TerrainData:
    height: np.ndarray
    mapping_layer1: np.ndarray
    mapping_layer2: np.ndarray
    mapping_alpha: np.ndarray
    attributes: np.ndarray


@dataclass
class MaterialState:
    name: str
    src_blend: str = "one"
    dst_blend: str = "zero"
    blend_op: str = "add"
    alpha_test: bool = False
    alpha_func: str = "always"
    alpha_ref: float = 0.0
    depth_write: bool = True
    depth_test: bool = True
    cull_mode: str = "back"
    emissive: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    specular: Tuple[float, float, float] = (0.2, 0.2, 0.2)
    specular_power: float = 16.0
    double_sided: bool = False
    additive: bool = False
    transparent: bool = False
    water: bool = False
    lava: bool = False
    normal_map: Optional[str] = None
    receive_shadows: bool = True

    def to_flags(self) -> int:
        flags = 0
        if self.water:
            flags |= MATERIAL_WATER | MATERIAL_TRANSPARENT
        if self.lava:
            flags |= MATERIAL_LAVA | MATERIAL_TRANSPARENT
        if self.transparent:
            flags |= MATERIAL_TRANSPARENT
        if self.additive:
            flags |= MATERIAL_ADDITIVE
        if np.linalg.norm(self.emissive) > 1e-3:
            flags |= MATERIAL_EMISSIVE
        if self.alpha_test:
            flags |= MATERIAL_ALPHA_TEST
        if self.double_sided or self.cull_mode.lower() == "none":
            flags |= MATERIAL_DOUBLE_SIDED
        if not self.receive_shadows:
            flags |= MATERIAL_NO_SHADOW
        if self.normal_map:
            flags |= MATERIAL_NORMAL_MAP
        return flags


class MaterialStateLibrary:
    def __init__(
        self,
        world_path: Optional[Path] = None,
        *,
        extra_roots: Sequence[Path] = (),
    ) -> None:
        roots: List[Path] = []
        if world_path is not None:
            world_root = world_path.parent if world_path.is_file() else world_path
            if world_root.exists():
                roots.append(world_root)
                textures = world_root / "Texture"
                if textures.exists():
                    roots.append(textures)
        for root in extra_roots:
            if root.exists():
                resolved = root.resolve()
                if resolved not in roots:
                    roots.append(resolved)
        self.search_roots = roots
        self._states: Dict[str, MaterialState] = {}
        self._aliases: Dict[str, str] = {}
        self._load_tables()

    def _normalize_key(self, name: str) -> str:
        lowered = name.replace("\\", "/").lower()
        for ext in IMAGE_EXTENSIONS + (".dds",):
            if lowered.endswith(ext.lower()):
                lowered = lowered[: -len(ext)]
        return lowered

    def _iter_candidate_files(self) -> Iterator[Path]:
        candidates = (
            "materials.json",
            "material_table.json",
            "material_table.txt",
            "material_table.csv",
            "materialstate.json",
            "materialstate.txt",
        )
        for root in self.search_roots:
            for name in candidates:
                path = root / name
                if path.is_file():
                    yield path

    def _load_tables(self) -> None:
        for table in self._iter_candidate_files():
            try:
                suffix = table.suffix.lower()
                if suffix == ".json":
                    self._load_json(table)
                elif suffix == ".csv":
                    self._load_csv(table)
                else:
                    self._load_txt(table)
            except Exception:  # noqa: BLE001
                continue

    def _load_json(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            entries = payload.items()
        elif isinstance(payload, Sequence):
            entries = []
            for item in payload:
                if isinstance(item, Mapping) and "name" in item:
                    entries.append((item["name"], item))
        else:
            return
        for key, value in entries:  # type: ignore[arg-type]
            self._store_entry(str(key), value)

    def _load_csv(self, path: Path) -> None:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                name = row.get("name") or row.get("texture")
                if not name:
                    continue
                self._store_entry(name, row)

    def _load_txt(self, path: Path) -> None:
        with path.open(encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    key, _, rest = line.partition(":")
                    data = {"params": rest.strip()}
                    self._store_entry(key.strip(), data)
                    continue
                parts = [part.strip() for part in line.split(",")]
                if not parts:
                    continue
                name = parts[0]
                values = {
                    "src_blend": parts[1] if len(parts) > 1 else "one",
                    "dst_blend": parts[2] if len(parts) > 2 else "zero",
                    "alpha_test": parts[3] if len(parts) > 3 else "false",
                    "alpha_ref": parts[4] if len(parts) > 4 else "0",
                    "depth_write": parts[5] if len(parts) > 5 else "true",
                    "depth_test": parts[6] if len(parts) > 6 else "true",
                    "cull_mode": parts[7] if len(parts) > 7 else "back",
                }
                self._store_entry(name, values)

    def _parse_bool(self, value: object, *, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        return default

    def _parse_vec3(self, value: object) -> Tuple[float, float, float]:
        if isinstance(value, Sequence) and len(value) >= 3:
            return (float(value[0]), float(value[1]), float(value[2]))
        if isinstance(value, str):
            parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
            if len(parts) >= 3:
                return (float(parts[0]), float(parts[1]), float(parts[2]))
        return (0.0, 0.0, 0.0)

    def _store_entry(self, name: str, payload: Mapping[str, object]) -> None:
        normalized = self._normalize_key(name)
        state = MaterialState(
            name=normalized,
            src_blend=str(payload.get("src_blend", payload.get("src", "one"))).lower(),
            dst_blend=str(payload.get("dst_blend", payload.get("dst", "zero"))).lower(),
            blend_op=str(payload.get("blend_op", payload.get("op", "add"))).lower(),
            alpha_test=self._parse_bool(payload.get("alpha_test", payload.get("alpha", False))),
            alpha_func=str(payload.get("alpha_func", payload.get("alpha_compare", "greater"))).lower(),
            alpha_ref=float(payload.get("alpha_ref", payload.get("alpha_value", 0.0)) or 0.0),
            depth_write=self._parse_bool(payload.get("depth_write", payload.get("zwrite", True)), default=True),
            depth_test=self._parse_bool(payload.get("depth_test", payload.get("ztest", True)), default=True),
            cull_mode=str(payload.get("cull_mode", payload.get("cull", "back"))).lower(),
            emissive=self._parse_vec3(payload.get("emissive", payload.get("emissive_color", (0, 0, 0)))),
            specular=self._parse_vec3(payload.get("specular", payload.get("specular_color", (0.2, 0.2, 0.2)))),
            specular_power=float(payload.get("specular_power", payload.get("power", 16.0)) or 16.0),
            double_sided=self._parse_bool(payload.get("double_sided", payload.get("two_sided", False))),
            additive=self._parse_bool(payload.get("additive", payload.get("blend_add", False))),
            transparent=self._parse_bool(payload.get("transparent", payload.get("alpha_blend", False))),
            water=self._parse_bool(payload.get("water", False)),
            lava=self._parse_bool(payload.get("lava", False)),
            normal_map=str(payload.get("normal_map", payload.get("normal", ""))) or None,
            receive_shadows=self._parse_bool(payload.get("receive_shadows", payload.get("shadow", True)), default=True),
        )
        self._states[normalized] = state
        base = Path(normalized).stem
        if base and base not in self._states:
            self._aliases[base] = normalized

    def lookup(self, texture_name: str) -> MaterialState:
        if not texture_name:
            return MaterialState(name="")
        key = self._normalize_key(texture_name)
        if key in self._states:
            return self._states[key]
        base = Path(key).stem
        if base in self._states:
            return self._states[base]
        alias = self._aliases.get(key)
        if alias and alias in self._states:
            return self._states[alias]
        return self._build_fallback(texture_name)

    def _build_fallback(self, texture_name: str) -> MaterialState:
        lowered = texture_name.lower()
        state = MaterialState(name=self._normalize_key(texture_name))
        if any(token in lowered for token in ("water", "river", "wave", "ocean", "sea")):
            state.water = True
            state.transparent = True
            state.depth_write = False
            state.src_blend = "src_alpha"
            state.dst_blend = "one_minus_src_alpha"
        if any(token in lowered for token in ("lava", "magma", "volcano", "fire")):
            state.lava = True
            state.transparent = True
            state.additive = True
            state.src_blend = "src_alpha"
            state.dst_blend = "one"
        if any(token in lowered for token in ("alpha", "glass", "trans", "smoke", "light", "flare")):
            state.transparent = True
            state.src_blend = "src_alpha"
            state.dst_blend = "one_minus_src_alpha"
        if "emissive" in lowered or "light" in lowered:
            state.emissive = (0.6, 0.6, 0.6)
        return state

    def flags_for_texture(self, texture_name: str) -> int:
        return self.lookup(texture_name).to_flags()

    def flags_for_tile(self, texture_name: str) -> int:
        return self.flags_for_texture(texture_name)


_FALLBACK_MATERIAL_LIBRARY = MaterialStateLibrary()


def _resolve_material_state(
    texture_name: str,
    material_library: Optional[MaterialStateLibrary],
) -> MaterialState:
    if material_library is not None:
        return material_library.lookup(texture_name)
    return _FALLBACK_MATERIAL_LIBRARY.lookup(texture_name)


def _blend_factor(name: str) -> int:
    if moderngl is None:
        raise RuntimeError("Blending avançado requer moderngl instalado.")
    mapping = {
        "zero": moderngl.ZERO,
        "one": moderngl.ONE,
        "src_alpha": moderngl.SRC_ALPHA,
        "one_minus_src_alpha": moderngl.ONE_MINUS_SRC_ALPHA,
        "dst_alpha": moderngl.DST_ALPHA,
        "one_minus_dst_alpha": moderngl.ONE_MINUS_DST_ALPHA,
        "src_color": moderngl.SRC_COLOR,
        "one_minus_src_color": moderngl.ONE_MINUS_SRC_COLOR,
        "dst_color": moderngl.DST_COLOR,
        "one_minus_dst_color": moderngl.ONE_MINUS_DST_COLOR,
    }
    key = name.strip().lower()
    return mapping.get(key, moderngl.SRC_ALPHA)


@dataclass
class TerrainObject:
    type_id: int
    position: Tuple[float, float, float]
    angles: Tuple[float, float, float]
    scale: float
    type_name: Optional[str] = None

    @property
    def tile_position(self) -> Tuple[float, float]:
        return (self.position[0] / TERRAIN_SCALE, self.position[1] / TERRAIN_SCALE)


@dataclass
class BMDMesh:
    name: str
    positions: np.ndarray
    normals: np.ndarray
    texcoords: np.ndarray
    indices: np.ndarray
    texture_name: str
    material_flags: int = 0
    material_state: MaterialState = field(default_factory=lambda: MaterialState(name=""))
    bone_indices: Optional[np.ndarray] = None


@dataclass
class BMDKeyframe:
    translation: np.ndarray
    rotation: np.ndarray
    time: float
    rotation_quat: np.ndarray


@dataclass
class BMDAnimationChannel:
    bone_index: int
    keyframes: List[BMDKeyframe] = field(default_factory=list)


@dataclass
class BMDAnimationEvent:
    name: str
    time: float
    params: Tuple[float, ...] = ()


@dataclass
class BMDAnimation:
    name: str
    duration: float
    frames_per_second: float
    channels: Dict[int, BMDAnimationChannel] = field(default_factory=dict)
    events: List[BMDAnimationEvent] = field(default_factory=list)
    default_blend: float = 0.25


@dataclass
class BMDHardpoint:
    name: str
    bone_index: int
    offset: np.ndarray
    rotation_quat: np.ndarray
    is_billboard: bool = False
    color: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0], dtype=np.float32))
    radius: float = 1200.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    size: float = 1.0


@dataclass
class BMDBone:
    name: str
    parent: int
    rest_translation: np.ndarray
    rest_rotation: np.ndarray
    rest_quaternion: np.ndarray
    rest_matrix: np.ndarray
    inverse_bind_matrix: np.ndarray


@dataclass
class BMDModel:
    name: str
    meshes: List[BMDMesh] = field(default_factory=list)
    version: int = 0
    bones: List[BMDBone] = field(default_factory=list)
    animations: Dict[str, BMDAnimation] = field(default_factory=dict)
    attachments: List[BMDHardpoint] = field(default_factory=list)
    billboards: List[BMDHardpoint] = field(default_factory=list)

    @property
    def has_transparency(self) -> bool:
        return any(mesh.material_flags & MATERIAL_TRANSPARENT for mesh in self.meshes)


@dataclass
class BMDActionInfo:
    name: str
    key_count: int
    lock_positions: bool
    base_positions: List[np.ndarray] = field(default_factory=list)


@dataclass
class TerrainLoadResult:
    world_path: Path
    data: TerrainData
    objects: List[TerrainObject]
    map_id: int
    map_id_attribute: int
    map_id_mapping: int
    map_id_objects: int
    model_names: Mapping[int, str]
    objects_path: Path
    objects_version: int
    all_objects: List[TerrainObject]
    warnings: List[str] = field(default_factory=list)


def _bilinear_resize(matrix: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return matrix.astype(np.float32, copy=False)
    height, width = matrix.shape
    new_height = height * factor
    new_width = width * factor
    src_y = np.arange(height, dtype=np.float32)
    src_x = np.arange(width, dtype=np.float32)
    dst_y = np.linspace(0.0, float(height - 1), new_height, dtype=np.float32)
    dst_x = np.linspace(0.0, float(width - 1), new_width, dtype=np.float32)

    matrix_f = matrix.astype(np.float32)
    temp = np.empty((height, new_width), dtype=np.float32)
    for y_idx in range(height):
        temp[y_idx] = np.interp(dst_x, src_x, matrix_f[y_idx])

    result = np.empty((new_height, new_width), dtype=np.float32)
    for x_idx in range(new_width):
        result[:, x_idx] = np.interp(dst_y, src_y, temp[:, x_idx])
    return result


def _compute_normal_map(heights: np.ndarray, spacing: float) -> np.ndarray:
    if heights.ndim != 2:
        raise ValueError("Mapa de altura deve ser bidimensional para gerar normal map.")
    grad_y, grad_x = np.gradient(heights, spacing, spacing)
    normals = np.stack((-grad_x, np.ones_like(grad_x), -grad_y), axis=2)
    lengths = np.linalg.norm(normals, axis=2, keepdims=True)
    lengths = np.where(lengths == 0.0, 1.0, lengths)
    normalized = normals / lengths
    return normalized[:-1, :-1, :]


def _compute_shadow_mask(heights: np.ndarray, light_dir: np.ndarray, step_scale: float) -> np.ndarray:
    h, w = heights.shape
    if h < 2 or w < 2:
        return np.ones((h, w), dtype=np.float32)
    dir2d = np.array([light_dir[0], light_dir[2]], dtype=np.float32)
    length = float(np.linalg.norm(dir2d))
    if length == 0.0:
        return np.ones((h - 1, w - 1), dtype=np.float32)
    dir2d /= length
    steps = 64
    mask = np.ones((h - 1, w - 1), dtype=np.float32)
    for y in range(h - 1):
        for x in range(w - 1):
            base_height = heights[y, x]
            shadowed = False
            px = float(x)
            py = float(y)
            for step in range(1, steps):
                px += dir2d[0]
                py += dir2d[1]
                ix = int(px)
                iy = int(py)
                if ix < 0 or iy < 0 or ix >= w - 1 or iy >= h - 1:
                    break
                sample_height = heights[iy, ix]
                expected = base_height + light_dir[1] * step * step_scale
                if sample_height > expected:
                    shadowed = True
                    break
            mask[y, x] = 0.35 if shadowed else 1.0
    return mask


def _ensure_rgba(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[2] == 4:
        return image
    alpha = np.full((*image.shape[:2], 1), 255, dtype=image.dtype)
    return np.concatenate([image, alpha], axis=2)


def _load_ozj(path: Path) -> Optional[np.ndarray]:
    data = path.read_bytes()
    if len(data) <= 24:
        return None
    payload = data[24:]
    try:
        with Image.open(io.BytesIO(payload)) as img:
            return _ensure_rgba(np.array(img.convert("RGBA")))
    except Exception:
        return None


def _load_ozt(path: Path) -> Optional[np.ndarray]:
    raw = path.read_bytes()
    if len(raw) <= 20:
        return None
    idx = 12
    idx += 4
    if idx + 5 > len(raw):
        return None
    nx = struct.unpack_from("<H", raw, idx)[0]
    idx += 2
    ny = struct.unpack_from("<H", raw, idx)[0]
    idx += 2
    bit_depth = raw[idx]
    idx += 1
    idx += 1  # image descriptor
    if bit_depth != 32:
        return None
    expected = nx * ny * 4
    if idx + expected > len(raw):
        return None
    buffer = np.frombuffer(raw, dtype=np.uint8, count=expected, offset=idx)
    buffer = buffer.reshape((ny, nx, 4))
    rgba = np.empty_like(buffer)
    rgba[..., 0] = buffer[..., 2]
    rgba[..., 1] = buffer[..., 1]
    rgba[..., 2] = buffer[..., 0]
    rgba[..., 3] = buffer[..., 3]
    rgba = np.flipud(rgba)
    return rgba


def _load_image_file(path: Path) -> Optional[np.ndarray]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".ozj":
            return _load_ozj(path)
        if suffix == ".ozt":
            return _load_ozt(path)
        with Image.open(path) as img:
            return _ensure_rgba(np.array(img.convert("RGBA")))
    except Exception:
        return None


def _iter_candidate_paths(
    search_roots: Sequence[Path],
    base_name: str,
    extensions: Sequence[str],
) -> Iterable[Path]:
    normalized = base_name.replace("\\", "/")
    variants = {base_name, normalized, normalized.lower(), normalized.upper()}
    for root in search_roots:
        for name in variants:
            path = root / name
            if path.is_file():
                yield path
            for ext in extensions:
                target = root / f"{name}{ext}"
                if target.is_file():
                    yield target
                target_with_dot = root / f"{name}.{ext.lstrip('.') }"
                if target_with_dot.is_file():
                    yield target_with_dot


class TextureLibrary:
    def __init__(
        self,
        world_path: Path,
        *,
        detail_factor: int = 2,
        object_path: Optional[Path] = None,
        map_id: Optional[int] = None,
    ) -> None:
        self.world_path = world_path
        self.detail_factor = max(1, detail_factor)
        self.object_path = object_path
        self.map_id = map_id
        self._image_cache: Dict[int, Optional[np.ndarray]] = {}
        self._resized_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._missing: set[int] = set()
        self._fallback_cmap = cm.get_cmap("tab20")
        self.search_roots = self._build_search_roots()
        self._light_image_loaded = False
        self._light_image_uint8: Optional[np.ndarray] = None
        self._light_map_cache: Dict[int, np.ndarray] = {}

    def _light_candidates(self) -> List[str]:
        base = [
            "TerrainLight",
            "TerrainLight0",
            "TerrainLight1",
            "TerrainLight2",
            "TerrainLight3",
        ]
        prioritized: List[str] = []
        if self.map_id == 30:  # Battle Castle
            prioritized.extend(["TerrainLight2", "TerrainLight"])
        elif self.map_id == 34:  # Crywolf
            prioritized.extend(["TerrainLight", "TerrainLight1", "TerrainLight2"])

        ordered: List[str] = []
        seen: Set[str] = set()
        for name in itertools.chain(prioritized, base):
            if name in seen:
                continue
            seen.add(name)
            ordered.append(name)
        return ordered

    def _build_search_roots(self) -> List[Path]:
        roots: List[Path] = []

        def add(path: Optional[Path]) -> None:
            if path and path.exists() and path.is_dir() and path not in roots:
                roots.append(path)

        add(self.world_path)
        add(self.world_path.parent)
        add(self.object_path)
        parent = self.world_path.parent
        if parent.exists():
            for child in sorted(parent.iterdir()):
                if child.is_dir():
                    lowered = child.name.lower()
                    if lowered.startswith("object") or lowered.startswith("world"):
                        add(child)
        return roots

    def _load_light_image(self) -> Optional[np.ndarray]:
        if self._light_image_loaded:
            return self._light_image_uint8
        self._light_image_loaded = True
        search = [self.world_path]
        for name in self._light_candidates():
            for path in _iter_candidate_paths(search, name, IMAGE_EXTENSIONS):
                image = _load_image_file(path)
                if image is None:
                    continue
                if image.ndim == 2:
                    rgb = np.stack([image, image, image], axis=-1)
                else:
                    rgb = image[..., :3]
                rgb_uint8 = np.asarray(rgb, dtype=np.uint8)
                self._light_image_uint8 = rgb_uint8
                base_size = rgb_uint8.shape[0]
                self._light_map_cache[base_size] = rgb_uint8.astype(np.float32) / 255.0
                return self._light_image_uint8
        self._light_image_uint8 = None
        return None

    def light_map_texture(self, target_size: int) -> Optional[np.ndarray]:
        if target_size <= 0:
            return None
        if target_size in self._light_map_cache:
            return self._light_map_cache[target_size]
        source = self._load_light_image()
        if source is None:
            return None
        rgb = source[..., :3]
        if rgb.shape[0] == target_size and rgb.shape[1] == target_size:
            result = rgb.astype(np.float32) / 255.0
        else:
            image = Image.fromarray(rgb)
            resized = image.resize((target_size, target_size), Image.BILINEAR)
            result = np.asarray(resized, dtype=np.float32) / 255.0
        if result.ndim == 2:
            result = np.stack([result, result, result], axis=-1)
        elif result.shape[2] == 4:
            result = result[:, :, :3]
        self._light_map_cache[target_size] = result.astype(np.float32)
        return self._light_map_cache[target_size]

    def light_direction(self) -> np.ndarray:
        if self.map_id == 30:
            direction = np.array([-0.5, 1.0, -1.0], dtype=np.float32)
        else:
            direction = np.array([-0.5, 0.5, -0.5], dtype=np.float32)
        return _normalize(direction)

    def _fallback_color(self, index: int) -> np.ndarray:
        rgba = self._fallback_cmap((index % 20) / 20.0)
        return np.array(rgba, dtype=np.float32)

    def _load_texture(self, index: int) -> Optional[np.ndarray]:
        if index in self._image_cache:
            return self._image_cache[index]
        candidates = TILE_TEXTURE_CANDIDATES.get(index, [])
        if not candidates:
            candidates = [f"ExtTile{index:02d}"]
        image: Optional[np.ndarray] = None
        for base_name in candidates:
            for path in _iter_candidate_paths(self.search_roots, base_name, IMAGE_EXTENSIONS):
                image = _load_image_file(path)
                if image is not None:
                    break
            if image is not None:
                break
        self._image_cache[index] = image
        if image is None:
            self._missing.add(index)
        return image

    def _tile_patch(self, index: int, size: int) -> np.ndarray:
        cache_key = (index, size)
        if cache_key in self._resized_cache:
            return self._resized_cache[cache_key]
        image = self._load_texture(index)
        if image is None:
            color = self._fallback_color(index)
            tile = np.ones((size, size, 4), dtype=np.float32)
            tile[..., 0:3] = color[:3]
            tile[..., 3] = color[3]
        else:
            pil_image = Image.fromarray(image)
            if pil_image.mode != "RGBA":
                pil_image = pil_image.convert("RGBA")
            resized = pil_image.resize((size, size), Image.BILINEAR)
            tile = np.asarray(resized, dtype=np.float32) / 255.0
            if tile.ndim == 2:
                tile = np.stack([tile, tile, tile, np.ones_like(tile)], axis=-1)
            elif tile.shape[2] == 3:
                alpha = np.ones((size, size, 1), dtype=np.float32)
                tile = np.concatenate([tile, alpha], axis=2)
        self._resized_cache[cache_key] = tile.astype(np.float32)
        return self._resized_cache[cache_key]

    def tile_texture_name(self, index: int) -> Optional[str]:
        candidates = TILE_TEXTURE_CANDIDATES.get(index, [])
        if candidates:
            return candidates[0]
        return None

    def compose_texture_pixels(
        self,
        layer1: np.ndarray,
        layer2: np.ndarray,
        alpha: np.ndarray,
    ) -> np.ndarray:
        factor = self.detail_factor
        height, width = layer1.shape
        pixels = np.zeros((height * factor, width * factor, 4), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                idx1 = int(layer1[y, x])
                idx2 = int(layer2[y, x])
                alpha_val = float(alpha[y, x])
                base_patch = self._tile_patch(idx1, factor)
                tile_patch = base_patch
                if idx2 != 255 and alpha_val > 0.0:
                    overlay_patch = self._tile_patch(idx2, factor)
                    tile_patch = (1.0 - alpha_val) * base_patch + alpha_val * overlay_patch
                y0 = y * factor
                y1 = y0 + factor
                x0 = x * factor
                x1 = x0 + factor
                pixels[y0:y1, x0:x1, :] = tile_patch
        return pixels

    def build_surface(self, data: TerrainData) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        factor = self.detail_factor
        heights = data.height.astype(np.float32)
        refined_heights = _bilinear_resize(heights, factor)
        texture_pixels = self.compose_texture_pixels(
            data.mapping_layer1, data.mapping_layer2, data.mapping_alpha
        )
        x_coords = np.linspace(0.0, float(TERRAIN_SIZE - 1), refined_heights.shape[1], dtype=np.float32)
        y_coords = np.linspace(0.0, float(TERRAIN_SIZE - 1), refined_heights.shape[0], dtype=np.float32)
        xx, yy = np.meshgrid(x_coords, y_coords)
        facecolors = np.clip(texture_pixels[:-1, :-1, :], 0.0, 1.0)
        normals = _compute_normal_map(refined_heights, float(TERRAIN_SCALE) / float(self.detail_factor))
        if normals.size == 0:
            shading_rgb = np.ones((facecolors.shape[0], facecolors.shape[1], 3), dtype=np.float32)
        else:
            light_dir = self.light_direction()
            luminosity = np.sum(normals * light_dir.reshape(1, 1, 3), axis=2)
            luminosity = np.clip(luminosity + 0.5, 0.0, 1.0)
            shading_rgb = np.repeat(luminosity[:, :, None], 3, axis=2)
            light_map = self.light_map_texture(texture_pixels.shape[0])
            if light_map is not None:
                shading_rgb *= np.clip(light_map[:-1, :-1, :3], 0.0, 1.0)
        facecolors[..., :3] *= shading_rgb
        facecolors = np.clip(facecolors, 0.0, 1.0)
        return xx, yy, refined_heights, facecolors

    @property
    def missing_indices(self) -> Sequence[int]:
        return sorted(self._missing)


def compute_tile_material_flags(
    layer1: np.ndarray,
    layer2: np.ndarray,
    alpha: np.ndarray,
    texture_library: TextureLibrary,
    material_library: Optional[MaterialStateLibrary],
) -> np.ndarray:
    flags = np.zeros_like(layer1, dtype=np.uint32)
    if layer1.size == 0:
        return flags

    cache: Dict[int, int] = {}

    def tile_flags(index: int) -> int:
        if index in cache:
            return cache[index]
        if index < 0 or index == 255:
            cache[index] = 0
            return 0
        texture_name = texture_library.tile_texture_name(index) or f"ExtTile{index:02d}"
        state = _resolve_material_state(texture_name, material_library)
        cache[index] = state.to_flags()
        return cache[index]

    it = np.ndindex(layer1.shape)
    for coord in it:
        idx1 = int(layer1[coord])
        idx2 = int(layer2[coord])
        base = tile_flags(idx1)
        overlay = 0
        if idx2 != 255 and float(alpha[coord]) > 0.01:
            overlay = tile_flags(idx2)
        flags[coord] = base | overlay
    return flags


def map_file_decrypt(data: bytes) -> bytes:
    """Reproduz a rotina inline MapFileDecrypt do cliente."""

    xor_key = MAP_XOR_KEY
    w_map_key = 0x5E
    out = bytearray(len(data))
    for idx, byte in enumerate(data):
        decrypted = ((byte ^ xor_key[idx % len(xor_key)]) - w_map_key) & 0xFF
        out[idx] = decrypted
        w_map_key = (byte + 0x3D) & 0xFF
    return bytes(out)


def map_file_encrypt(data: bytes) -> bytes:
    """Inverte a rotina inline para gerar novos arquivos EncTerrain."""

    xor_key = MAP_XOR_KEY
    w_map_key = 0x5E
    out = bytearray(len(data))
    for idx, byte in enumerate(data):
        value = (byte + w_map_key) & 0xFF
        encrypted = value ^ xor_key[idx % len(xor_key)]
        out[idx] = encrypted
        w_map_key = (encrypted + 0x3D) & 0xFF
    return bytes(out)


def _sanitize_c_string(raw: bytes) -> str:
    text = raw.split(b"\x00", 1)[0]
    return text.decode("latin1", errors="ignore").strip()


def _lerp_vec(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + (b - a) * t


def _quat_normalize(quat: np.ndarray) -> np.ndarray:
    quat = quat.astype(np.float32, copy=False)
    norm = float(np.linalg.norm(quat))
    if norm == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return quat / norm


def _quat_from_axis_angle(axis: Tuple[float, float, float], angle: float) -> np.ndarray:
    half = angle * 0.5
    s = math.sin(half)
    x, y, z = axis
    return np.array([x * s, y * s, z * s, math.cos(half)], dtype=np.float32)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float32,
    )


def _quat_from_euler_deg(rotation_deg: np.ndarray) -> np.ndarray:
    rx, ry, rz = np.radians(rotation_deg.astype(np.float32))
    qx = _quat_from_axis_angle((1.0, 0.0, 0.0), rx)
    qy = _quat_from_axis_angle((0.0, 1.0, 0.0), ry)
    qz = _quat_from_axis_angle((0.0, 0.0, 1.0), rz)
    quat = _quat_multiply(_quat_multiply(qz, qy), qx)
    return _quat_normalize(quat)


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def _quat_slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    a = _quat_normalize(a)
    b = _quat_normalize(b)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    dot = min(max(dot, -1.0), 1.0)
    if dot > 0.9995:
        return _quat_normalize(a + t * (b - a))
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return _quat_normalize((a * s0) + (b * s1))


def _compose_transform_matrix(translation: np.ndarray, rotation_deg: np.ndarray) -> np.ndarray:
    translation = translation.astype(np.float32, copy=False)
    rotation_deg = rotation_deg.astype(np.float32, copy=False)
    rx, ry, rz = np.radians(rotation_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    rotation = rot_z @ rot_y @ rot_x
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def _compose_transform_quaternion(translation: np.ndarray, rotation_quat: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = _quat_to_matrix(rotation_quat)
    matrix[:3, 3] = translation.astype(np.float32, copy=False)
    return matrix


def load_bmd_model(
    path: Path,
    material_library: Optional[MaterialStateLibrary] = None,
) -> BMDModel:
    raw = _read_file(path)
    if len(raw) < 7 or not raw.startswith(b"BMD"):
        raise ValueError(f"Arquivo BMD inválido: {path}")

    data = raw
    ptr = 3
    version = data[ptr]
    ptr += 1
    if version == 12:
        if ptr + 4 > len(data):
            raise ValueError(f"Cabeçalho corrompido em {path}")
        enc_size = struct.unpack_from("<I", data, ptr)[0]
        ptr += 4
        encrypted = data[ptr : ptr + enc_size]
        data = map_file_decrypt(encrypted)
        ptr = 0
    name = _sanitize_c_string(data[ptr : ptr + 32]) or path.stem
    ptr += 32
    if ptr + 6 > len(data):
        raise ValueError(f"Arquivo BMD incompleto: {path}")
    num_mesh, num_bones, num_actions = struct.unpack_from("<3H", data, ptr)
    ptr += 6
    meshes: List[BMDMesh] = []

    try:
        for mesh_index in range(num_mesh):
            if ptr + 10 > len(data):
                raise ValueError("Fim inesperado ao ler cabeçalho da malha")
            num_vertices, num_normals, num_texcoords, num_triangles, _texture_idx = struct.unpack_from(
                "<5H", data, ptr
            )
            ptr += 10

            vertices: List[Tuple[float, float, float]] = []
            vertex_nodes: List[int] = []
            for _ in range(num_vertices):
                node, x, y, z = struct.unpack_from("<hxx3f", data, ptr)
                vertex_nodes.append(int(node))
                vertices.append((x, y, z))
                ptr += 16

            for _ in range(num_normals):
                _ = struct.unpack_from("<hxx3fh", data, ptr)
                ptr += 18

            texcoords_raw: List[Tuple[float, float]] = []
            for _ in range(num_texcoords):
                u, v = struct.unpack_from("<2f", data, ptr)
                texcoords_raw.append((u, 1.0 - v))
                ptr += 8

            triangles: List[Tuple[int, Tuple[int, ...], Tuple[int, ...]]] = []
            for _ in range(num_triangles):
                polygon = struct.unpack_from("<b", data, ptr)[0]
                vertex_idx = struct.unpack_from("<4h", data, ptr + 1)
                tex_idx = struct.unpack_from("<4h", data, ptr + 17)
                triangles.append((polygon, tuple(vertex_idx), tuple(tex_idx)))
                ptr += 32

            texture_name = _sanitize_c_string(data[ptr : ptr + 32])
            ptr += 32

            vertex_array = np.asarray(vertices, dtype=np.float32)
            vertex_node_array = np.asarray(vertex_nodes, dtype=np.int16)
            texcoord_array = np.asarray(texcoords_raw, dtype=np.float32)

            mesh_vertices: List[Tuple[float, float, float]] = []
            mesh_normals: List[Tuple[float, float, float]] = []
            mesh_uvs: List[Tuple[float, float]] = []
            mesh_bones: List[int] = []

            def _safe_fetch(source: np.ndarray, index: int, fallback: Tuple[float, ...]) -> Tuple[float, ...]:
                if 0 <= index < len(source):
                    return tuple(source[index])  # type: ignore[return-value]
                return fallback

            for polygon, vert_idx, tex_idx in triangles:
                corners = 4 if polygon == 4 else 3
                base_indices = [0, 1, 2] if corners == 3 else [0, 1, 2, 3]
                first = [base_indices[0], base_indices[1], base_indices[2]]
                quads = []
                if corners == 4:
                    quads.append([base_indices[0], base_indices[2], base_indices[3]])

                def _append_triangle(order: Sequence[int]) -> None:
                    pts = []
                    for offset in order:
                        vid = vert_idx[offset]
                        tex_id = tex_idx[offset]
                        position = _safe_fetch(vertex_array, vid, (0.0, 0.0, 0.0))
                        uv = _safe_fetch(texcoord_array, tex_id, (0.0, 0.0))
                        pts.append((position, uv))
                    v0, v1, v2 = (np.array(p[0], dtype=np.float32) for p in pts[:3])
                    normal = np.cross(v1 - v0, v2 - v0)
                    length = float(np.linalg.norm(normal))
                    if length > 0.0:
                        normal = normal / length
                    else:
                        normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                    for index, (position, uv) in enumerate(pts):
                        mesh_vertices.append(position)
                        mesh_normals.append(tuple(normal.tolist()))
                        mesh_uvs.append(uv)
                        if index < len(pts):
                            source_idx = vert_idx[order[index]] if index < len(order) else vert_idx[0]
                        else:
                            source_idx = vert_idx[0]
                        if 0 <= source_idx < len(vertex_node_array):
                            mesh_bones.append(int(vertex_node_array[source_idx]))
                        else:
                            mesh_bones.append(-1)

                _append_triangle(first)
                for quad in quads:
                    _append_triangle(quad)

            positions = np.asarray(mesh_vertices, dtype=np.float32)
            normals = np.asarray(mesh_normals, dtype=np.float32)
            uvs = np.asarray(mesh_uvs, dtype=np.float32)
            indices = np.arange(len(positions), dtype=np.uint32)
            material_state = _resolve_material_state(texture_name, material_library)
            meshes.append(
                BMDMesh(
                    name=f"{name}_mesh{mesh_index}",
                    positions=positions,
                    normals=normals,
                    texcoords=uvs,
                    indices=indices,
                    texture_name=texture_name,
                    material_flags=material_state.to_flags(),
                    material_state=material_state,
                    bone_indices=np.asarray(mesh_bones, dtype=np.int16) if mesh_bones else None,
                )
            )

        action_infos: List[BMDActionInfo] = []
        for action_index in range(num_actions):
            if ptr + 3 > len(data):
                break
            keys = struct.unpack_from("<H", data, ptr)[0]
            ptr += 2
            lock_positions = struct.unpack_from("<?", data, ptr)[0]
            ptr += 1
            base_positions: List[np.ndarray] = []
            if lock_positions and keys > 0:
                available = min(keys, max((len(data) - ptr) // 12, 0))
                for _ in range(available):
                    if ptr + 12 > len(data):
                        break
                    vec = np.array(struct.unpack_from("<3f", data, ptr), dtype=np.float32)
                    base_positions.append(vec)
                    ptr += 12
                if available < keys:
                    base_positions.extend([np.zeros(3, dtype=np.float32)] * (keys - available))
                    ptr = len(data)
            action_infos.append(
                BMDActionInfo(
                    name=f"action_{action_index:02d}",
                    key_count=keys,
                    lock_positions=lock_positions,
                    base_positions=base_positions,
                )
            )

        bones: List[BMDBone] = []
        animations: Dict[str, BMDAnimation] = {}
        frames_per_second = 20.0
        for info in action_infos:
            duration = 0.0
            if info.key_count > 1:
                duration = float(info.key_count - 1) / frames_per_second
            animations[info.name] = BMDAnimation(
                name=info.name,
                duration=duration,
                frames_per_second=frames_per_second,
            )

        bone_index_map: Dict[int, int] = {}
        channel_stride = sum(info.key_count * 24 for info in action_infos)

        for raw_index in range(num_bones):
            if ptr >= len(data):
                break
            dummy = struct.unpack_from("<b", data, ptr)[0]
            ptr += 1
            if dummy:
                bone_index_map[raw_index] = -1
                continue

            bone_name = _sanitize_c_string(data[ptr : ptr + 32])
            ptr += 32
            parent = struct.unpack_from("<h", data, ptr)[0]
            ptr += 2

            remaining_bones = num_bones - raw_index
            bytes_remaining = len(data) - ptr
            include_rest = False
            if channel_stride > 0:
                include_rest = bytes_remaining >= remaining_bones * (channel_stride + 24)
            else:
                include_rest = bytes_remaining >= remaining_bones * 24

            rest_translation = np.zeros(3, dtype=np.float32)
            rest_rotation = np.zeros(3, dtype=np.float32)
            if include_rest and ptr + 24 <= len(data):
                rest_translation = np.array(struct.unpack_from("<3f", data, ptr), dtype=np.float32)
                ptr += 12
                rest_rotation = np.array(struct.unpack_from("<3f", data, ptr), dtype=np.float32)
                ptr += 12
            rest_quaternion = _quat_from_euler_deg(rest_rotation)

            bone = BMDBone(
                name=bone_name or f"bone_{len(bones)}",
                parent=parent,
                rest_translation=rest_translation,
                rest_rotation=rest_rotation,
                rest_quaternion=rest_quaternion,
                rest_matrix=np.eye(4, dtype=np.float32),
                inverse_bind_matrix=np.eye(4, dtype=np.float32),
            )
            bone_index = len(bones)
            bones.append(bone)
            bone_index_map[raw_index] = bone_index

            for info in action_infos:
                if info.key_count <= 0:
                    continue
                translations: List[np.ndarray] = []
                rotations: List[np.ndarray] = []
                for _ in range(info.key_count):
                    if ptr + 12 <= len(data):
                        translations.append(
                            np.array(struct.unpack_from("<3f", data, ptr), dtype=np.float32)
                        )
                        ptr += 12
                    else:
                        translations.append(np.zeros(3, dtype=np.float32))
                for _ in range(info.key_count):
                    if ptr + 12 <= len(data):
                        rotations.append(
                            np.array(struct.unpack_from("<3f", data, ptr), dtype=np.float32)
                        )
                        ptr += 12
                    else:
                        rotations.append(np.zeros(3, dtype=np.float32))

                channel = BMDAnimationChannel(bone_index=bone_index)
                fps = animations[info.name].frames_per_second
                for frame_index in range(info.key_count):
                    time_value = float(frame_index) / fps if fps > 0 else 0.0
                    translation = translations[min(frame_index, len(translations) - 1)]
                    if info.lock_positions and info.base_positions:
                        base = info.base_positions[min(frame_index, len(info.base_positions) - 1)]
                        translation = base
                    rotation = rotations[min(frame_index, len(rotations) - 1)]
                    channel.keyframes.append(
                        BMDKeyframe(
                            translation=translation.astype(np.float32),
                            rotation=rotation.astype(np.float32),
                            time=time_value,
                            rotation_quat=_quat_from_euler_deg(rotation),
                        )
                    )
                if channel.keyframes:
                    animations[info.name].channels[bone_index] = channel

        rest_globals: List[np.ndarray] = []
        for idx, bone in enumerate(bones):
            local = _compose_transform_quaternion(bone.rest_translation, bone.rest_quaternion)
            if 0 <= bone.parent < len(rest_globals):
                global_matrix = rest_globals[bone.parent] @ local
            else:
                global_matrix = local
            rest_globals.append(global_matrix)
            bones[idx].rest_matrix = global_matrix.astype(np.float32)
            try:
                bones[idx].inverse_bind_matrix = np.linalg.inv(global_matrix).astype(np.float32)
            except np.linalg.LinAlgError:
                bones[idx].inverse_bind_matrix = np.linalg.pinv(global_matrix).astype(np.float32)

        if bone_index_map:
            max_index = max(bone_index_map.keys())
            remap = np.full(max_index + 1, -1, dtype=np.int16)
            for raw_idx, mapped_idx in bone_index_map.items():
                if raw_idx <= max_index and mapped_idx is not None and mapped_idx >= 0:
                    remap[raw_idx] = mapped_idx
            for mesh in meshes:
                if mesh.bone_indices is None:
                    continue
                remapped = np.full_like(mesh.bone_indices, -1)
                for i, raw_idx in enumerate(mesh.bone_indices):
                    if 0 <= raw_idx < len(remap):
                        remapped[i] = remap[raw_idx]
                mesh.bone_indices = remapped

    except struct.error as exc:  # noqa: PERF203
        raise ValueError(f"Falha ao decodificar {path}: {exc}") from exc

    model = BMDModel(name=name, meshes=meshes, version=version, bones=bones, animations=animations)
    metadata = _load_bmd_metadata(path)
    if metadata:
        _apply_bmd_metadata(model, metadata)
    return model


def _load_bmd_metadata(path: Path) -> Optional[Mapping[str, object]]:
    candidates = [
        path.with_suffix(".meta.json"),
        Path(f"{path}.meta.json"),
        path.with_suffix(".json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
    return None


def _parse_event_list(raw: object) -> List[BMDAnimationEvent]:
    events: List[BMDAnimationEvent] = []
    if isinstance(raw, Sequence):
        for item in raw:
            if isinstance(item, Mapping):
                name = str(item.get("name") or item.get("event") or "")
                if not name:
                    continue
                time_value = float(item.get("time", 0.0) or 0.0)
                params_raw = item.get("params", [])
                params: List[float] = []
                if isinstance(params_raw, Sequence):
                    for value in params_raw[:4]:
                        try:
                            params.append(float(value))
                        except (TypeError, ValueError):
                            params.append(0.0)
                events.append(BMDAnimationEvent(name=name, time=time_value, params=tuple(params)))
            elif isinstance(item, str):
                if "@" in item:
                    name_part, _, time_part = item.partition("@")
                    try:
                        time_value = float(time_part)
                    except ValueError:
                        time_value = 0.0
                    events.append(BMDAnimationEvent(name=name_part.strip(), time=time_value))
                else:
                    events.append(BMDAnimationEvent(name=item.strip(), time=0.0))
    return sorted(events, key=lambda event: event.time)


def _apply_bmd_metadata(model: BMDModel, payload: Mapping[str, object]) -> None:
    bone_lookup: Dict[str, int] = {
        bone.name.lower(): index for index, bone in enumerate(model.bones)
    }

    attachments_raw = payload.get("attachments")
    if isinstance(attachments_raw, Sequence):
        for entry in attachments_raw:
            if not isinstance(entry, Mapping):
                continue
            bone_ref = entry.get("bone")
            bone_index: Optional[int] = None
            if isinstance(bone_ref, str):
                bone_index = bone_lookup.get(bone_ref.lower())
            elif isinstance(bone_ref, int):
                bone_index = bone_ref
            if bone_index is None or bone_index < 0 or bone_index >= len(model.bones):
                continue
            name = str(entry.get("name") or entry.get("id") or f"attachment_{len(model.attachments)}")
            offset_raw = entry.get("offset", (0.0, 0.0, 0.0))
            if isinstance(offset_raw, Sequence) and len(offset_raw) >= 3:
                offset = np.array([float(offset_raw[0]), float(offset_raw[1]), float(offset_raw[2])], dtype=np.float32)
            else:
                offset = np.zeros(3, dtype=np.float32)
            rotation_raw = entry.get("rotation_quat") or entry.get("rotation")
            if isinstance(rotation_raw, Sequence):
                values = list(rotation_raw)
                if len(values) >= 4:
                    quat = _quat_normalize(np.array(values[:4], dtype=np.float32))
                else:
                    quat = _quat_from_euler_deg(np.array(values[:3], dtype=np.float32))
            else:
                quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            color_raw = entry.get("color") or entry.get("colour")
            if isinstance(color_raw, Sequence) and len(color_raw) >= 3:
                try:
                    color = np.array(
                        [float(color_raw[0]), float(color_raw[1]), float(color_raw[2])],
                        dtype=np.float32,
                    )
                except (TypeError, ValueError):
                    color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            radius_raw = entry.get("radius") or entry.get("range") or entry.get("distance")
            try:
                radius = float(radius_raw) if radius_raw is not None else 1200.0
            except (TypeError, ValueError):
                radius = 1200.0
            velocity = np.zeros(3, dtype=np.float32)
            velocity_raw = entry.get("velocity") or entry.get("speed")
            if isinstance(velocity_raw, Sequence) and len(velocity_raw) >= 3:
                try:
                    velocity = np.array(
                        [float(velocity_raw[0]), float(velocity_raw[1]), float(velocity_raw[2])],
                        dtype=np.float32,
                    )
                except (TypeError, ValueError):
                    velocity = np.zeros(3, dtype=np.float32)
            size_raw = entry.get("size") or entry.get("scale")
            try:
                size = float(size_raw) if size_raw is not None else 1.0
            except (TypeError, ValueError):
                size = 1.0

            hardpoint = BMDHardpoint(
                name=name,
                bone_index=int(bone_index),
                offset=offset,
                rotation_quat=quat,
                is_billboard=bool(entry.get("billboard") or str(entry.get("type", "")).lower() == "billboard"),
                color=color,
                radius=radius,
                velocity=velocity,
                size=size,
            )
            if hardpoint.is_billboard:
                model.billboards.append(hardpoint)
            else:
                model.attachments.append(hardpoint)

    animations_meta = payload.get("animations")
    if isinstance(animations_meta, Mapping):
        for name, meta in animations_meta.items():
            if name not in model.animations:
                continue
            animation = model.animations[name]
            if isinstance(meta, Mapping):
                if "default_blend" in meta:
                    try:
                        animation.default_blend = float(meta["default_blend"])
                    except (TypeError, ValueError):
                        pass
                if "events" in meta:
                    animation.events = _parse_event_list(meta["events"])

    events_meta = payload.get("events")
    if isinstance(events_meta, Mapping):
        for name, events in events_meta.items():
            if name in model.animations:
                model.animations[name].events = _parse_event_list(events)

    default_events = payload.get("default_events")
    if isinstance(default_events, Sequence):
        for animation in model.animations.values():
            if not animation.events:
                animation.events = _parse_event_list(default_events)
class BMDLibrary:
    def __init__(
        self,
        search_roots: Sequence[Path],
        material_library: Optional[MaterialStateLibrary] = None,
    ) -> None:
        unique_roots = []
        for root in search_roots:
            if root and root.exists() and root.is_dir():
                resolved = root.resolve()
                if resolved not in unique_roots:
                    unique_roots.append(resolved)
        self.search_roots = unique_roots
        self._index: Dict[str, List[Path]] = defaultdict(list)
        self._cache: Dict[Path, BMDModel] = {}
        self._failures: Dict[str, str] = {}
        self.material_library = material_library
        self._build_index()

    def _build_index(self) -> None:
        for root in self.search_roots:
            for path in root.rglob("*.bmd"):
                lowered_name = path.stem.lower()
                lowered_full = path.name.lower()
                rel = path.relative_to(root).as_posix().lower()
                self._index[lowered_name].append(path)
                self._index[lowered_full].append(path)
                self._index[rel].append(path)
                parent = path.parent.name.lower()
                if parent.startswith("object"):
                    suffix = parent[len("object") :]
                    self._index[f"object{suffix}_{lowered_name}"].append(path)
                    digits = "".join(ch for ch in path.stem if ch.isdigit())
                    if digits:
                        self._index[f"object{suffix}_{digits}"].append(path)
                        self._index[digits].append(path)

    def _iter_candidates(self, obj: TerrainObject) -> Iterator[str]:
        if obj.type_name:
            lowered = obj.type_name.lower()
            yield lowered
            if lowered.startswith("model_"):
                trimmed = lowered[len("model_") :]
                yield trimmed
                yield trimmed.replace("_", "")
            yield lowered.replace("_", "")
        yield str(obj.type_id)
        yield f"object{obj.type_id}"
        yield f"object{obj.type_id:02d}"

    def resolve(self, obj: TerrainObject) -> Optional[Path]:
        for key in self._iter_candidates(obj):
            if key in self._index:
                for path in self._index[key]:
                    if path.exists():
                        return path
        return None

    def load(self, obj: TerrainObject) -> Optional[BMDModel]:
        path = self.resolve(obj)
        if path is None:
            return None
        try:
            if path not in self._cache:
                self._cache[path] = load_bmd_model(path, self.material_library)
            return self._cache[path]
        except Exception as exc:  # noqa: BLE001
            self._failures[str(path)] = str(exc)
            return None

    @property
    def failures(self) -> Mapping[str, str]:
        return dict(self._failures)

    def load_texture_image(self, base_name: str) -> Optional[np.ndarray]:
        if not base_name:
            return None
        for path in _iter_candidate_paths(self.search_roots, base_name, IMAGE_EXTENSIONS):
            image = _load_image_file(path)
            if image is not None:
                return image
        return None


def _upsample_height_map(matrix: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return matrix.astype(np.float32)
    height, width = matrix.shape
    if height != width:
        raise ValueError("Mapa de altura deve ser quadrado para upsampling uniforme.")
    size = (height - 1) * factor + 1
    src = np.arange(height, dtype=np.float32)
    dst = np.linspace(0.0, float(height - 1), size, dtype=np.float32)
    temp = np.empty((height, size), dtype=np.float32)
    for idx in range(height):
        temp[idx] = np.interp(dst, src, matrix[idx].astype(np.float32))
    result = np.empty((size, size), dtype=np.float32)
    for idx in range(size):
        result[:, idx] = np.interp(dst, src, temp[:, idx])
    return result


def _compose_model_matrix(obj: TerrainObject) -> np.ndarray:
    pitch, yaw, roll = (math.radians(angle) for angle in obj.angles)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float32)
    rotation = rz @ ry @ rx
    scale = obj.scale if obj.scale > 0 else 1.0
    model = np.eye(4, dtype=np.float32)
    model[:3, :3] = rotation @ np.diag([scale, scale, scale])
    model[:3, 3] = np.array(obj.position, dtype=np.float32)
    return model


class _TerrainBuffers:
    def __init__(
        self,
        ctx: "moderngl.Context",
        diffuse_program: "moderngl.Program",
        specular_program: Optional["moderngl.Program"],
        data: TerrainData,
        texture_library: TextureLibrary,
        overlay: str,
        material_library: Optional[MaterialStateLibrary],
    ) -> None:
        detail = texture_library.detail_factor
        heights = _upsample_height_map(data.height, detail)
        light_dir = texture_library.light_direction()
        tile_flags = compute_tile_material_flags(
            data.mapping_layer1,
            data.mapping_layer2,
            data.mapping_alpha,
            texture_library,
            material_library,
        )
        spacing = float(TERRAIN_SCALE) / max(1, detail)
        normal_map = _compute_normal_map(heights, spacing)
        shadow_mask = _compute_shadow_mask(
            heights,
            light_dir,
            spacing,
        )
        if detail > 1:
            expanded_flags = np.repeat(np.repeat(tile_flags, detail, axis=0), detail, axis=1)
            expanded_flags = np.pad(expanded_flags, ((0, 1), (0, 1)), mode="edge")
        else:
            expanded_flags = tile_flags
        grid_size = heights.shape[0]
        axis_coords = np.linspace(
            0.0,
            float((TERRAIN_SIZE - 1) * TERRAIN_SCALE),
            grid_size,
            dtype=np.float32,
        )
        positions = np.zeros((grid_size * grid_size, 3), dtype=np.float32)
        normals = np.zeros_like(positions)
        uvs = np.zeros((grid_size * grid_size, 2), dtype=np.float32)
        materials = np.zeros((grid_size * grid_size,), dtype=np.float32)
        for y in range(grid_size):
            for x in range(grid_size):
                idx = y * grid_size + x
                x_world = axis_coords[x]
                z_world = axis_coords[y]
                positions[idx] = (x_world, heights[y, x], z_world)
                uvs[idx] = (
                    x / float(grid_size - 1 if grid_size > 1 else 1),
                    y / float(grid_size - 1 if grid_size > 1 else 1),
                )
                materials[idx] = float(expanded_flags[min(y, expanded_flags.shape[0] - 1), min(x, expanded_flags.shape[1] - 1)])
                left = heights[y, max(x - 1, 0)]
                right = heights[y, min(x + 1, grid_size - 1)]
                down = heights[max(y - 1, 0), x]
                up = heights[min(y + 1, grid_size - 1), x]
                dx = (left - right) / (2.0 * spacing)
                dz = (down - up) / (2.0 * spacing)
                normal = np.array([dx, 1.0, dz], dtype=np.float32)
                normals[idx] = _normalize(normal)

        indices: List[int] = []
        for y in range(grid_size - 1):
            for x in range(grid_size - 1):
                i0 = y * grid_size + x
                i1 = i0 + 1
                i2 = i0 + grid_size
                i3 = i2 + 1
                indices.extend([i0, i2, i1, i1, i2, i3])

        vertex_data = np.hstack([
            positions,
            normals,
            uvs,
            materials[:, None],
        ]).astype("f4")
        self.vbo = ctx.buffer(vertex_data.tobytes())
        self.ibo = ctx.buffer(np.asarray(indices, dtype=np.uint32).tobytes())
        self.vao_diffuse = ctx.vertex_array(
            diffuse_program,
            [
                (self.vbo, "3f 3f 2f 1f", "in_position", "in_normal", "in_uv", "in_material"),
            ],
            self.ibo,
        )
        self.vao_specular = (
            ctx.vertex_array(
                specular_program,
                [
                    (self.vbo, "3f 3f 2f 1f", "in_position", "in_normal", "in_uv", "in_material"),
                ],
                self.ibo,
            )
            if specular_program is not None
            else None
        )

        if overlay == "textures":
            pixels = texture_library.compose_texture_pixels(
                data.mapping_layer1, data.mapping_layer2, data.mapping_alpha
            )
        else:
            matrix, cmap_name = _overlay_matrix(data, overlay)
            cmap = plt.get_cmap(cmap_name)
            normalized = _normalize_for_colormap(matrix)
            pixels = cmap(normalized)
        pixels = np.clip(pixels, 0.0, 1.0)
        height_px, width_px = pixels.shape[:2]
        texture_bytes = (pixels * 255).astype(np.uint8).tobytes()
        self.texture = ctx.texture((width_px, height_px), 4, texture_bytes)
        self.texture.build_mipmaps()
        self.texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.texture.repeat_x = True
        self.texture.repeat_y = True
        refined_heights = _bilinear_resize(data.height, detail)
        refined_normals = _compute_normal_map(refined_heights, float(TERRAIN_SCALE) / float(detail))
        if refined_normals.size == 0:
            lum_rgb = np.ones((height_px - 1, width_px - 1, 3), dtype=np.float32)
        else:
            luminosity = np.sum(refined_normals * light_dir.reshape(1, 1, 3), axis=2)
            luminosity = np.clip(luminosity + 0.5, 0.0, 1.0)
            lum_rgb = np.repeat(luminosity[:, :, None], 3, axis=2)
        light_rgb = np.ones((height_px, width_px, 3), dtype=np.float32)
        light_rgb[:-1, :-1, :] *= lum_rgb
        sampled_light_map = texture_library.light_map_texture(height_px)
        if sampled_light_map is not None:
            light_rgb *= np.clip(sampled_light_map[:, :, :3], 0.0, 1.0)
        light_bytes = (np.clip(light_rgb, 0.0, 1.0) * 255).astype(np.uint8).tobytes()
        self.light_texture = ctx.texture((light_rgb.shape[1], light_rgb.shape[0]), 3, light_bytes)
        self.light_texture.build_mipmaps()
        self.light_texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.light_texture.repeat_x = True
        self.light_texture.repeat_y = True
        normal_pixels = np.clip((normal_map * 0.5) + 0.5, 0.0, 1.0)
        normal_bytes = (normal_pixels * 255).astype(np.uint8).tobytes()
        self.normal_texture = ctx.texture(
            (normal_pixels.shape[1], normal_pixels.shape[0]), 3, normal_bytes
        )
        self.normal_texture.build_mipmaps()
        self.normal_texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.normal_texture.repeat_x = True
        self.normal_texture.repeat_y = True
        shadow_pixels = np.clip(shadow_mask, 0.0, 1.0)
        shadow_bytes = (shadow_pixels * 255).astype(np.uint8).tobytes()
        self.shadow_texture = ctx.texture((shadow_pixels.shape[1], shadow_pixels.shape[0]), 1, shadow_bytes)
        self.shadow_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.shadow_texture.repeat_x = True
        self.shadow_texture.repeat_y = True
        self.light_direction = light_dir


class _BMDMeshRenderer:
    def __init__(
        self,
        ctx: "moderngl.Context",
        diffuse_program: "moderngl.Program",
        specular_program: Optional["moderngl.Program"],
        mesh: BMDMesh,
        texture_loader: Optional[callable],
    ) -> None:
        self.base_positions = mesh.positions.astype(np.float32, copy=True)
        self.base_normals = mesh.normals.astype(np.float32, copy=True)
        self.base_texcoords = mesh.texcoords.astype(np.float32, copy=True)
        self.vbo: Optional["moderngl.Buffer"]
        self.ibo: Optional["moderngl.Buffer"]
        self.vao_diffuse: Optional["moderngl.VertexArray"]
        self.vao_specular: Optional["moderngl.VertexArray"]
        if self.base_positions.size == 0 or mesh.indices.size == 0:
            self.vbo = None
            self.ibo = None
            self.vao_diffuse = None
            self.vao_specular = None
        else:
            vertices = np.hstack([self.base_positions, self.base_normals, self.base_texcoords])
            self.vbo = ctx.buffer(vertices.astype("f4").tobytes())
            self.ibo = ctx.buffer(mesh.indices.astype(np.uint32).tobytes())
            self.vao_diffuse = ctx.vertex_array(
                diffuse_program,
                [
                    (self.vbo, "3f 3f 2f", "in_position", "in_normal", "in_uv"),
                ],
                self.ibo,
            )
            self.vao_specular = (
                ctx.vertex_array(
                    specular_program,
                    [
                        (self.vbo, "3f 3f 2f", "in_position", "in_normal", "in_uv"),
                    ],
                    self.ibo,
                )
                if specular_program is not None
                else None
            )
        self.material_flags = mesh.material_flags
        self.material_state = mesh.material_state
        self.texture = None
        self.normal_texture = None
        self.bone_indices = (
            mesh.bone_indices.astype(np.int32, copy=True) if mesh.bone_indices is not None else None
        )
        self.has_skinning = bool(
            self.bone_indices is not None and np.any(self.bone_indices >= 0) and self.vbo is not None
        )
        self._skinned_positions = self.base_positions.copy()
        self._skinned_normals = self.base_normals.copy()
        if texture_loader is not None and mesh.texture_name:
            image = texture_loader(mesh.texture_name)
            if image is not None:
                rgba = image.astype(np.uint8)
                if rgba.ndim == 2:
                    rgba = np.stack([rgba, rgba, rgba, np.full_like(rgba, 255)], axis=-1)
                elif rgba.shape[2] == 3:
                    alpha = np.full((rgba.shape[0], rgba.shape[1], 1), 255, dtype=np.uint8)
                    rgba = np.concatenate([rgba, alpha], axis=2)
                height_px, width_px = rgba.shape[:2]
                self.texture = ctx.texture((width_px, height_px), 4, rgba.tobytes())
                self.texture.build_mipmaps()
                self.texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                self.texture.repeat_x = True
                self.texture.repeat_y = True
        if texture_loader is not None and mesh.material_state.normal_map:
            image = texture_loader(mesh.material_state.normal_map)
            if image is not None:
                normal = image.astype(np.uint8)
                if normal.ndim == 2:
                    normal = np.stack([normal, normal, normal], axis=-1)
                if normal.shape[2] > 3:
                    normal = normal[..., :3]
                h_px, w_px = normal.shape[:2]
                self.normal_texture = ctx.texture((w_px, h_px), 3, normal.tobytes())
                self.normal_texture.build_mipmaps()
                self.normal_texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                self.normal_texture.repeat_x = True
                self.normal_texture.repeat_y = True

    def update_pose(self, bone_matrices: Sequence[np.ndarray]) -> None:
        if not self.has_skinning or not bone_matrices or self.vbo is None:
            return
        identity = np.eye(4, dtype=np.float32)
        for idx, bone_idx in enumerate(self.bone_indices):
            if bone_idx < 0 or bone_idx >= len(bone_matrices):
                transform = identity
            else:
                transform = bone_matrices[bone_idx]
            position = np.append(self.base_positions[idx], 1.0)
            transformed = transform @ position
            self._skinned_positions[idx] = transformed[:3]
            normal_matrix = transform[:3, :3]
            normal = normal_matrix @ self.base_normals[idx]
            self._skinned_normals[idx] = _normalize(normal)
        vertex_data = np.hstack(
            [
                self._skinned_positions.astype(np.float32, copy=False),
                self._skinned_normals.astype(np.float32, copy=False),
                self.base_texcoords.astype(np.float32, copy=False),
            ]
        )
        self.vbo.write(vertex_data.astype("f4").tobytes())

    @property
    def is_empty(self) -> bool:
        return self.vbo is None or self.ibo is None or self.vao_diffuse is None

    def apply_state(self, ctx: "moderngl.Context") -> None:
        if moderngl is None:
            return
        state = self.material_state
        if state.transparent or state.additive:
            ctx.enable(moderngl.BLEND)
            try:
                src = _blend_factor(state.src_blend)
                dst = _blend_factor(state.dst_blend if not state.additive else "one")
            except RuntimeError:
                src, dst = moderngl.SRC_ALPHA, moderngl.ONE
            ctx.blend_func = (src, dst)
        else:
            ctx.disable(moderngl.BLEND)
        _set_depth_mask(ctx, state.depth_write)
        if state.depth_test:
            ctx.enable(moderngl.DEPTH_TEST)
        else:
            ctx.disable(moderngl.DEPTH_TEST)
        if state.double_sided:
            ctx.disable(moderngl.CULL_FACE)
        else:
            ctx.enable(moderngl.CULL_FACE)


class BMDAnimationPlayer:
    def __init__(self, model: BMDModel, *, loop: bool = True) -> None:
        self.model = model
        self.loop = loop
        self.active: Optional[BMDAnimation] = None
        self.current_time = 0.0
        self._pose_cache: List[np.ndarray] = []
        self._dirty = True
        self._next_animation: Optional[BMDAnimation] = None
        self._next_time = 0.0
        self._blend_elapsed = 0.0
        self._blend_duration = 0.0
        self._event_queue: deque[BMDAnimationEvent] = deque()
        self._global_cache: List[np.ndarray] = []
        if model.animations:
            first = next(iter(model.animations))
            self.set_animation(first)

    def set_animation(
        self,
        name: Optional[str],
        *,
        blend_time: Optional[float] = None,
        restart: bool = True,
    ) -> None:
        if not name:
            self.active = None
            self.current_time = 0.0
            self._dirty = True
            return
        if name not in self.model.animations:
            return
        target = self.model.animations[name]
        blend_duration = target.default_blend if blend_time is None else blend_time
        if self.active is None or blend_duration <= 0.0:
            self.active = target
            if restart:
                self.current_time = 0.0
            self._next_animation = None
            self._blend_elapsed = 0.0
            self._blend_duration = 0.0
        else:
            if self.active is target and not restart:
                return
            self._next_animation = target
            self._next_time = 0.0
            self._blend_elapsed = 0.0
            self._blend_duration = max(0.0, blend_duration)
            if restart:
                self._next_time = 0.0
        if restart:
            self.current_time = 0.0
        self._dirty = True

    def _emit_events_interval(
        self,
        animation: BMDAnimation,
        start: float,
        end: float,
    ) -> None:
        if not animation.events:
            return
        if end < start:
            start, end = end, start
        for event in animation.events:
            if start < event.time <= end:
                self._event_queue.append(event)

    def update(self, delta: float) -> None:
        if self.active is None:
            return
        duration = self.active.duration
        prev_time = self.current_time
        if duration > 0.0:
            total = prev_time + delta
            if self.loop and duration > 0.0:
                wraps = int(total // duration)
                new_time = total % duration
                if wraps == 0:
                    self._emit_events_interval(self.active, prev_time, new_time)
                else:
                    self._emit_events_interval(self.active, prev_time, duration)
                    for _ in range(max(0, wraps - 1)):
                        self._emit_events_interval(self.active, 0.0, duration)
                    self._emit_events_interval(self.active, 0.0, new_time)
                self.current_time = new_time
            else:
                new_time = min(total, duration)
                self._emit_events_interval(self.active, prev_time, new_time)
                self.current_time = new_time
        else:
            self.current_time = 0.0

        if self._next_animation is not None:
            self._next_time += delta
            next_duration = self._next_animation.duration
            if next_duration > 0.0 and self.loop:
                self._next_time = math.fmod(self._next_time, next_duration)
            self._blend_elapsed += delta
            if self._blend_duration <= 0.0 or self._blend_elapsed >= self._blend_duration:
                self.active = self._next_animation
                self.current_time = self._next_time
                self._next_animation = None
                self._next_time = 0.0
                self._blend_elapsed = 0.0
                self._blend_duration = 0.0
        elif self._blend_duration > 0.0 and self._blend_elapsed > 0.0:
            self._blend_elapsed = min(self._blend_elapsed + delta, self._blend_duration)
        self._dirty = True

    def consume_events(self) -> List[BMDAnimationEvent]:
        events = list(self._event_queue)
        self._event_queue.clear()
        return events

    def pose_matrices(self) -> List[np.ndarray]:
        if not self._dirty and self._pose_cache:
            return self._pose_cache
        if not self.model.bones:
            self._pose_cache = []
            self._dirty = False
            return []
        animation = self.active
        if animation is None or not animation.channels:
            pose = [bone.rest_matrix.astype(np.float32) for bone in self.model.bones]
            self._global_cache = pose
            self._pose_cache = [pose_matrix @ bone.inverse_bind_matrix for pose_matrix, bone in zip(pose, self.model.bones)]
            self._dirty = False
            return self._pose_cache

        blend_factor = 0.0
        next_animation = self._next_animation
        next_time = self._next_time
        if next_animation is not None and self._blend_duration > 0.0:
            blend_factor = min(self._blend_elapsed / max(self._blend_duration, 1e-5), 1.0)

        local_matrices: List[np.ndarray] = []
        for index, bone in enumerate(self.model.bones):
            if index in animation.channels:
                trans_a, quat_a = self._sample_channel(
                    animation.channels[index], animation, self.current_time
                )
            else:
                trans_a, quat_a = bone.rest_translation, bone.rest_quaternion

            if blend_factor > 0.0 and next_animation is not None:
                if index in next_animation.channels:
                    trans_b, quat_b = self._sample_channel(
                        next_animation.channels[index], next_animation, next_time
                    )
                else:
                    trans_b, quat_b = bone.rest_translation, bone.rest_quaternion
                translation = _lerp_vec(trans_a, trans_b, blend_factor)
                quaternion = _quat_slerp(quat_a, quat_b, blend_factor)
            else:
                translation, quaternion = trans_a, quat_a
            local_matrices.append(_compose_transform_quaternion(translation, quaternion))

        global_matrices: List[np.ndarray] = []
        skinning: List[np.ndarray] = []
        for idx, local in enumerate(local_matrices):
            parent = self.model.bones[idx].parent
            if 0 <= parent < len(global_matrices):
                global_matrix = global_matrices[parent] @ local
            else:
                global_matrix = local
            global_matrices.append(global_matrix)
            skinning.append(global_matrix @ self.model.bones[idx].inverse_bind_matrix)

        self._global_cache = [matrix.astype(np.float32) for matrix in global_matrices]
        self._pose_cache = [matrix.astype(np.float32) for matrix in skinning]
        self._dirty = False
        return self._pose_cache

    def global_matrices(self) -> List[np.ndarray]:
        if self._global_cache:
            return [matrix.copy() for matrix in self._global_cache]
        return [bone.rest_matrix.astype(np.float32) for bone in self.model.bones]

    def _sample_channel(
        self,
        channel: BMDAnimationChannel,
        animation: BMDAnimation,
        time_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not channel.keyframes:
            zero = np.zeros(3, dtype=np.float32)
            return zero, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        if len(channel.keyframes) == 1:
            frame = channel.keyframes[0]
            return frame.translation, frame.rotation_quat
        duration = max(animation.duration, channel.keyframes[-1].time)
        if duration > 0.0 and self.loop:
            cycles = time_value / duration
            frac = cycles - math.floor(cycles)
            time_value = frac * duration
        prev = channel.keyframes[0]
        for frame in channel.keyframes[1:]:
            if time_value <= frame.time:
                span = frame.time - prev.time
                if span <= 0.0:
                    return frame.translation, frame.rotation_quat
                t = (time_value - prev.time) / span
                translation = _lerp_vec(prev.translation, frame.translation, t)
                quaternion = _quat_slerp(prev.rotation_quat, frame.rotation_quat, t)
                return translation, quaternion
            prev = frame
        last = channel.keyframes[-1]
        return last.translation, last.rotation_quat


@dataclass
class BMDInstance:
    model_matrix: np.ndarray
    mesh_renderers: List[_BMDMeshRenderer]
    animation_player: Optional[BMDAnimationPlayer] = None
    attachments: List[BMDHardpoint] = field(default_factory=list)
    billboards: List[BMDHardpoint] = field(default_factory=list)
    pose_matrices: List[np.ndarray] = field(default_factory=list)
    global_matrices: List[np.ndarray] = field(default_factory=list)


class OpenGLTerrainApp:
    def __init__(
        self,
        data: TerrainData,
        objects: Sequence[TerrainObject],
        *,
        texture_library: TextureLibrary,
        material_library: Optional[MaterialStateLibrary],
        bmd_library: Optional[BMDLibrary],
        overlay: str,
        title: str,
        fog_color: Tuple[float, float, float] = (0.25, 0.33, 0.45),
        fog_density: float = 0.00025,
        scene_focus: str = "terrain",
    ) -> None:
        if scene_focus not in {"terrain", "full"}:
            raise ValueError(f"Scene focus inválido: {scene_focus}")
        self.data = data
        self.objects = list(objects)
        self.texture_library = texture_library
        self.material_library = material_library
        self.bmd_library = bmd_library
        self.overlay = overlay
        self.title = title
        self.fog_color = np.array(fog_color, dtype=np.float32)
        self.fog_density = fog_density
        self.scene_focus = scene_focus
        self.focus_terrain_only = scene_focus == "terrain"
        self.window_size = (1280, 720)
        self.ctx: Optional["moderngl.Context"] = None
        self.window: Optional["pyglet.window.Window"] = None
        self.terrain: Optional[_TerrainBuffers] = None
        self.terrain_program: Optional["moderngl.Program"] = None
        self.terrain_specular_program: Optional["moderngl.Program"] = None
        self.object_program: Optional["moderngl.Program"] = None
        self.object_specular_program: Optional["moderngl.Program"] = None
        self.sky_program: Optional["moderngl.Program"] = None
        self.skybox_program: Optional["moderngl.Program"] = None
        self.particle_program: Optional["moderngl.Program"] = None
        self.sky_texture: Optional["moderngl.Texture"] = None
        self.camera: Optional[FreeCamera] = None
        self._pressed_keys: set[int] = set()
        self._start_time = 0.0
        self._last_frame_time = 0.0
        self._object_mesh_cache: Dict[str, List[_BMDMeshRenderer]] = {}
        self._particle_vbo: Optional["moderngl.Buffer"] = None
        self._particle_count = 0
        self._particle_vao: Optional["moderngl.VertexArray"] = None
        self._skybox_vao: Optional["moderngl.VertexArray"] = None
        self._sky_gradient_vao: Optional["moderngl.VertexArray"] = None
        self._sky_vbo: Optional["moderngl.Buffer"] = None
        self.object_instances: List[BMDInstance] = []
        self.directional_light_dir = texture_library.light_direction()
        self.directional_light_color = np.array([1.0, 0.96, 0.88], dtype=np.float32)
        self.shadow_strength = 0.65
        self.static_point_lights: List[Tuple[np.ndarray, np.ndarray, float]] = []
        self.dynamic_point_lights: List[Tuple[np.ndarray, np.ndarray, float]] = []
        self.emissive_override = np.zeros(3, dtype=np.float32)
        self._default_white_texture: Optional["moderngl.Texture"] = None
        self._default_normal_texture: Optional["moderngl.Texture"] = None
        self._mouse_captured = False

    def _init_programs(self) -> None:
        assert self.ctx is not None
        terrain_vertex_shader = textwrap.dedent(
            """
                #version 330
                in vec3 in_position;
                in vec3 in_normal;
                in vec2 in_uv;
                in float in_material;
                uniform mat4 u_model;
                uniform mat4 u_view;
                uniform mat4 u_projection;
                flat out int v_material;
                out vec3 v_normal;
                out vec3 v_world_pos;
                out vec2 v_uv;
                void main() {
                    vec4 world_pos = u_model * vec4(in_position, 1.0);
                    v_world_pos = world_pos.xyz;
                    mat3 normal_matrix = mat3(u_model);
                    v_normal = normalize(normal_matrix * in_normal);
                    v_uv = in_uv;
                    v_material = int(in_material + 0.5);
                    gl_Position = u_projection * u_view * world_pos;
                }
                """
        )
        terrain_fragment_shader = textwrap.dedent(
            """
                #version 330
                flat in int v_material;
                in vec3 v_normal;
                in vec3 v_world_pos;
                in vec2 v_uv;
                uniform sampler2D u_texture;
                uniform sampler2D u_normal_map;
                uniform sampler2D u_shadow_map;
                uniform sampler2D u_light_map;
                uniform vec3 u_dir_light_dir;
                uniform vec3 u_dir_light_color;
                uniform int u_point_light_count;
                uniform vec3 u_point_light_pos[__MAX_POINT_LIGHTS__];
                uniform vec3 u_point_light_color[__MAX_POINT_LIGHTS__];
                uniform float u_point_light_range[__MAX_POINT_LIGHTS__];
                uniform vec3 u_fog_color;
                uniform float u_fog_density;
                uniform float u_time;
                uniform vec3 u_camera_pos;
                uniform float u_shadow_strength;
                uniform vec3 u_emissive_override;
                out vec4 frag_color;
                const int FLAG_WATER = 1;
                const int FLAG_LAVA = 2;
                const int FLAG_TRANSPARENT = 4;
                const int FLAG_ADDITIVE = 8;
                const int FLAG_EMISSIVE = 16;
                const int FLAG_ALPHA_TEST = 32;
                const int FLAG_DOUBLE_SIDED = 64;
                const int FLAG_NO_SHADOW = 128;
                const int FLAG_NORMAL_MAP = 256;
                void main() {
                    vec2 uv = v_uv;
                    float wave = 0.0;
                    if ((v_material & FLAG_WATER) != 0) {
                        vec2 flow = vec2(sin(u_time * 0.35), cos(u_time * 0.4)) * 0.03;
                        uv += flow;
                        wave += sin(u_time * 1.6 + v_world_pos.x * 0.002 + v_world_pos.z * 0.002) * 0.08;
                    }
                    if ((v_material & FLAG_LAVA) != 0) {
                        float lava = sin(u_time * 2.0 + v_world_pos.x * 0.003) * 0.07;
                        uv += vec2(0.0, lava);
                        wave += sin(u_time * 3.1 + v_world_pos.z * 0.004) * 0.12;
                    }
                    vec4 tex = texture(u_texture, uv);
                    float alpha = tex.a;
                    if ((v_material & FLAG_ALPHA_TEST) != 0 && alpha < 0.35) {
                        discard;
                    }
                    vec3 static_light = texture(u_light_map, v_uv).rgb;
                    vec3 base_color = tex.rgb * static_light;
                    vec3 map_normal = texture(u_normal_map, v_uv).xyz * 2.0 - 1.0;
                    vec3 normal = normalize(v_normal);
                    if (wave != 0.0) {
                        normal = normalize(vec3(normal.x + wave, normal.y, normal.z + wave));
                    }
                    float normal_mix = ((v_material & FLAG_NORMAL_MAP) != 0) ? 0.7 : 0.45;
                    normal = normalize(mix(normal, map_normal, normal_mix));
                    vec3 lighting = base_color * 0.22;
                    vec3 dir = normalize(-u_dir_light_dir);
                    float diff = max(dot(normal, dir), 0.0);
                    float shadow = ((v_material & FLAG_NO_SHADOW) != 0) ? 1.0 : texture(u_shadow_map, v_uv).r;
                    lighting += base_color * diff * u_dir_light_color * mix(1.0, shadow, u_shadow_strength);
                    vec3 view_dir = normalize(u_camera_pos - v_world_pos);
                    vec3 half_dir = normalize(dir + view_dir);
                    float specular = pow(max(dot(normal, half_dir), 0.0), 30.0);
                    lighting += base_color * specular * 0.18;
                    for (int i = 0; i < u_point_light_count; ++i) {
                        vec3 to_light = u_point_light_pos[i] - v_world_pos;
                        float dist = length(to_light);
                        float range = max(u_point_light_range[i], 0.001);
                        float attenuation = clamp(1.0 - dist / range, 0.0, 1.0);
                        vec3 light_dir = normalize(to_light);
                        float ndotl = max(dot(normal, light_dir), 0.0);
                        vec3 contrib = base_color * ndotl * u_point_light_color[i] * attenuation * attenuation;
                        lighting += contrib;
                        vec3 half_vec = normalize(light_dir + view_dir);
                        float spec = pow(max(dot(normal, half_vec), 0.0), 28.0);
                        lighting += u_point_light_color[i] * spec * 0.05;
                    }
                    if ((v_material & FLAG_WATER) != 0) {
                        lighting *= vec3(0.9, 1.05, 1.1);
                    }
                    if ((v_material & FLAG_LAVA) != 0) {
                        lighting *= vec3(1.4, 0.6, 0.4);
                    }
                    if ((v_material & FLAG_EMISSIVE) != 0) {
                        lighting += base_color * 0.6;
                    }
                    lighting += u_emissive_override;
                    float distance = length(v_world_pos - u_camera_pos);
                    float fog = clamp(exp(-u_fog_density * distance), 0.0, 1.0);
                    vec3 final_color = mix(u_fog_color, lighting, fog);
                    if ((v_material & FLAG_ADDITIVE) != 0) {
                        final_color *= 1.2;
                        alpha = 1.0;
                    } else if ((v_material & FLAG_TRANSPARENT) == 0) {
                        alpha = 1.0;
                    }
                    frag_color = vec4(final_color, alpha);
                }
            """
        ).replace("__MAX_POINT_LIGHTS__", str(MAX_POINT_LIGHTS))
        self.terrain_program = self.ctx.program(
            vertex_shader=terrain_vertex_shader,
            fragment_shader=terrain_fragment_shader,
        )
        self.terrain_specular_program = None

        object_vertex_shader = textwrap.dedent(
            """
                #version 330
                in vec3 in_position;
                in vec3 in_normal;
                in vec2 in_uv;
                uniform mat4 u_model;
                uniform mat4 u_view;
                uniform mat4 u_projection;
                out vec3 v_normal;
                out vec3 v_world_pos;
                out vec2 v_uv;
                void main() {
                    vec4 world_pos = u_model * vec4(in_position, 1.0);
                    v_world_pos = world_pos.xyz;
                    mat3 normal_matrix = mat3(u_model);
                    v_normal = normalize(normal_matrix * in_normal);
                    v_uv = in_uv;
                    gl_Position = u_projection * u_view * world_pos;
                }
                """
        )
        object_fragment_shader = textwrap.dedent(
            """
                #version 330
                in vec3 v_normal;
                in vec3 v_world_pos;
                in vec2 v_uv;
                uniform sampler2D u_texture;
                uniform sampler2D u_normal_map;
                uniform bool u_has_normal_map;
                uniform vec3 u_dir_light_dir;
                uniform vec3 u_dir_light_color;
                uniform int u_point_light_count;
                uniform vec3 u_point_light_pos[__MAX_POINT_LIGHTS__];
                uniform vec3 u_point_light_color[__MAX_POINT_LIGHTS__];
                uniform float u_point_light_range[__MAX_POINT_LIGHTS__];
                uniform vec3 u_fog_color;
                uniform float u_fog_density;
                uniform vec3 u_camera_pos;
                uniform float u_time;
                uniform int u_material_flags;
                uniform vec3 u_material_emissive;
                uniform vec3 u_material_specular;
                uniform float u_specular_power;
                uniform float u_alpha_ref;
                out vec4 frag_color;
                const int FLAG_WATER = 1;
                const int FLAG_LAVA = 2;
                const int FLAG_TRANSPARENT = 4;
                const int FLAG_ADDITIVE = 8;
                const int FLAG_EMISSIVE = 16;
                const int FLAG_ALPHA_TEST = 32;
                const int FLAG_DOUBLE_SIDED = 64;
                const int FLAG_NO_SHADOW = 128;
                const int FLAG_NORMAL_MAP = 256;
                void main() {
                    vec2 uv = v_uv;
                    if ((u_material_flags & FLAG_WATER) != 0) {
                        uv += vec2(cos(u_time * 0.4), sin(u_time * 0.45)) * 0.03;
                    }
                    if ((u_material_flags & FLAG_LAVA) != 0) {
                        uv += vec2(0.0, sin(u_time * 2.4 + v_world_pos.x * 0.006) * 0.08);
                    }
                    vec4 tex = texture(u_texture, uv);
                    float alpha = tex.a;
                    if ((u_material_flags & FLAG_ALPHA_TEST) != 0 && alpha < u_alpha_ref) {
                        discard;
                    }
                    vec3 base_color = tex.rgb;
                    vec3 normal = normalize(v_normal);
                    if (u_has_normal_map) {
                        vec3 map_normal = texture(u_normal_map, v_uv).xyz * 2.0 - 1.0;
                        normal = normalize(map_normal);
                    }
                    vec3 lighting = base_color * 0.25;
                    vec3 dir = normalize(-u_dir_light_dir);
                    float diff = max(dot(normal, dir), 0.0);
                    lighting += base_color * diff * u_dir_light_color;
                    vec3 view_dir = normalize(u_camera_pos - v_world_pos);
                    vec3 half_dir = normalize(dir + view_dir);
                    float specular = pow(max(dot(normal, half_dir), 0.0), u_specular_power);
                    lighting += u_material_specular * specular;
                    for (int i = 0; i < u_point_light_count; ++i) {
                        vec3 to_light = u_point_light_pos[i] - v_world_pos;
                        float dist = length(to_light);
                        float range = max(u_point_light_range[i], 0.001);
                        float attenuation = clamp(1.0 - dist / range, 0.0, 1.0);
                        vec3 light_dir = normalize(to_light);
                        float ndotl = max(dot(normal, light_dir), 0.0);
                        vec3 contrib = base_color * ndotl * u_point_light_color[i] * attenuation * attenuation;
                        lighting += contrib;
                        vec3 half_vec = normalize(light_dir + view_dir);
                        float spec = pow(max(dot(normal, half_vec), 0.0), u_specular_power);
                        lighting += u_material_specular * spec * attenuation;
                    }
                    if ((u_material_flags & FLAG_EMISSIVE) != 0) {
                        lighting += base_color * 0.6;
                    }
                    lighting += u_material_emissive;
                    if ((u_material_flags & FLAG_WATER) != 0) {
                        lighting *= vec3(0.9, 1.05, 1.1);
                    }
                    if ((u_material_flags & FLAG_LAVA) != 0) {
                        lighting *= vec3(1.4, 0.7, 0.5);
                    }
                    float distance = length(v_world_pos - u_camera_pos);
                    float fog = clamp(exp(-u_fog_density * distance), 0.0, 1.0);
                    vec3 final_color = mix(u_fog_color, lighting, fog);
                    if ((u_material_flags & FLAG_ADDITIVE) != 0) {
                        final_color *= 1.2;
                        alpha = 1.0;
                    } else if ((u_material_flags & FLAG_TRANSPARENT) == 0) {
                        alpha = 1.0;
                    }
                    frag_color = vec4(final_color, clamp(alpha, 0.0, 1.0));
                }
            """
        ).replace("__MAX_POINT_LIGHTS__", str(MAX_POINT_LIGHTS))
        self.object_program = self.ctx.program(
            vertex_shader=object_vertex_shader,
            fragment_shader=object_fragment_shader,
        )
        self.object_specular_program = None

        if not self.focus_terrain_only:
            self.sky_program = self.ctx.program(
                vertex_shader=textwrap.dedent(
                    """
                    #version 330
                    in vec2 in_position;
                    out vec2 v_pos;
                    void main() {
                        v_pos = in_position;
                        gl_Position = vec4(in_position, 0.999, 1.0);
                    }
                    """
                ),
                fragment_shader=textwrap.dedent(
                    """
                    #version 330
                    in vec2 v_pos;
                    uniform vec3 u_color_top;
                    uniform vec3 u_color_bottom;
                    out vec4 frag_color;
                    void main() {
                        float t = clamp((v_pos.y + 1.0) * 0.5, 0.0, 1.0);
                        vec3 color = mix(u_color_bottom, u_color_top, pow(t, 1.6));
                        frag_color = vec4(color, 1.0);
                    }
                    """
                ),
            )

            self.skybox_program = self.ctx.program(
                vertex_shader=textwrap.dedent(
                    """
                    #version 330
                    in vec2 in_position;
                    out vec2 v_uv;
                    void main() {
                        v_uv = in_position * 0.5 + 0.5;
                        gl_Position = vec4(in_position, 0.999, 1.0);
                    }
                    """
                ),
                fragment_shader=textwrap.dedent(
                    """
                    #version 330
                    in vec2 v_uv;
                    uniform sampler2D u_sky;
                    out vec4 frag_color;
                    void main() {
                        vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
                        frag_color = texture(u_sky, uv);
                    }
                    """
                ),
            )

            if self._skybox_vao is not None:
                try:
                    self._skybox_vao.release()
                except Exception:  # noqa: BLE001
                    pass
            if self._sky_gradient_vao is not None:
                try:
                    self._sky_gradient_vao.release()
                except Exception:  # noqa: BLE001
                    pass
            if self._sky_vbo is not None:
                try:
                    self._sky_vbo.release()
                except Exception:  # noqa: BLE001
                    pass
            fullscreen_triangle = np.array(
                [
                    -1.0,
                    -1.0,
                    3.0,
                    -1.0,
                    -1.0,
                    3.0,
                ],
                dtype="f4",
            )
            self._sky_vbo = self.ctx.buffer(fullscreen_triangle.tobytes())
            self._skybox_vao = self.ctx.vertex_array(
                self.skybox_program, [(self._sky_vbo, "2f", "in_position")]
            )
            self._sky_gradient_vao = self.ctx.vertex_array(
                self.sky_program, [(self._sky_vbo, "2f", "in_position")]
            )

            self.particle_program = self.ctx.program(
                vertex_shader=textwrap.dedent(
                    """
                    #version 330
                    in vec3 in_position;
                    in vec3 in_velocity;
                    in float in_birth;
                    uniform mat4 u_view;
                    uniform mat4 u_projection;
                    uniform float u_time;
                    out float v_alpha;
                    void main() {
                        float age = u_time - in_birth;
                        vec3 pos = in_position + in_velocity * max(age, 0.0);
                        pos.y += sin(age * 0.8 + pos.x * 0.01) * 50.0;
                        gl_Position = u_projection * u_view * vec4(pos, 1.0);
                        gl_PointSize = clamp(6.0 - age * 0.4, 1.5, 6.0);
                        v_alpha = clamp(1.0 - age * 0.2, 0.0, 1.0);
                    }
                    """
                ),
                fragment_shader=textwrap.dedent(
                    """
                    #version 330
                    in float v_alpha;
                    uniform vec3 u_fog_color;
                    out vec4 frag_color;
                    void main() {
                        float t = clamp(v_alpha, 0.0, 1.0);
                        vec3 color = mix(u_fog_color, vec3(1.0, 0.85, 0.45), pow(t, 1.5));
                        frag_color = vec4(color, t);
                    }
                    """
                ),
            )
        else:
            self.sky_program = None
            self.skybox_program = None
            self.particle_program = None
            if self._skybox_vao is not None:
                try:
                    self._skybox_vao.release()
                except Exception:  # noqa: BLE001
                    pass
                self._skybox_vao = None
            if self._sky_gradient_vao is not None:
                try:
                    self._sky_gradient_vao.release()
                except Exception:  # noqa: BLE001
                    pass
                self._sky_gradient_vao = None
            if self._sky_vbo is not None:
                try:
                    self._sky_vbo.release()
                except Exception:  # noqa: BLE001
                    pass
                self._sky_vbo = None

    def _build_particles(self) -> None:
        if self.focus_terrain_only:
            self._particle_count = 0
            return
        assert self.ctx is not None
        if self._particle_vbo is not None:
            try:
                self._particle_vbo.release()
            except Exception:  # noqa: BLE001
                pass
        if self._particle_vao is not None:
            try:
                self._particle_vao.release()
            except Exception:  # noqa: BLE001
                pass
        self._particle_vbo = None
        self._particle_vao = None
        self._particle_count = 0
        self._update_dynamic_particles(0.0)

    def _find_sky_image(self) -> Optional[np.ndarray]:
        search_roots = getattr(self.texture_library, "search_roots", [])
        candidates: List[Path] = []
        for root in search_roots:
            if not root.exists():
                continue
            potential_dirs = [
                root,
                root / "Sky",
                root / "sky",
                root / "Texture",
                root / "Texture" / "Sky",
            ]
            for directory in potential_dirs:
                if not directory.exists() or not directory.is_dir():
                    continue
                for path in sorted(directory.iterdir()):
                    if not path.is_file():
                        continue
                    if path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    if "sky" in path.stem.lower():
                        candidates.append(path)
        for path in candidates:
            image = _load_image_file(path)
            if image is not None:
                return image
        return None

    def _initialize_sky_texture(self) -> None:
        if self.ctx is None or self.focus_terrain_only:
            self.sky_texture = None
            return
        image = self._find_sky_image()
        if image is None:
            self.sky_texture = None
            return
        rgba = _ensure_rgba(image)
        height_px, width_px = rgba.shape[:2]
        texture = self.ctx.texture((width_px, height_px), 4, rgba.tobytes())
        texture.build_mipmaps()
        texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        texture.repeat_x = True
        texture.repeat_y = True
        self.sky_texture = texture

    def _load_objects(self) -> List[BMDInstance]:
        assert self.ctx is not None
        if self.bmd_library is None:
            return []
        instances: List[BMDInstance] = []
        for obj in self.objects:
            model = self.bmd_library.load(obj)
            if not model or not model.meshes:
                continue
            cache_key = model.name
            if cache_key not in self._object_mesh_cache:
                renderers: List[_BMDMeshRenderer] = []
                for mesh in model.meshes:
                    renderer = _BMDMeshRenderer(
                        self.ctx,
                        self.object_program,
                        self.object_specular_program,
                        mesh,
                        self.bmd_library.load_texture_image if self.bmd_library else None,
                    )
                    if renderer.is_empty:
                        continue
                    renderers.append(renderer)
                self._object_mesh_cache[cache_key] = renderers
            render_meshes = [renderer for renderer in self._object_mesh_cache[cache_key] if not renderer.is_empty]
            if not render_meshes:
                continue
            animation_player = BMDAnimationPlayer(model) if model.animations else None
            if animation_player:
                bone_matrices = animation_player.pose_matrices()
                for renderer in render_meshes:
                    renderer.update_pose(bone_matrices)
            instance = BMDInstance(
                model_matrix=_compose_model_matrix(obj),
                mesh_renderers=render_meshes,
                animation_player=animation_player,
                attachments=list(model.attachments),
                billboards=list(model.billboards),
            )
            if animation_player:
                instance.pose_matrices = bone_matrices
                instance.global_matrices = animation_player.global_matrices()
            elif model.bones:
                instance.global_matrices = [bone.rest_matrix.astype(np.float32) for bone in model.bones]
            if self.focus_terrain_only:
                instance.attachments = []
                instance.billboards = []
            instances.append(instance)
        return instances

    def _setup(self) -> None:
        if moderngl is None or pyglet is None:
            raise RuntimeError(
                "Renderer OpenGL indisponível: é necessário ter 'moderngl' e 'pyglet' instalados."
            )

        if self.window is None:
            width, height = self.window_size
            last_error: Optional[Exception] = None
            created_window: Optional["pyglet.window.Window"] = None
            config_candidates: List[Optional["pyglet.gl.Config"]] = []
            if pyglet_gl is not None:
                for samples in (8, 4, 2, 0):
                    kwargs = {"double_buffer": True, "depth_size": 24}
                    if samples > 0:
                        kwargs.update({"sample_buffers": 1, "samples": samples})
                    try:
                        config_candidates.append(pyglet_gl.Config(**kwargs))
                    except Exception:  # noqa: BLE001
                        continue
            config_candidates.append(None)

            for config in config_candidates:
                try:
                    created_window = pyglet.window.Window(
                        width=width,
                        height=height,
                        caption=self.title,
                        resizable=True,
                        visible=False,
                        config=config,
                    )
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    continue
                else:
                    break

            if created_window is None:
                if last_error is None:
                    raise RuntimeError("Não foi possível criar a janela OpenGL.")
                raise RuntimeError("Não foi possível criar a janela OpenGL.") from last_error

            self.window = created_window
            try:
                framebuffer_size = self.window.get_framebuffer_size()
            except Exception:  # noqa: BLE001
                framebuffer_size = self.window_size
            if framebuffer_size and framebuffer_size[0] > 0 and framebuffer_size[1] > 0:
                self.window_size = (int(framebuffer_size[0]), int(framebuffer_size[1]))

        try:
            self.window.switch_to()
        except Exception:  # noqa: BLE001
            # Alguns drivers precisam que o contexto seja ativado explicitamente
            # antes da criação via moderngl. Se falhar aqui, permitimos que a
            # criação do contexto trate o erro normalmente.
            pass
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        white_pixel = bytes([255, 255, 255, 255])
        self._default_white_texture = self.ctx.texture((1, 1), 4, white_pixel)
        self._default_white_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._default_white_texture.repeat_x = False
        self._default_white_texture.repeat_y = False
        normal_pixel = bytes([127, 127, 255])
        self._default_normal_texture = self.ctx.texture((1, 1), 3, normal_pixel)
        self._default_normal_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._default_normal_texture.repeat_x = False
        self._default_normal_texture.repeat_y = False
        self._init_programs()
        self.terrain = _TerrainBuffers(
            self.ctx,
            self.terrain_program,
            self.terrain_specular_program,
            self.data,
            self.texture_library,
            self.overlay,
            self.material_library,
        )
        if self.terrain is not None:
            self.directional_light_dir = self.terrain.light_direction
        self._initialize_sky_texture()
        self.object_instances = self._load_objects()
        self._build_static_lights()
        self._build_particles()
        center = np.array(
            [
                (TERRAIN_SIZE - 1) * TERRAIN_SCALE / 2.0,
                float(np.max(self.data.height)) + 1000.0,
                (TERRAIN_SIZE - 1) * TERRAIN_SCALE / 2.0,
            ],
            dtype=np.float32,
        )
        extent = (TERRAIN_SIZE - 1) * TERRAIN_SCALE
        start = center + np.array([-0.25 * extent, 0.2 * extent, 0.28 * extent], dtype=np.float32)
        self.camera = FreeCamera.from_look_at(start, center)
        self._start_time = time.perf_counter()
        self._last_frame_time = self._start_time
        self.window.set_visible(True)
        self._bind_events()
        print(
            "Controles: W/A/S/D movem, Q/E ou Ctrl/Espaço ajustam altitude,"
            " Shift acelera. Segure o botão direito do mouse para mirar livremente e"
            " use a rolagem para ajustar a velocidade de voo."
        )

    def _bind_events(self) -> None:
        assert self.window is not None

        @self.window.event
        def on_draw() -> None:  # noqa: ANN001
            self.render_frame()

        @self.window.event
        def on_close() -> None:  # noqa: ANN001
            if self._mouse_captured:
                try:
                    self.window.set_exclusive_mouse(False)
                except Exception:  # noqa: BLE001
                    pass
                self._mouse_captured = False
            pyglet.app.exit()

        @self.window.event
        def on_deactivate() -> None:  # noqa: ANN001
            if self._mouse_captured:
                try:
                    self.window.set_exclusive_mouse(False)
                except Exception:  # noqa: BLE001
                    pass
                self._mouse_captured = False

        @self.window.event
        def on_key_press(symbol: int, _modifiers: int) -> None:
            if symbol == pyglet_key.ESCAPE:
                if self._mouse_captured:
                    try:
                        self.window.set_exclusive_mouse(False)
                    except Exception:  # noqa: BLE001
                        pass
                    self._mouse_captured = False
                pyglet.app.exit()
                return
            self._pressed_keys.add(symbol)

        @self.window.event
        def on_key_release(symbol: int, _modifiers: int) -> None:
            if symbol in self._pressed_keys:
                self._pressed_keys.remove(symbol)

        if pyglet_mouse is not None:

            @self.window.event
            def on_mouse_press(_x: int, _y: int, button: int, _modifiers: int) -> None:
                if button == pyglet_mouse.RIGHT:
                    try:
                        self.window.set_exclusive_mouse(True)
                    except Exception:  # noqa: BLE001
                        return
                    self._mouse_captured = True

            @self.window.event
            def on_mouse_release(_x: int, _y: int, button: int, _modifiers: int) -> None:
                if button == pyglet_mouse.RIGHT and self._mouse_captured:
                    try:
                        self.window.set_exclusive_mouse(False)
                    except Exception:  # noqa: BLE001
                        pass
                    self._mouse_captured = False

        @self.window.event
        def on_mouse_motion(_x: int, _y: int, dx: float, dy: float) -> None:
            if self._mouse_captured and self.camera:
                self.camera.look(dx, dy)

        @self.window.event
        def on_mouse_drag(_x: int, _y: int, dx: float, dy: float, _buttons: int, _modifiers: int) -> None:
            if self._mouse_captured and self.camera:
                self.camera.look(dx, dy)

        @self.window.event
        def on_mouse_scroll(_x: int, _y: int, _dx: float, dy: float) -> None:
            if self.camera:
                self.camera.zoom(-dy * 400.0)

        def _tick(_dt: float) -> None:
            if self.window is not None:
                self.window.invalid = True

        pyglet.clock.schedule_interval(_tick, 1 / 120.0)

    def _render_sky(self, time_value: float) -> None:
        if self.focus_terrain_only:
            return
        assert self.ctx is not None
        self.ctx.disable(moderngl.DEPTH_TEST)
        previous_depth_mask = _get_depth_mask(self.ctx)
        if previous_depth_mask is not None:
            _set_depth_mask(self.ctx, False)
        self.ctx.screen.use()
        if (
            self.sky_texture is not None
            and self.skybox_program is not None
            and self._skybox_vao is not None
        ):
            self.sky_texture.use(location=3)
            self.skybox_program["u_sky"].value = 3
            self._skybox_vao.render(mode=moderngl.TRIANGLES, vertices=3)
        if self.sky_program is not None and self._sky_gradient_vao is not None:
            cycle = (math.sin(time_value * 0.05) + 1.0) * 0.5
            day_top = np.array([0.32, 0.45, 0.72], dtype=np.float32)
            night_top = np.array([0.05, 0.08, 0.18], dtype=np.float32)
            top = day_top * (1.0 - cycle) + night_top * cycle
            bottom = self.fog_color * (0.7 + 0.3 * cycle)
            self.sky_program["u_color_top"].value = tuple(np.clip(top, 0.0, 1.0).tolist())
            self.sky_program["u_color_bottom"].value = tuple(np.clip(bottom, 0.0, 1.0).tolist())
            self._sky_gradient_vao.render(mode=moderngl.TRIANGLES, vertices=3)
        if previous_depth_mask is not None:
            _set_depth_mask(self.ctx, previous_depth_mask)
        self.ctx.enable(moderngl.DEPTH_TEST)

    def _render_particles(self, view: np.ndarray, projection: np.ndarray, time_value: float) -> None:
        if (
            self.focus_terrain_only
            or self._particle_vbo is None
            or self._particle_count == 0
            or self._particle_vao is None
        ):
            return
        assert self.ctx is not None and self.particle_program is not None
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.particle_program["u_view"].write(view.astype("f4").tobytes())
        self.particle_program["u_projection"].write(projection.astype("f4").tobytes())
        self.particle_program["u_time"].value = time_value
        self.particle_program["u_fog_color"].value = tuple(self.fog_color.tolist())
        self._particle_vao.render(mode=moderngl.POINTS, vertices=self._particle_count)
        self.ctx.disable(moderngl.BLEND)

    def render_frame(self) -> None:
        if self.ctx is None or self.camera is None or self.terrain_program is None or self.object_program is None:
            return
        current_time = time.perf_counter()
        if self._last_frame_time == 0.0:
            delta_time = 1.0 / 60.0
        else:
            delta_time = current_time - self._last_frame_time
        delta_time = min(delta_time, 0.25)
        self._last_frame_time = current_time
        time_value = current_time - self._start_time
        if self.window is not None:
            width, height = self.window.get_framebuffer_size()
        else:
            width, height = self.window_size
        aspect = width / float(max(height, 1))
        projection = _perspective_matrix(60.0, aspect, 10.0, 60000.0)
        self.camera.update(self._pressed_keys, delta_time)
        view = self.camera.view_matrix()
        eye = self.camera.position
        for instance in self.object_instances:
            if instance.animation_player is not None:
                instance.animation_player.update(delta_time)
                bone_matrices = instance.animation_player.pose_matrices()
                instance.pose_matrices = bone_matrices
                instance.global_matrices = instance.animation_player.global_matrices()
                for renderer in instance.mesh_renderers:
                    renderer.update_pose(bone_matrices)
            elif instance.mesh_renderers and instance.global_matrices:
                pose = instance.global_matrices
                for renderer in instance.mesh_renderers:
                    renderer.update_pose(pose)
        if self.focus_terrain_only:
            self.dynamic_point_lights = []
            self._particle_count = 0
        else:
            self.dynamic_point_lights = self._collect_dynamic_lights()
            self._update_dynamic_particles(time_value)
        point_lights = (self.dynamic_point_lights + self.static_point_lights)[:MAX_POINT_LIGHTS]
        point_count = len(point_lights)
        point_positions = np.zeros((MAX_POINT_LIGHTS, 3), dtype=np.float32)
        point_colors = np.zeros((MAX_POINT_LIGHTS, 3), dtype=np.float32)
        point_ranges = np.zeros((MAX_POINT_LIGHTS,), dtype=np.float32)
        for idx, (position, color, radius) in enumerate(point_lights):
            point_positions[idx] = position
            point_colors[idx] = color
            point_ranges[idx] = radius
        dir_light = _normalize(self.directional_light_dir)
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.screen.clear(*self.fog_color.tolist(), 1.0)
        if not self.focus_terrain_only:
            self._render_sky(time_value)

        terrain = self.terrain
        assert terrain is not None
        terrain.texture.use(location=0)
        if terrain.normal_texture is not None:
            terrain.normal_texture.use(location=1)
        if terrain.shadow_texture is not None:
            terrain.shadow_texture.use(location=2)
        if terrain.light_texture is not None:
            terrain.light_texture.use(location=3)
        elif self._default_white_texture is not None:
            self._default_white_texture.use(location=3)
        model_identity = np.eye(4, dtype=np.float32)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.terrain_program["u_model"].write(model_identity.tobytes())
        self.terrain_program["u_view"].write(view.astype("f4").tobytes())
        self.terrain_program["u_projection"].write(projection.astype("f4").tobytes())
        self.terrain_program["u_texture"].value = 0
        if terrain.normal_texture is not None:
            self.terrain_program["u_normal_map"].value = 1
        if terrain.shadow_texture is not None:
            self.terrain_program["u_shadow_map"].value = 2
        self.terrain_program["u_light_map"].value = 3
        self.terrain_program["u_dir_light_dir"].value = tuple(dir_light.tolist())
        self.terrain_program["u_dir_light_color"].value = tuple(self.directional_light_color.tolist())
        self.terrain_program["u_point_light_count"].value = point_count
        self.terrain_program["u_point_light_pos"].write(point_positions.astype("f4").tobytes())
        self.terrain_program["u_point_light_color"].write(point_colors.astype("f4").tobytes())
        self.terrain_program["u_point_light_range"].write(point_ranges.astype("f4").tobytes())
        self.terrain_program["u_fog_color"].value = tuple(self.fog_color.tolist())
        self.terrain_program["u_fog_density"].value = self.fog_density
        self.terrain_program["u_time"].value = time_value
        self.terrain_program["u_camera_pos"].value = tuple(eye.tolist())
        self.terrain_program["u_shadow_strength"].value = self.shadow_strength
        self.terrain_program["u_emissive_override"].value = tuple(self.emissive_override.tolist())
        if terrain.vao_diffuse is not None:
            terrain.vao_diffuse.render()

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.object_program["u_view"].write(view.astype("f4").tobytes())
        self.object_program["u_projection"].write(projection.astype("f4").tobytes())
        self.object_program["u_dir_light_dir"].value = tuple(dir_light.tolist())
        self.object_program["u_dir_light_color"].value = tuple(self.directional_light_color.tolist())
        self.object_program["u_point_light_count"].value = point_count
        self.object_program["u_point_light_pos"].write(point_positions.astype("f4").tobytes())
        self.object_program["u_point_light_color"].write(point_colors.astype("f4").tobytes())
        self.object_program["u_point_light_range"].write(point_ranges.astype("f4").tobytes())
        self.object_program["u_fog_color"].value = tuple(self.fog_color.tolist())
        self.object_program["u_fog_density"].value = self.fog_density
        self.object_program["u_time"].value = time_value
        self.object_program["u_camera_pos"].value = tuple(eye.tolist())
        self.object_program["u_texture"].value = 0
        self.object_program["u_normal_map"].value = 1
        for instance in self.object_instances:
            self.object_program["u_model"].write(instance.model_matrix.astype(np.float32).tobytes())
            for mesh in instance.mesh_renderers:
                mesh.apply_state(self.ctx)
                texture = mesh.texture or self._default_white_texture
                normal_map = mesh.normal_texture or self._default_normal_texture
                if texture is not None:
                    texture.use(location=0)
                if normal_map is not None:
                    normal_map.use(location=1)
                self.object_program["u_has_normal_map"].value = int(mesh.normal_texture is not None)
                self.object_program["u_material_flags"].value = mesh.material_flags
                emissive = np.clip(np.array(mesh.material_state.emissive, dtype=np.float32), 0.0, 4.0)
                specular = np.array(mesh.material_state.specular, dtype=np.float32)
                self.object_program["u_material_emissive"].value = tuple(emissive.tolist())
                self.object_program["u_material_specular"].value = tuple(specular.tolist())
                self.object_program["u_specular_power"].value = float(mesh.material_state.specular_power)
                alpha_ref = float(mesh.material_state.alpha_ref if mesh.material_state.alpha_test else 0.0)
                self.object_program["u_alpha_ref"].value = alpha_ref
                if mesh.vao_diffuse is None:
                    continue
                mesh.vao_diffuse.render()
        self.ctx.disable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        _set_depth_mask(self.ctx, True)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

        if not self.focus_terrain_only:
            self._render_particles(view, projection, time_value)

        if self.window is not None:
            self.window.flip()

    def save_framebuffer(self, destination: Path) -> None:
        if self.ctx is None:
            return
        if self.window is not None:
            width, height = self.window.get_framebuffer_size()
        else:
            width, height = self.window_size
        buffer = self.ctx.screen.read(components=4)
        image = Image.frombytes("RGBA", (width, height), buffer)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image.save(destination)

    def run(self, *, show: bool, output: Optional[Path]) -> None:
        try:
            self._setup()
        except Exception:
            if self.window is not None:
                try:
                    self.window.close()
                except Exception:  # noqa: BLE001
                    pass
            raise
        assert self.window is not None
        if not show and output is not None:
            self.render_frame()
            self.save_framebuffer(output)
            self.window.close()
            return
        pyglet.app.run()
        if output is not None:
            self.save_framebuffer(output)


def bux_convert(data: bytearray) -> None:
    for idx in range(len(data)):
        data[idx] ^= BUX_CODE[idx % len(BUX_CODE)]


def _read_file(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Arquivo não encontrado: {path}") from exc


def open_terrain_attribute(path: Path) -> Tuple[int, np.ndarray]:
    raw = _read_file(path)
    decrypted = bytearray(map_file_decrypt(raw))
    bux_convert(decrypted)

    if len(decrypted) not in (131_076, 65_540):
        raise ValueError(
            "Tamanho inesperado para arquivo de atributos."
            f" Esperado 65540 ou 131076 bytes, recebi {len(decrypted)}."
        )

    version = decrypted[0]
    map_id = decrypted[1]
    width = decrypted[2]
    height = decrypted[3]
    if version != 0 or width != 255 or height != 255:
        raise ValueError(
            "Cabeçalho de atributos inválido (versão, largura ou altura)."
        )

    offset = 4
    if len(decrypted) == 65_540:
        data = np.frombuffer(decrypted, dtype=np.uint8, count=TERRAIN_SIZE * TERRAIN_SIZE, offset=offset)
        attributes = data.astype(np.uint16)
    else:
        data = np.frombuffer(decrypted, dtype=np.uint16, count=TERRAIN_SIZE * TERRAIN_SIZE, offset=offset)
        attributes = data.copy()
    return int(map_id), attributes.reshape((TERRAIN_SIZE, TERRAIN_SIZE))


def open_terrain_mapping(path: Path) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    raw = _read_file(path)
    decrypted = map_file_decrypt(raw)
    ptr = 0
    ptr += 1  # versão
    map_id = decrypted[ptr]
    ptr += 1  # número do mapa

    layer_count = TERRAIN_SIZE * TERRAIN_SIZE
    layer1 = np.frombuffer(decrypted, dtype=np.uint8, count=layer_count, offset=ptr)
    ptr += layer_count
    layer2 = np.frombuffer(decrypted, dtype=np.uint8, count=layer_count, offset=ptr)
    ptr += layer_count
    alpha_bytes = decrypted[ptr : ptr + layer_count]
    alpha = np.frombuffer(alpha_bytes, dtype=np.uint8).astype(np.float32) / 255.0
    return (
        int(map_id),
        layer1.reshape((TERRAIN_SIZE, TERRAIN_SIZE)),
        layer2.reshape((TERRAIN_SIZE, TERRAIN_SIZE)),
        alpha.reshape((TERRAIN_SIZE, TERRAIN_SIZE)),
    )


def load_height_file(path: Path, *, extended: bool = False, scale_override: Optional[float] = None) -> np.ndarray:
    raw = _read_file(path)
    if len(raw) < 4:
        raise ValueError("Arquivo de altura muito pequeno.")
    payload = raw[4:]

    if not extended:
        if len(payload) < 1080 + TERRAIN_SIZE * TERRAIN_SIZE:
            raise ValueError("Arquivo de altura (formato clássico) truncado.")
        height_bytes = payload[1080 : 1080 + TERRAIN_SIZE * TERRAIN_SIZE]
        heights = np.frombuffer(height_bytes, dtype=np.uint8).astype(np.float32)
        scale = scale_override if scale_override is not None else 1.5
        heights *= scale
    else:
        header_size = 14 + 40
        if len(payload) < header_size + TERRAIN_SIZE * TERRAIN_SIZE * 3:
            raise ValueError("Arquivo de altura (formato estendido) truncado.")
        pixel_data = payload[header_size : header_size + TERRAIN_SIZE * TERRAIN_SIZE * 3]
        heights = np.empty(TERRAIN_SIZE * TERRAIN_SIZE, dtype=np.float32)
        for idx in range(TERRAIN_SIZE * TERRAIN_SIZE):
            b = pixel_data[idx * 3 + 0]
            g = pixel_data[idx * 3 + 1]
            r = pixel_data[idx * 3 + 2]
            value = (r << 16) | (g << 8) | b
            heights[idx] = float(value) + G_MIN_HEIGHT
    return heights.reshape((TERRAIN_SIZE, TERRAIN_SIZE))


def open_objects_enc(
    path: Path, model_names: Mapping[int, str]
) -> Tuple[int, int, List[TerrainObject]]:
    raw = _read_file(path)
    decrypted = map_file_decrypt(raw)
    ptr = 0
    version = decrypted[ptr]
    ptr += 1  # versão
    map_id = decrypted[ptr]
    ptr += 1  # número do mapa
    count = struct.unpack_from("<h", decrypted, ptr)[0]
    ptr += 2
    objects: List[TerrainObject] = []
    for _ in range(count):
        type_id = struct.unpack_from("<h", decrypted, ptr)[0]
        ptr += 2
        position = struct.unpack_from("<3f", decrypted, ptr)
        ptr += 12
        angles = struct.unpack_from("<3f", decrypted, ptr)
        ptr += 12
        scale = struct.unpack_from("<f", decrypted, ptr)[0]
        ptr += 4
        objects.append(
            TerrainObject(
                type_id=type_id,
                position=(position[0], position[1], position[2]),
                angles=(angles[0], angles[1], angles[2]),
                scale=scale,
                type_name=model_names.get(type_id),
            )
        )
    return int(map_id), int(version), objects


def bilinear_height(height: np.ndarray, x: float, y: float) -> float:
    xi = int(math.floor(x))
    yi = int(math.floor(y))
    if xi < 0 or yi < 0 or xi >= TERRAIN_SIZE - 1 or yi >= TERRAIN_SIZE - 1:
        return float(height[min(max(yi, 0), TERRAIN_SIZE - 1), min(max(xi, 0), TERRAIN_SIZE - 1)])
    xd = x - xi
    yd = y - yi
    h1 = height[yi, xi] * (1 - yd) + height[yi + 1, xi] * yd
    h2 = height[yi, xi + 1] * (1 - yd) + height[yi + 1, xi + 1] * yd
    return h1 * (1 - xd) + h2 * xd


def _perspective_matrix(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = f / aspect
    matrix[1, 1] = f
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    fwd = _normalize(target - eye)
    side = _normalize(np.cross(fwd, up))
    up_vec = np.cross(side, fwd)
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, :3] = side
    matrix[1, :3] = up_vec
    matrix[2, :3] = -fwd
    matrix[:3, 3] = -matrix[:3, :3] @ eye
    return matrix


class FreeCamera:
    def __init__(
        self,
        position: np.ndarray,
        *,
        yaw: float = math.radians(135.0),
        pitch: float = math.radians(-20.0),
        move_speed: float = 1200.0,
        sprint_multiplier: float = 4.0,
        mouse_sensitivity: float = 0.0025,
    ) -> None:
        self._position = position.astype(np.float32)
        self.yaw = yaw
        self.pitch = pitch
        self.move_speed = move_speed
        self.sprint_multiplier = sprint_multiplier
        self.mouse_sensitivity = mouse_sensitivity
        self.min_speed = 100.0
        self.max_speed = 20000.0
        self.max_pitch = math.radians(89.0)
        self.velocity = np.zeros(3, dtype=np.float32)
        self._acceleration = 12.0
        self._damping = 0.15

    @property
    def position(self) -> np.ndarray:
        return self._position

    @classmethod
    def from_look_at(
        cls,
        position: np.ndarray,
        target: np.ndarray,
        **kwargs: object,
    ) -> "FreeCamera":
        position = position.astype(np.float32)
        target = target.astype(np.float32)
        forward = _normalize(target - position)
        yaw = math.atan2(forward[2], forward[0])
        pitch = math.asin(float(np.clip(forward[1], -0.99999, 0.99999)))
        return cls(position, yaw=yaw, pitch=pitch, **kwargs)

    def direction(self) -> np.ndarray:
        cos_pitch = math.cos(self.pitch)
        sin_pitch = math.sin(self.pitch)
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        return np.array(
            [
                cos_pitch * cos_yaw,
                sin_pitch,
                cos_pitch * sin_yaw,
            ],
            dtype=np.float32,
        )

    def look(self, dx: float, dy: float) -> None:
        self.yaw += dx * self.mouse_sensitivity
        self.pitch -= dy * self.mouse_sensitivity
        self.pitch = float(np.clip(self.pitch, -self.max_pitch, self.max_pitch))

    def adjust_speed(self, factor: float) -> None:
        self.move_speed = float(np.clip(self.move_speed * factor, self.min_speed, self.max_speed))

    def zoom(self, delta: float) -> None:
        if delta == 0.0:
            return
        factor = 1.0 + abs(delta) / 4000.0
        if delta > 0:
            self.adjust_speed(factor)
        else:
            self.adjust_speed(1.0 / factor)

    def update(self, pressed: Sequence[int], dt: float) -> None:
        if dt <= 0.0:
            return
        forward = _normalize(self.direction())
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-5:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right = _normalize(right)
        move = np.zeros(3, dtype=np.float32)
        if pyglet_key:
            if pyglet_key.W in pressed:
                move += forward
            if pyglet_key.S in pressed:
                move -= forward
            if pyglet_key.A in pressed:
                move -= right
            if pyglet_key.D in pressed:
                move += right
            if pyglet_key.Q in pressed or getattr(pyglet_key, "LCTRL", None) in pressed:
                move -= up
            if pyglet_key.E in pressed or getattr(pyglet_key, "SPACE", None) in pressed:
                move += up
        if np.linalg.norm(move) > 0.0:
            move = _normalize(move)
        target_velocity = move * self.move_speed
        if pyglet_key:
            if getattr(pyglet_key, "LSHIFT", None) in pressed or getattr(pyglet_key, "RSHIFT", None) in pressed:
                target_velocity *= self.sprint_multiplier
        lerp = float(np.clip(self._acceleration * dt, 0.0, 1.0))
        self.velocity = self.velocity * (1.0 - lerp) + target_velocity * lerp
        if np.linalg.norm(move) == 0.0:
            self.velocity *= max(0.0, 1.0 - self._damping)
        self._position += self.velocity * dt

    def view_matrix(self) -> np.ndarray:
        target = self.position + self.direction()
        return _look_at(self.position, target, np.array([0.0, 1.0, 0.0], dtype=np.float32))


class OrbitCamera:
    def __init__(self, target: np.ndarray, *, distance: float = 9000.0) -> None:
        self.target = target.astype(np.float32)
        self.distance = distance
        self.yaw = math.radians(135.0)
        self.pitch = math.radians(45.0)
        self.min_pitch = math.radians(5.0)
        self.max_pitch = math.radians(85.0)
        self.min_distance = 1000.0
        self.max_distance = 30000.0
        self.move_speed = 600.0
        self.orbit_speed = math.radians(60.0)
        self.zoom_speed = 2000.0
        self.velocity = np.zeros(3, dtype=np.float32)

    @property
    def position(self) -> np.ndarray:
        cos_pitch = math.cos(self.pitch)
        sin_pitch = math.sin(self.pitch)
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        direction = np.array(
            [
                cos_pitch * cos_yaw,
                sin_pitch,
                cos_pitch * sin_yaw,
            ],
            dtype=np.float32,
        )
        return self.target - direction * self.distance

    def orbit(self, delta_yaw: float, delta_pitch: float) -> None:
        self.yaw += delta_yaw
        self.pitch = float(np.clip(self.pitch + delta_pitch, self.min_pitch, self.max_pitch))

    def zoom(self, delta: float) -> None:
        self.distance = float(np.clip(self.distance + delta, self.min_distance, self.max_distance))

    def pan(self, offset: np.ndarray) -> None:
        self.target += offset

    def update(self, pressed: Sequence[int], dt: float) -> None:
        forward = _normalize(self.target - self.position)
        forward[1] = 0.0
        forward = _normalize(forward)
        right = _normalize(np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=np.float32)))
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        move = np.zeros(3, dtype=np.float32)
        if pyglet_key:
            if pyglet_key.W in pressed:
                move += forward
            if pyglet_key.S in pressed:
                move -= forward
            if pyglet_key.A in pressed:
                move -= right
            if pyglet_key.D in pressed:
                move += right
            if pyglet_key.Q in pressed:
                move -= up
            if pyglet_key.E in pressed:
                move += up
            if pyglet_key.LEFT in pressed:
                self.orbit(self.orbit_speed * dt, 0.0)
            if pyglet_key.RIGHT in pressed:
                self.orbit(-self.orbit_speed * dt, 0.0)
            if pyglet_key.UP in pressed:
                self.orbit(0.0, self.orbit_speed * dt)
            if pyglet_key.DOWN in pressed:
                self.orbit(0.0, -self.orbit_speed * dt)
            if pyglet_key.Z in pressed:
                self.zoom(-self.zoom_speed * dt)
            if pyglet_key.X in pressed:
                self.zoom(self.zoom_speed * dt)
        if np.linalg.norm(move) > 0.0:
            move = _normalize(move) * self.move_speed * dt
            self.pan(move)

    def view_matrix(self) -> np.ndarray:
        return _look_at(self.position, self.target, np.array([0.0, 1.0, 0.0], dtype=np.float32))

def _split_filter_values(values: Optional[Sequence[str]]) -> List[str]:
    tokens: List[str] = []
    if not values:
        return tokens
    for value in values:
        if value is None:
            continue
        for piece in re.split(r"[;,]", value):
            piece = piece.strip()
            if piece:
                tokens.append(piece)
    return tokens


def _object_matches(obj: TerrainObject, tokens: Sequence[str]) -> bool:
    if not tokens:
        return False
    type_name = (obj.type_name or "").lower()
    simplified = type_name.replace("model_", "")
    for token in tokens:
        lowered = token.lower()
        if token.isdigit() and int(token) == obj.type_id:
            return True
        if lowered in type_name or lowered in simplified:
            return True
    return False


def apply_object_filters(
    objects: Sequence[TerrainObject],
    include_tokens: Sequence[str],
    exclude_tokens: Sequence[str],
) -> List[TerrainObject]:
    filtered: List[TerrainObject] = []
    for obj in objects:
        if exclude_tokens and _object_matches(obj, exclude_tokens):
            continue
        if include_tokens:
            if _object_matches(obj, include_tokens):
                filtered.append(obj)
        else:
            filtered.append(obj)
    return filtered


def _normalize_for_colormap(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float32)
    min_val = float(np.min(matrix))
    max_val = float(np.max(matrix))
    if math.isclose(max_val, min_val):
        return np.zeros_like(matrix, dtype=np.float32)
    return (matrix - min_val) / (max_val - min_val)


def _overlay_matrix(data: TerrainData, overlay: str) -> Tuple[np.ndarray, str]:
    if overlay == "attributes":
        return data.attributes.astype(np.float32), "tab20"
    if overlay == "height":
        return data.height.astype(np.float32), "viridis"
    return data.mapping_layer1.astype(np.float32), "terrain"


def render_scene(
    data: TerrainData,
    objects: Sequence[TerrainObject],
    *,
    output: Optional[Path],
    show: bool,
    title: Optional[str] = None,
    enable_object_edit: bool = False,
    view_mode: str = "3d",
    overlay: str = "textures",
    texture_library: Optional[TextureLibrary] = None,
    material_library: Optional[MaterialStateLibrary] = None,
    renderer: str = "matplotlib",
    bmd_library: Optional[BMDLibrary] = None,
    fog_color: Optional[Tuple[float, float, float]] = None,
    fog_density: Optional[float] = None,
    scene_focus: str = "terrain",
) -> None:
    valid_focus = {"terrain", "full"}
    if scene_focus not in valid_focus:
        raise ValueError(f"Scene focus inválido: {scene_focus}. Opções: {sorted(valid_focus)}")
    if view_mode == "2d":
        if scene_focus == "terrain":
            raise ValueError("Scene focus 'terrain' requer visualização 3D.")
        matrix, cmap_name = _overlay_matrix(data, overlay)
        fig, ax = plt.subplots(figsize=(9, 8))
        image = ax.imshow(matrix, origin="lower", cmap=cmap_name)
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("Overlay")
        if objects:
            ox = np.array([obj.tile_position[0] for obj in objects])
            oy = np.array([obj.tile_position[1] for obj in objects])
            ax.scatter(ox, oy, c="white", s=10, edgecolors="black", linewidths=0.2)
        ax.set_xlim(0, TERRAIN_SIZE - 1)
        ax.set_ylim(0, TERRAIN_SIZE - 1)
        ax.set_xlabel("X (tiles)")
        ax.set_ylabel("Y (tiles)")
        ax.set_title(title or "Visualização 2D do terreno")
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        if output:
            fig.savefig(output)
            print(f"Visualização salva em {output}")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return

    if renderer == "opengl":
        if moderngl is None or pyglet is None:
            raise RuntimeError(
                "Renderer OpenGL indisponível: instale 'moderngl' e 'pyglet' ou use --renderer matplotlib."
            )
        if texture_library is None:
            raise ValueError("O renderer OpenGL requer um TextureLibrary inicializado.")
        app = OpenGLTerrainApp(
            data,
            objects,
            texture_library=texture_library,
            material_library=material_library,
            bmd_library=bmd_library,
            overlay=overlay,
            title=title or "Visualização OpenGL",
            fog_color=fog_color or (0.25, 0.33, 0.45),
            fog_density=fog_density or 0.00025,
            scene_focus=scene_focus,
        )
        app.run(show=show, output=output)
        return

    heights = data.height
    matrix, cmap_name = _overlay_matrix(data, overlay)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    if overlay == "textures" and texture_library is not None:
        xx, yy, render_heights, facecolors = texture_library.build_surface(data)
    else:
        x = np.arange(TERRAIN_SIZE, dtype=np.float32)
        y = np.arange(TERRAIN_SIZE, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        render_heights = heights.astype(np.float32)
        cmap = plt.get_cmap(cmap_name)
        normalized = _normalize_for_colormap(matrix)
        base_colors = cmap(normalized)
        facecolors = base_colors[:-1, :-1, :].copy()
        light_source = LightSource(azdeg=315, altdeg=55)
        shaded_rgb = light_source.shade_rgb(
            facecolors[..., :3], render_heights[:-1, :-1], fraction=0.6
        )
        facecolors[..., :3] = shaded_rgb
        facecolors = np.clip(facecolors, 0.0, 1.0)

    ax.plot_surface(
        xx,
        yy,
        render_heights,
        rstride=1,
        cstride=1,
        facecolors=facecolors,
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    scatter: Optional[Path3DCollection] = None
    if objects:
        ox = np.array([obj.tile_position[0] for obj in objects])
        oy = np.array([obj.tile_position[1] for obj in objects])
        oz = np.array([bilinear_height(heights, x, y) for x, y in zip(ox, oy)])
        type_ids = np.array([obj.type_id for obj in objects], dtype=float)
        if type_ids.size > 0:
            ptp_val = np.ptp(type_ids)
            if ptp_val > 0:
                colors = (type_ids - np.min(type_ids)) / ptp_val
            else:
                colors = np.zeros_like(type_ids)
        else:
            colors = np.zeros_like(type_ids)
        scatter = ax.scatter(
            ox,
            oy,
            oz + 50.0,
            c=colors,
            cmap="tab20",
            s=10,
            depthshade=False,
            picker=True,
            pickradius=5,
        )

        if enable_object_edit and show:
            editor = ObjectEditor(
                fig,
                ax,
                scatter,
                objects,
                data.height,
                camera_controls=True,
            )
            fig._terrain_object_editor = editor  # type: ignore[attr-defined]

    if show:
        fig._terrain_camera_navigator = CameraNavigator(  # type: ignore[attr-defined]
            fig,
            ax,
            terrain_bounds=(0.0, float(TERRAIN_SIZE - 1)),
        )

        if enable_object_edit and show:
            editor = ObjectEditor(
                fig,
                ax,
                scatter,
                objects,
                data.height,
                camera_controls=True,
            )
            fig._terrain_object_editor = editor  # type: ignore[attr-defined]

    if show:
        fig._terrain_camera_navigator = CameraNavigator(  # type: ignore[attr-defined]
            fig,
            ax,
            terrain_bounds=(0.0, float(TERRAIN_SIZE - 1)),
        )

        ax.set_xlabel("X (tiles)")
        ax.set_ylabel("Y (tiles)")
        ax.set_zlabel("Altura")
        ax.view_init(elev=60, azim=45)
        if scene_focus == "terrain":
            ax.set_proj_type("persp")
            ax.grid(False)
        if title:
            ax.set_title(title)
        plt.tight_layout()

    if output:
        fig.savefig(output)
        print(f"Visualização salva em {output}")

    if show:
        plt.show()
    else:
        plt.close(fig)


class CameraNavigator:
    # Adiciona controles de camera para navegar na cena tridimensional.

    def __init__(
        self,
        fig: Figure,
        ax: Axes,
        *,
        terrain_bounds: Tuple[float, float],
        pan_fraction: float = 0.12,
        zoom_step: float = 0.8,
        min_window: float = 8.0,
    ) -> None:
        self.fig = fig
        self.ax = ax
        self.terrain_min, self.terrain_max = terrain_bounds
        self.pan_fraction = pan_fraction
        self.zoom_step = zoom_step
        self.min_window = min_window
        self.azim = getattr(ax, "azim", 45.0)
        self.elev = getattr(ax, "elev", 30.0)

        canvas = fig.canvas
        self.cid_key = canvas.mpl_connect("key_press_event", self.on_key_press)
        self.cid_scroll = canvas.mpl_connect("scroll_event", self.on_scroll)
        self.cid_close = canvas.mpl_connect("close_event", self.on_close)

        self.info_label = ax.text2D(
            0.02,
            0.86,
            self._info_text(),
            transform=ax.transAxes,
            color="white",
            fontsize=9,
            ha="left",
            va="top",
            bbox={"facecolor": "black", "alpha": 0.5, "pad": 4},
        )

        print(
            "Controles de câmera: WASD move, Q/E ajusta zoom, I/K altera a inclinação,",
            "J/L gira a cena e o scroll também dá zoom.",
        )

    def _info_text(self) -> str:
        return (
            "WASD: mover câmera  |  Q/E: zoom  |  I/K: inclinar  |  J/L: girar\n"
            "Scroll do mouse: zoom"
        )

    def _clamp_interval(self, lower: float, upper: float) -> Tuple[float, float]:
        min_bound = self.terrain_min
        max_bound = self.terrain_max
        if lower < min_bound:
            shift = min_bound - lower
            lower += shift
            upper += shift
        if upper > max_bound:
            shift = upper - max_bound
            lower -= shift
            upper -= shift
        if lower < min_bound:
            lower = min_bound
        if upper > max_bound:
            upper = max_bound
        return lower, upper

    def _apply_limits(self, x_limits: Tuple[float, float], y_limits: Tuple[float, float]) -> None:
        self.ax.set_xlim3d(*x_limits)
        self.ax.set_ylim3d(*y_limits)
        self.fig.canvas.draw_idle()

    def _pan(self, dx: float, dy: float) -> None:
        x_min, x_max = self.ax.get_xlim3d()
        y_min, y_max = self.ax.get_ylim3d()
        width = x_max - x_min
        height = y_max - y_min
        if width <= 0 or height <= 0:
            return
        shift_x = dx * width * self.pan_fraction
        shift_y = dy * height * self.pan_fraction
        new_x = self._clamp_interval(x_min + shift_x, x_max + shift_x)
        new_y = self._clamp_interval(y_min + shift_y, y_max + shift_y)
        self._apply_limits(new_x, new_y)

    def _zoom(self, factor: float) -> None:
        x_min, x_max = self.ax.get_xlim3d()
        y_min, y_max = self.ax.get_ylim3d()
        width = x_max - x_min
        height = y_max - y_min
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        span_limit = max(self.min_window, 1e-6)
        max_span = max(self.terrain_max - self.terrain_min, span_limit)
        new_width = min(max_span, max(span_limit, width * factor))
        new_height = min(max_span, max(span_limit, height * factor))
        x_limits = (
            center_x - new_width / 2.0,
            center_x + new_width / 2.0,
        )
        y_limits = (
            center_y - new_height / 2.0,
            center_y + new_height / 2.0,
        )
        x_limits = self._clamp_interval(*x_limits)
        y_limits = self._clamp_interval(*y_limits)
        self._apply_limits(x_limits, y_limits)

    def _orbit(self, delta_azim: float, delta_elev: float) -> None:
        self.azim = (self.azim + delta_azim) % 360
        self.elev = float(np.clip(self.elev + delta_elev, -10.0, 90.0))
        self.ax.view_init(elev=self.elev, azim=self.azim)
        self.fig.canvas.draw_idle()

    def on_key_press(self, event: KeyEvent) -> None:
        key = (event.key or "").lower()
        if not key:
            return
        if "+" in key:
            key = key.split("+")[-1]
        if key == "w":
            self._pan(0.0, 1.0)
        elif key == "s":
            self._pan(0.0, -1.0)
        elif key == "a":
            self._pan(-1.0, 0.0)
        elif key == "d":
            self._pan(1.0, 0.0)
        elif key == "q":
            self._zoom(self.zoom_step)
        elif key == "e":
            self._zoom(1.0 / self.zoom_step)
        elif key == "j":
            self._orbit(-5.0, 0.0)
        elif key == "l":
            self._orbit(5.0, 0.0)
        elif key == "i":
            self._orbit(0.0, 3.0)
        elif key == "k":
            self._orbit(0.0, -3.0)

    def on_scroll(self, event) -> None:  # type: ignore[override]
        if getattr(event, "step", 0) > 0:
            self._zoom(self.zoom_step)
        else:
            self._zoom(1.0 / self.zoom_step)

    def on_close(self, _event: Optional[object]) -> None:
        canvas = self.fig.canvas
        canvas.mpl_disconnect(self.cid_key)
        canvas.mpl_disconnect(self.cid_scroll)
        canvas.mpl_disconnect(self.cid_close)
        self.info_label.remove()


class ObjectEditor:
    # Permite mover objetos renderizados usando eventos do Matplotlib.

    def __init__(
        self,
        fig: Figure,
        ax: Axes,
        scatter: Path3DCollection,
        objects: Sequence[TerrainObject],
        heights: np.ndarray,
        *,
        camera_controls: bool = False,
    ) -> None:
        self.fig = fig
        self.ax = ax
        self.scatter = scatter
        self.objects = list(objects)
        self.heights = heights
        self.camera_controls = camera_controls
        self.selected_index: Optional[int] = None
        self.step_tiles = 0.5

        self.info_label = ax.text2D(
            0.02,
            0.02,
            self._format_info_text(camera_controls),
            transform=ax.transAxes,
            color="yellow",
            fontsize=9,
            ha="left",
            va="bottom",
            bbox={"facecolor": "black", "alpha": 0.6, "pad": 4},
        )
        self.selection_label = ax.text2D(
            0.02,
            0.94,
            "",
            transform=ax.transAxes,
            color="cyan",
            fontsize=10,
            ha="left",
            va="top",
            bbox={"facecolor": "black", "alpha": 0.5, "pad": 4},
        )
        self.selection_label.set_visible(False)

        canvas = fig.canvas
        self.cid_pick = canvas.mpl_connect("pick_event", self.on_pick)
        self.cid_key = canvas.mpl_connect("key_press_event", self.on_key_press)
        self.cid_close = canvas.mpl_connect("close_event", self.on_close)

        print(
            "Edição habilitada: clique em um ponto para selecionar um objeto e use"
            " as setas para mover. Shift acelera o passo. Use [ e ] para ajustar"
            " o passo."
        )

    def _format_info_text(self, camera_controls: bool) -> str:
        text = (
            "Setas: mover objeto  |  Shift: x5  |  [: passo/2  |  ]: passo*2  "
            f"| Passo atual: {self.step_tiles:.2f} tiles"
        )
        if camera_controls:
            text += "\nWASD/QE/IJKL: mover câmera"
        return text

    def on_pick(self, event: PickEvent) -> None:
        if event.artist is not self.scatter:
            return
        indices = getattr(event, "ind", [])
        if not indices:
            return
        index_array = np.atleast_1d(indices)
        self.selected_index = int(index_array[0])
        obj = self.objects[self.selected_index]
        text = (
            f"Selecionado #{self.selected_index} — ID {obj.type_id}"
            f" ({obj.type_name or 'sem nome'})\n"
            f"Tile: {obj.tile_position[0]:.2f}, {obj.tile_position[1]:.2f}"
        )
        self.selection_label.set_text(text)
        self.selection_label.set_visible(True)
        self.fig.canvas.draw_idle()

    def on_key_press(self, event: KeyEvent) -> None:
        if self.selected_index is None:
            return
        if not event.key:
            return

        parts = event.key.split("+")
        key = parts[-1]
        modifiers = set(parts[:-1])

        if key == "escape":
            self.selected_index = None
            self.selection_label.set_visible(False)
            self.fig.canvas.draw_idle()
            return

        if key == "]":
            self.step_tiles = min(self.step_tiles * 2.0, 10.0)
            self.info_label.set_text(self._format_info_text(self.camera_controls))
            self.fig.canvas.draw_idle()
            return

        if key == "[":
            self.step_tiles = max(self.step_tiles / 2.0, 0.03125)
            self.info_label.set_text(self._format_info_text(self.camera_controls))
            self.fig.canvas.draw_idle()
            return

        multiplier = 5.0 if "shift" in modifiers else 1.0
        step = self.step_tiles * multiplier

        delta_x = 0.0
        delta_y = 0.0
        if key == "up":
            delta_y += step
        elif key == "down":
            delta_y -= step
        elif key == "left":
            delta_x -= step
        elif key == "right":
            delta_x += step
        else:
            return

        self._move_selected(delta_x, delta_y)

    def _move_selected(self, delta_x: float, delta_y: float) -> None:
        if self.selected_index is None:
            return
        obj = self.objects[self.selected_index]
        tile_x = obj.tile_position[0] + delta_x
        tile_y = obj.tile_position[1] + delta_y
        tile_x = float(np.clip(tile_x, 0.0, TERRAIN_SIZE - 1))
        tile_y = float(np.clip(tile_y, 0.0, TERRAIN_SIZE - 1))
        height = bilinear_height(self.heights, tile_x, tile_y)

        obj.position = (
            tile_x * TERRAIN_SCALE,
            tile_y * TERRAIN_SCALE,
            height,
        )

        ox, oy, oz = self.scatter._offsets3d
        ox_arr = np.asarray(ox)
        oy_arr = np.asarray(oy)
        oz_arr = np.asarray(oz)
        ox_arr[self.selected_index] = tile_x
        oy_arr[self.selected_index] = tile_y
        oz_arr[self.selected_index] = height + 50.0
        self.scatter._offsets3d = (ox_arr, oy_arr, oz_arr)  # type: ignore[assignment]

        self.selection_label.set_text(
            f"Selecionado #{self.selected_index} — ID {obj.type_id}"
            f" ({obj.type_name or 'sem nome'})\n"
            f"Tile: {tile_x:.2f}, {tile_y:.2f}"
        )
        self.fig.canvas.draw_idle()

    def on_close(self, _event: Optional[object]) -> None:
        canvas = self.fig.canvas
        canvas.mpl_disconnect(self.cid_pick)
        canvas.mpl_disconnect(self.cid_key)
        canvas.mpl_disconnect(self.cid_close)


def find_first(pattern: str, directory: Path) -> Optional[Path]:
    for candidate in sorted(directory.glob(pattern)):
        if candidate.is_file():
            return candidate
    return None


def infer_map_id(world_path: Path) -> Optional[int]:
    for part in world_path.parts[::-1]:
        if part.lower().startswith("world"):
            digits = "".join(ch for ch in part if ch.isdigit())
            if digits:
                return int(digits)
    return None


def guess_object_folder(world_path: Path) -> Optional[Path]:
    name = world_path.name
    lowered = name.lower()
    if lowered.startswith("world"):
        suffix = name[len("World") :]
        candidate = world_path.parent / f"Object{suffix}"
        if candidate.is_dir():
            return candidate
    # Também tente a variação com caixa baixa, caso o pacote use minúsculas.
    if lowered.startswith("world"):
        suffix = lowered[len("world") :]
        candidate = world_path.parent / f"object{suffix}"
        if candidate.is_dir():
            return candidate
    return None


def resolve_files(
    world_path: Path, map_id: Optional[int], *, object_path: Optional[Path]
) -> Tuple[Path, Path, Path, Path]:
    if map_id is None:
        map_id = infer_map_id(world_path)

    def build_pattern(prefix: str, extension: str) -> str:
        return f"{prefix}{map_id if map_id is not None else ''}{extension}"

    mapping = find_first(build_pattern("EncTerrain", ".map"), world_path)
    attributes = find_first(build_pattern("EncTerrain", ".att"), world_path)
    if not mapping or not attributes:
        raise FileNotFoundError(
            "Não foi possível localizar arquivos .map/.att correspondentes no diretório informado."
        )

    object_dir_candidates: List[Path] = []
    if object_path is not None:
        if not object_path.is_dir():
            raise FileNotFoundError(f"Diretório de objetos inválido: {object_path}")
        object_dir_candidates.append(object_path)
    object_guess = guess_object_folder(world_path)
    if object_guess is not None:
        object_dir_candidates.append(object_guess)
    object_dir_candidates.append(world_path)

    objects: Optional[Path] = None
    for candidate in object_dir_candidates:
        objects = find_first(build_pattern("EncTerrain", ".obj"), candidate)
        if objects:
            break
    if not objects:
        raise FileNotFoundError(
            "Não foi possível localizar arquivo .obj correspondente."
            " Informe --object-path para apontar para a pasta ObjectX correta."
        )

    classic_min_size = 4 + 1080 + TERRAIN_SIZE * TERRAIN_SIZE
    extended_min_size = 4 + 54 + TERRAIN_SIZE * TERRAIN_SIZE * 3

    height = world_path / "TerrainHeight.OZB"
    if height.exists() and height.stat().st_size < classic_min_size:
        alt_height = world_path / "TerrainHeightNew.OZB"
        if alt_height.exists() and alt_height.stat().st_size >= extended_min_size:
            height = alt_height
    if not height.exists():
        height = world_path / "TerrainHeightNew.OZB"
    if not height.exists():
        raise FileNotFoundError("Arquivo de altura TerrainHeight.OZB ou TerrainHeightNew.OZB não encontrado.")

    return attributes, mapping, objects, height


def load_world_data(
    world_path: Path,
    *,
    map_id: Optional[int],
    object_path: Optional[Path],
    extended_height: bool,
    height_scale: Optional[float],
    enum_path: Optional[Path],
) -> TerrainLoadResult:
    if not world_path.is_dir():
        raise FileNotFoundError(f"Diretório inválido: {world_path}")

    attributes_path, mapping_path, objects_path, height_path = resolve_files(
        world_path, map_id, object_path=object_path
    )

    enum_candidate = enum_path
    if enum_candidate is None:
        if DEFAULT_ENUM_PATH.exists():
            enum_candidate = DEFAULT_ENUM_PATH
    model_names: Mapping[int, str] = {}
    if enum_candidate and enum_candidate.exists():
        model_names = load_model_names(str(enum_candidate.resolve()))

    attr_map_id, attributes = open_terrain_attribute(attributes_path)
    mapping_map_id, layer1, layer2, alpha = open_terrain_mapping(mapping_path)
    obj_map_id, objects_version, objects = open_objects_enc(objects_path, model_names)
    height = load_height_file(
        height_path,
        extended=extended_height or height_path.name.endswith("New.OZB"),
        scale_override=height_scale,
    )

    all_objects = list(objects)
    terrain = TerrainData(
        height=height,
        mapping_layer1=layer1,
        mapping_layer2=layer2,
        mapping_alpha=alpha,
        attributes=attributes,
    )

    resolved_map_id: Optional[int] = None
    id_sources = [
        ("parâmetro", map_id),
        ("EncTerrain.map", mapping_map_id),
        ("EncTerrain.att", attr_map_id),
        ("EncTerrain.obj", obj_map_id),
    ]
    for label, value in id_sources:
        if value is None:
            continue
        if resolved_map_id is None:
            resolved_map_id = value
        elif value != resolved_map_id:
            raise ValueError(
                f"ID do mapa inconsistente: {label} aponta para {value},"
                f" mas o esperado é {resolved_map_id}."
            )

    if resolved_map_id is None:
        resolved_map_id = 0

    return TerrainLoadResult(
        world_path=world_path,
        data=terrain,
        objects=objects,
        map_id=resolved_map_id,
        map_id_attribute=attr_map_id,
        map_id_mapping=mapping_map_id,
        map_id_objects=obj_map_id,
        model_names=model_names,
        objects_path=objects_path,
        objects_version=objects_version,
        all_objects=all_objects,
    )


def object_summary(result: TerrainLoadResult, *, limit: int = 8) -> List[Tuple[int, int, Optional[str]]]:
    counter = Counter(obj.type_id for obj in result.objects)
    summary: List[Tuple[int, int, Optional[str]]] = []
    for type_id, count in counter.most_common(limit):
        summary.append((type_id, count, result.model_names.get(type_id)))
    return summary


def format_summary_line(result: TerrainLoadResult, *, limit: int = 5) -> str:
    pieces = [
        f"Mapa {result.map_id}",
    ]
    visible = len(result.objects)
    total = len(result.all_objects)
    if visible != total:
        pieces.append(f"{visible} de {total} objetos")
    else:
        pieces.append(f"{visible} objetos")
    highlights = []
    for type_id, count, name in object_summary(result, limit=limit):
        label = name.replace("MODEL_", "") if name else "ID"
        highlights.append(f"{count}× {label} ({type_id})")
    if highlights:
        pieces.append("principais: " + ", ".join(highlights))
    return " | ".join(pieces)


def print_summary(result: TerrainLoadResult, *, limit: int = 8) -> None:
    print(format_summary_line(result, limit=limit))


def attribute_summary(
    attributes: np.ndarray, *, limit: int = 5
) -> List[Tuple[int, int, float]]:
    values, counts = np.unique(attributes, return_counts=True)
    if counts.size == 0:
        return []
    order = np.argsort(counts)[::-1]
    total = attributes.size
    summary: List[Tuple[int, int, float]] = []
    for idx in order[:limit]:
        summary.append((int(values[idx]), int(counts[idx]), counts[idx] / float(total)))
    return summary


def format_detailed_summary(
    result: TerrainLoadResult,
    *,
    object_limit: int = 5,
    attribute_limit: int = 5,
) -> str:
    lines = [format_summary_line(result, limit=object_limit)]
    heights = result.data.height
    lines.append(
        "Altura: mín {:.1f}, máx {:.1f}, média {:.1f}".format(
            float(np.min(heights)), float(np.max(heights)), float(np.mean(heights))
        )
    )

    layer1_unique = np.unique(result.data.mapping_layer1)
    layer2_unique = np.unique(result.data.mapping_layer2)
    alpha_active = np.count_nonzero(result.data.mapping_alpha > 0.01)
    total_tiles = result.data.mapping_alpha.size
    alpha_pct = 100.0 * alpha_active / float(total_tiles)
    lines.append(
        f"Texturas: {len(layer1_unique)} IDs na camada 1, {len(layer2_unique)} na camada 2"
    )
    lines.append(f"Alpha misto presente em {alpha_pct:.1f}% dos tiles")

    attr_lines = []
    for value, count, ratio in attribute_summary(result.data.attributes, limit=attribute_limit):
        attr_lines.append(f"{value} ({count} tiles, {ratio * 100:.1f}%)")
    if attr_lines:
        lines.append("Atributos mais comuns: " + ", ".join(attr_lines))

    if result.objects:
        dominant = []
        for type_id, count, name in object_summary(result, limit=object_limit):
            label = name.replace("MODEL_", "") if name else str(type_id)
            dominant.append(f"{label}: {count}")
        if dominant:
            lines.append("Objetos em destaque: " + ", ".join(dominant))

    return "\n".join(lines)


def print_detailed_summary(
    result: TerrainLoadResult, *, object_limit: int = 8, attribute_limit: int = 5
) -> None:
    print(
        format_detailed_summary(
            result, object_limit=object_limit, attribute_limit=attribute_limit
        )
    )


def export_objects_csv(
    result: TerrainLoadResult,
    destination: Path,
    *,
    include_tile_coords: bool = True,
    objects: Optional[Sequence[TerrainObject]] = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "type_id",
        "type_name",
        "x",
        "y",
        "z",
        "pitch",
        "yaw",
        "roll",
        "scale",
    ]
    if include_tile_coords:
        fieldnames.extend(["tile_x", "tile_y"])

    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for obj in (objects if objects is not None else result.objects):
            row = {
                "type_id": obj.type_id,
                "type_name": obj.type_name or "",
                "x": f"{obj.position[0]:.3f}",
                "y": f"{obj.position[1]:.3f}",
                "z": f"{obj.position[2]:.3f}",
                "pitch": f"{obj.angles[0]:.3f}",
                "yaw": f"{obj.angles[1]:.3f}",
                "roll": f"{obj.angles[2]:.3f}",
                "scale": f"{obj.scale:.3f}",
            }
            if include_tile_coords:
                tile_x, tile_y = obj.tile_position
                row["tile_x"] = f"{tile_x:.3f}"
                row["tile_y"] = f"{tile_y:.3f}"
            writer.writerow(row)


def save_objects_file(
    result: TerrainLoadResult, destination: Path, *, objects: Optional[Sequence[TerrainObject]] = None
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    used_objects = list(objects if objects is not None else result.all_objects)
    payload = bytearray()
    payload.append(result.objects_version & 0xFF)
    payload.append(result.map_id_objects & 0xFF)
    payload.extend(struct.pack("<h", len(used_objects)))
    for obj in used_objects:
        payload.extend(struct.pack("<h", obj.type_id))
        payload.extend(struct.pack("<3f", *obj.position))
        payload.extend(struct.pack("<3f", *obj.angles))
        payload.extend(struct.pack("<f", obj.scale))
    encrypted = map_file_encrypt(bytes(payload))
    destination.write_bytes(encrypted)


def export_result_json(
    result: TerrainLoadResult,
    destination: Path,
    *,
    objects: Optional[Sequence[TerrainObject]] = None,
    include_height_stats: bool = True,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    selected = list(objects if objects is not None else result.objects)
    payload: Dict[str, object] = {
        "map_id": result.map_id,
        "world": result.world_path.name,
        "object_count": len(selected),
        "total_objects": len(result.all_objects),
        "objects": [
            {
                "type_id": obj.type_id,
                "type_name": obj.type_name,
                "position": {
                    "x": obj.position[0],
                    "y": obj.position[1],
                    "z": obj.position[2],
                },
                "angles": {
                    "pitch": obj.angles[0],
                    "yaw": obj.angles[1],
                    "roll": obj.angles[2],
                },
                "scale": obj.scale,
                "tile": {
                    "x": obj.tile_position[0],
                    "y": obj.tile_position[1],
                },
            }
            for obj in selected
        ],
    }
    if include_height_stats:
        heights = result.data.height
        payload["height"] = {
            "min": float(np.min(heights)),
            "max": float(np.max(heights)),
            "mean": float(np.mean(heights)),
        }
    payload["attribute_histogram"] = [
        {"value": int(value), "count": int(count)}
        for value, count in Counter(result.data.attributes.flatten()).items()
    ]
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)



def run_viewer(
    world_path: Path,
    *,
    map_id: Optional[int],
    object_path: Optional[Path],
    extended_height: bool,
    height_scale: Optional[float],
    output: Optional[Path],
    show: bool,
    max_objects: Optional[int],
    enum_path: Optional[Path] = None,
    log_summary: bool = True,
    detailed_summary: bool = False,
    summary_limit: int = 8,
    render: bool = True,
    export_objects: Optional[Path] = None,
    enable_object_edit: bool = False,
    view_mode: str = "3d",
    overlay: str = "textures",
    scene_focus: str = "terrain",
    include_filters: Optional[Sequence[str]] = None,
    exclude_filters: Optional[Sequence[str]] = None,
    export_json: Optional[Path] = None,
    save_objects: Optional[Path] = None,
    texture_detail: int = 2,
    renderer: str = "matplotlib",
    fog_density: Optional[float] = None,
    fog_color: Optional[Tuple[float, float, float]] = None,
) -> TerrainLoadResult:
    if scene_focus not in {"terrain", "full"}:
        raise ValueError("Scene focus inválido. Use 'terrain' ou 'full'.")

    result = load_world_data(
        world_path,
        map_id=map_id,
        object_path=object_path,
        extended_height=extended_height,
        height_scale=height_scale,
        enum_path=enum_path,
    )

    include_tokens = _split_filter_values(include_filters)
    exclude_tokens = _split_filter_values(exclude_filters)
    filtered_objects = apply_object_filters(result.all_objects, include_tokens, exclude_tokens)
    display_objects = filtered_objects
    truncated = False
    if max_objects is not None and len(display_objects) > max_objects:
        display_objects = display_objects[:max_objects]
        truncated = True

    display_result = replace(result, objects=list(display_objects))
    warnings: List[str] = []

    requested_renderer = renderer
    active_renderer = renderer
    opengl_available = moderngl is not None and pyglet is not None
    if render and renderer == "opengl" and not opengl_available:
        warning = (
            "Renderer OpenGL indisponível: instale 'moderngl' e 'pyglet' ou selecione o modo Matplotlib."
        )
        print("Aviso:", warning)
        warnings.append(warning)
        active_renderer = "matplotlib"

    if enable_object_edit and not show:
        raise ValueError(
            "A edição de objetos requer a janela interativa. Remova --no-show para usar esta opção."
        )

    if enable_object_edit and view_mode != "3d":
        raise ValueError("A movimentação de objetos só está disponível no modo 3D.")

    if render:
        texture_library: Optional[TextureLibrary] = None
        material_library: Optional[MaterialStateLibrary] = None
        if requested_renderer == "opengl":
            texture_library = TextureLibrary(
                display_result.world_path,
                detail_factor=max(1, texture_detail),
                object_path=object_path,
                map_id=display_result.map_id,
            )
            if active_renderer == "opengl":
                material_roots: List[Path] = []
                if object_path:
                    material_roots.append(object_path)
                guessed = guess_object_folder(world_path)
                if guessed:
                    material_roots.append(guessed)
                material_library = MaterialStateLibrary(display_result.world_path, extra_roots=material_roots)
        if active_renderer != "opengl" and texture_library is None and view_mode == "3d" and overlay == "textures":
            texture_library = TextureLibrary(
                display_result.world_path,
                detail_factor=max(1, texture_detail),
                object_path=object_path,
                map_id=display_result.map_id,
            )

        bmd_library: Optional[BMDLibrary] = None
        if requested_renderer == "opengl" and active_renderer == "opengl":
            search_roots: List[Path] = []
            if object_path:
                search_roots.append(object_path)
            guessed = guess_object_folder(world_path)
            if guessed:
                search_roots.append(guessed)
            parent = world_path.parent
            if parent not in search_roots:
                search_roots.append(parent)
            bmd_library = BMDLibrary(search_roots, material_library=material_library)

        render_success = False
        if active_renderer == "opengl":
            try:
                render_scene(
                    display_result.data,
                    display_result.objects,
                    output=output,
                    show=show,
                    title=f"{world_path.name} (mapa {display_result.map_id}) — {len(display_result.objects)} objetos",
                    enable_object_edit=enable_object_edit,
                    view_mode=view_mode,
                    overlay=overlay,
                    texture_library=texture_library,
                    material_library=material_library,
                    renderer="opengl",
                    bmd_library=bmd_library,
                    fog_density=fog_density,
                    fog_color=fog_color,
                    scene_focus=scene_focus,
                )
                render_success = True
            except Exception as exc:  # noqa: BLE001
                warning = (
                    "Falha ao inicializar o renderer OpenGL. Alternando para Matplotlib."
                    f" Detalhes: {exc}"
                )
                print("Aviso:", warning)
                warnings.append(warning)
                active_renderer = "matplotlib"

        if not render_success and active_renderer != "opengl":
            if texture_library is None and view_mode == "3d" and overlay == "textures":
                texture_library = TextureLibrary(
                    display_result.world_path,
                    detail_factor=max(1, texture_detail),
                    object_path=object_path,
                    map_id=display_result.map_id,
                )
            render_scene(
                display_result.data,
                display_result.objects,
                output=output,
                show=show,
                title=f"{world_path.name} (mapa {display_result.map_id}) — {len(display_result.objects)} objetos",
                enable_object_edit=enable_object_edit,
                view_mode=view_mode,
                overlay=overlay,
                texture_library=texture_library,
                material_library=None,
                renderer="matplotlib",
                bmd_library=None,
                fog_density=fog_density,
                fog_color=fog_color,
                scene_focus=scene_focus,
            )
            render_success = True

        if texture_library is not None and texture_library.missing_indices:
            preview = ", ".join(map(str, texture_library.missing_indices[:10]))
            if len(texture_library.missing_indices) > 10:
                preview += ", ..."
            message = (
                "Não foi possível localizar todas as texturas. Índices ausentes: "
                + preview
            )
            print("Aviso:", message)
            warnings.append(message)

    if truncated:
        notice = (
            "Limite de objetos aplicado. Apenas"
            f" {len(display_result.objects)} de {len(filtered_objects)} objetos foram renderizados."
        )
        print("Aviso:", notice)
        warnings.append(notice)

    if export_objects is not None:
        export_objects_csv(display_result, export_objects)
        print(f"Objetos exportados em {export_objects}")

    if export_json is not None:
        export_result_json(display_result, export_json)
        print(f"Dados exportados em {export_json}")

    if save_objects is not None:
        save_objects_file(result, save_objects)
        print(f"Arquivo EncTerrain salvo em {save_objects}")

    if warnings:
        display_result.warnings.extend(warnings)

    if log_summary:
        if detailed_summary:
            print_detailed_summary(display_result, object_limit=summary_limit)
        else:
            print_summary(display_result, limit=summary_limit)
    return display_result


def list_world_directories(data_path: Path) -> List[Path]:
    worlds: List[Path] = []
    if not data_path.is_dir():
        return worlds
    for child in sorted(data_path.iterdir()):
        if not child.is_dir():
            continue
        lower = child.name.lower()
        if lower.startswith("world") and find_first("EncTerrain*.map", child):
            worlds.append(child)
            continue
        if find_first("EncTerrain*.map", child):
            worlds.append(child)
    return worlds


def _safe_int(value: str) -> Optional[int]:
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


class TerrainViewerGUI:
    def __init__(
        self,
        *,
        initial_data_dir: Optional[Path] = None,
        enum_path: Optional[Path] = None,
    ):
        self.root = tk.Tk()
        self.root.title("Visualizador de Terreno")

        self.data_dir_var = tk.StringVar()
        self.world_var = tk.StringVar()
        self.map_id_var = tk.StringVar()
        self.height_scale_var = tk.StringVar()
        self.max_objects_var = tk.StringVar()
        self.extended_height_var = tk.BooleanVar()
        self.edit_objects_var = tk.BooleanVar()
        self.view_mode_var = tk.StringVar(value="3D")
        self.overlay_var = tk.StringVar(value="Texturas")
        self.scene_focus_var = tk.StringVar(value="Somente terreno e instâncias")
        self.include_filter_var = tk.StringVar()
        self.exclude_filter_var = tk.StringVar()
        self.texture_detail_var = tk.StringVar(value="2")
        self.renderer_var = tk.StringVar(value="OpenGL")
        self.fog_density_var = tk.StringVar()
        self.fog_color_var = tk.StringVar()

        self.object_dir_var = tk.StringVar()
        self.status_var = tk.StringVar()

        self.world_options: List[Path] = []
        self.enum_path = None
        self.view_mode_map = {"3D": "3d", "2D": "2d"}
        self.overlay_map = {
            "Texturas": "textures",
            "Altura": "height",
            "Atributos": "attributes",
        }
        self.scene_focus_map = {
            "Somente terreno e instâncias": "terrain",
            "Cena completa": "full",
        }
        self.renderer_map = {"OpenGL": "opengl", "Matplotlib": "matplotlib"}
        if enum_path and enum_path.exists():
            self.enum_path = enum_path
        elif DEFAULT_ENUM_PATH.exists():
            self.enum_path = DEFAULT_ENUM_PATH

        self.last_result: Optional[TerrainLoadResult] = None
        self.last_params: Optional[Tuple[str, ...]] = None

        if initial_data_dir and initial_data_dir.is_dir():
            self.data_dir_var.set(str(initial_data_dir))
            self._populate_worlds(initial_data_dir)

        self._build_layout()

    def _on_view_mode_change(self, *_: object) -> None:
        mode = self.view_mode_map.get(self.view_mode_var.get(), "3d")
        if mode != "3d":
            if self.edit_objects_var.get():
                self.edit_objects_var.set(False)
            self.edit_checkbox.config(state="disabled")
        else:
            self.edit_checkbox.config(state="normal")

    def _current_view_mode(self) -> str:
        return self.view_mode_map.get(self.view_mode_var.get(), "3d")

    def _current_overlay(self) -> str:
        return self.overlay_map.get(self.overlay_var.get(), "textures")

    def _current_filter_values(self) -> Tuple[Optional[str], Optional[str]]:
        include = self.include_filter_var.get().strip()
        exclude = self.exclude_filter_var.get().strip()
        return (include or None, exclude or None)

    def _compose_params(
        self,
        world_path: Path,
        map_id: Optional[int],
        height_scale: Optional[float],
        max_objects: Optional[int],
        object_dir: Optional[Path],
        view_mode: str,
        overlay: str,
        scene_focus: str,
        include: Optional[str],
        exclude: Optional[str],
        texture_detail: int,
        renderer: str,
        fog_density: Optional[float],
        fog_color: Optional[Tuple[float, float, float]],
    ) -> Tuple[str, ...]:
        return (
            str(world_path.resolve()),
            "" if map_id is None else str(map_id),
            "" if height_scale is None else f"{height_scale:.6f}",
            "" if max_objects is None else str(max_objects),
            "" if object_dir is None else str(object_dir.resolve()),
            "1" if self.extended_height_var.get() else "0",
            view_mode,
            overlay,
            scene_focus,
            include or "",
            exclude or "",
            str(texture_detail),
            renderer,
            "" if fog_density is None else f"{fog_density:.6f}",
            "" if fog_color is None else ",".join(f"{component:.3f}" for component in fog_color),
        )

    def _can_reuse_last(self, params: Tuple[str, ...]) -> bool:
        return self.last_result is not None and self.last_params == params

    def _gather_context(
        self,
    ) -> Tuple[
        Path,
        Optional[int],
        Optional[float],
        Optional[int],
        Optional[Path],
        str,
        str,
        str,
        Optional[str],
        Optional[str],
        int,
        str,
        Optional[float],
        Optional[Tuple[float, float, float]],
        Tuple[str, ...],
    ]:
        world_path = self._current_world_path()
        if world_path is None:
            raise ValueError("Selecione uma pasta World válida.")
        map_id = _safe_int(self.map_id_var.get())
        height_scale = self._parse_float(self.height_scale_var.get())
        max_objects = _safe_int(self.max_objects_var.get())
        object_dir = Path(self.object_dir_var.get()) if self.object_dir_var.get() else None
        view_mode = self._current_view_mode()
        overlay = self._current_overlay()
        scene_focus = self.scene_focus_map.get(self.scene_focus_var.get(), "terrain")
        include, exclude = self._current_filter_values()
        texture_detail = _safe_int(self.texture_detail_var.get()) or 2
        if texture_detail < 1:
            texture_detail = 1
        renderer = self.renderer_map.get(self.renderer_var.get(), "opengl")
        fog_density = self._parse_float(self.fog_density_var.get())
        fog_color = self._parse_color(self.fog_color_var.get())
        params = self._compose_params(
            world_path,
            map_id,
            height_scale,
            max_objects,
            object_dir,
            view_mode,
            overlay,
            scene_focus,
            include,
            exclude,
            texture_detail,
            renderer,
            fog_density,
            fog_color,
        )
        return (
            world_path,
            map_id,
            height_scale,
            max_objects,
            object_dir,
            view_mode,
            overlay,
            scene_focus,
            include,
            exclude,
            texture_detail,
            renderer,
            fog_density,
            fog_color,
            params,
        )

    def _build_layout(self) -> None:
        padding = {"padx": 8, "pady": 4}

        data_frame = tk.LabelFrame(self.root, text="Diretórios")
        data_frame.grid(row=0, column=0, sticky="ew", **padding)

        tk.Label(data_frame, text="Pasta Data:").grid(row=0, column=0, sticky="w")
        data_entry = tk.Entry(data_frame, textvariable=self.data_dir_var, width=50)
        data_entry.grid(row=0, column=1, sticky="ew", padx=(4, 4))
        tk.Button(data_frame, text="Escolher...", command=self.choose_data_dir).grid(row=0, column=2)

        tk.Label(data_frame, text="Pasta World:").grid(row=1, column=0, sticky="w")
        self.world_menu = tk.OptionMenu(data_frame, self.world_var, "")
        self.world_menu.config(width=45)
        self.world_menu.grid(row=1, column=1, sticky="ew", padx=(4, 4))
        tk.Button(data_frame, text="Atualizar", command=self.refresh_worlds).grid(row=1, column=2)

        tk.Label(data_frame, text="Pasta Object (opcional):").grid(row=2, column=0, sticky="w")
        object_entry = tk.Entry(data_frame, textvariable=self.object_dir_var, width=50)
        object_entry.grid(row=2, column=1, sticky="ew", padx=(4, 4))
        tk.Button(data_frame, text="Escolher...", command=self.choose_object_dir).grid(row=2, column=2)

        data_frame.columnconfigure(1, weight=1)

        options_frame = tk.LabelFrame(self.root, text="Opções")
        options_frame.grid(row=1, column=0, sticky="ew", **padding)

        tk.Label(options_frame, text="Map ID:").grid(row=0, column=0, sticky="w")
        tk.Entry(options_frame, textvariable=self.map_id_var, width=10).grid(row=0, column=1, sticky="w")

        tk.Label(options_frame, text="Height scale:").grid(row=0, column=2, sticky="w")
        tk.Entry(options_frame, textvariable=self.height_scale_var, width=10).grid(row=0, column=3, sticky="w")

        tk.Label(options_frame, text="Max objects:").grid(row=1, column=0, sticky="w")
        tk.Entry(options_frame, textvariable=self.max_objects_var, width=10).grid(row=1, column=1, sticky="w")

        tk.Checkbutton(
            options_frame,
            text="Forçar TerrainHeightNew",
            variable=self.extended_height_var,
        ).grid(row=1, column=2, columnspan=2, sticky="w")

        tk.Label(options_frame, text="Modo de visualização:").grid(row=2, column=0, sticky="w")
        tk.OptionMenu(
            options_frame,
            self.view_mode_var,
            *self.view_mode_map.keys(),
        ).grid(row=2, column=1, sticky="ew")

        tk.Label(options_frame, text="Overlay:").grid(row=2, column=2, sticky="w")
        tk.OptionMenu(
            options_frame,
            self.overlay_var,
            *self.overlay_map.keys(),
        ).grid(row=2, column=3, sticky="ew")

        tk.Label(options_frame, text="Detalhe textura:").grid(row=2, column=4, sticky="w")
        tk.Entry(options_frame, textvariable=self.texture_detail_var, width=5).grid(
            row=2, column=5, sticky="w"
        )

        tk.Label(options_frame, text="Renderer:").grid(row=2, column=6, sticky="w")
        tk.OptionMenu(
            options_frame,
            self.renderer_var,
            *self.renderer_map.keys(),
        ).grid(row=2, column=7, sticky="ew")

        tk.Label(options_frame, text="Foco da cena:").grid(row=3, column=0, sticky="w")
        tk.OptionMenu(
            options_frame,
            self.scene_focus_var,
            *self.scene_focus_map.keys(),
        ).grid(row=3, column=1, sticky="ew")

        tk.Label(options_frame, text="Névoa densidade:").grid(row=4, column=0, sticky="w")
        tk.Entry(options_frame, textvariable=self.fog_density_var, width=10).grid(
            row=4, column=1, sticky="w"
        )

        tk.Label(options_frame, text="Cor névoa (R,G,B):").grid(row=4, column=2, sticky="w")
        tk.Entry(options_frame, textvariable=self.fog_color_var, width=20).grid(
            row=4, column=3, columnspan=3, sticky="ew"
        )

        tk.Label(options_frame, text="Mostrar apenas (ID/nome):").grid(row=5, column=0, sticky="w")
        tk.Entry(options_frame, textvariable=self.include_filter_var, width=20).grid(row=5, column=1, sticky="ew")

        tk.Label(options_frame, text="Ocultar (ID/nome):").grid(row=5, column=2, sticky="w")
        tk.Entry(options_frame, textvariable=self.exclude_filter_var, width=20).grid(row=5, column=3, sticky="ew")

        self.edit_checkbox = tk.Checkbutton(
            options_frame,
            text="Permitir mover objetos (janela interativa)",
            variable=self.edit_objects_var,
        )
        self.edit_checkbox.grid(row=6, column=0, columnspan=4, sticky="w")

        options_frame.columnconfigure(1, weight=1)
        options_frame.columnconfigure(3, weight=1)
        options_frame.columnconfigure(7, weight=1)

        buttons_frame = tk.Frame(self.root)
        buttons_frame.grid(row=2, column=0, sticky="e", **padding)

        tk.Button(buttons_frame, text="Visualizar", command=self.visualize).grid(row=0, column=0, padx=4)
        tk.Button(buttons_frame, text="Salvar PNG", command=self.save_png).grid(row=0, column=1, padx=4)
        tk.Button(buttons_frame, text="Exportar objetos", command=self.export_objects).grid(row=0, column=2, padx=4)
        tk.Button(buttons_frame, text="Exportar JSON", command=self.export_json).grid(row=0, column=3, padx=4)
        tk.Button(buttons_frame, text="Salvar EncTerrain", command=self.save_objects_dialog).grid(row=0, column=4, padx=4)
        tk.Button(buttons_frame, text="Resumo", command=self.show_summary).grid(row=0, column=5, padx=4)
        tk.Button(buttons_frame, text="Sair", command=self.root.quit).grid(row=0, column=6, padx=4)

        status_label = tk.Label(self.root, textvariable=self.status_var, anchor="w")
        status_label.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 8))

        self.root.columnconfigure(0, weight=1)
        self.view_mode_var.trace_add("write", self._on_view_mode_change)
        self._on_view_mode_change()

    def choose_data_dir(self) -> None:
        path = filedialog.askdirectory(title="Selecione a pasta Data")
        if path:
            self.data_dir_var.set(path)
            self._populate_worlds(Path(path))

    def choose_object_dir(self) -> None:
        path = filedialog.askdirectory(title="Selecione a pasta ObjectX")
        if path:
            self.object_dir_var.set(path)

    def refresh_worlds(self) -> None:
        data_path = Path(self.data_dir_var.get())
        if not data_path.exists():
            messagebox.showerror("Erro", "Selecione uma pasta Data válida.")
            return
        self._populate_worlds(data_path)

    def _populate_worlds(self, data_path: Path) -> None:
        self.world_options = list_world_directories(data_path)
        menu = self.world_menu["menu"]
        menu.delete(0, "end")
        if not self.world_options:
            self.world_var.set("")
            return
        for world in self.world_options:
            menu.add_command(label=world.name, command=lambda w=world: self._select_world(w))
        self._select_world(self.world_options[0])

    def _select_world(self, world_path: Path) -> None:
        self.world_var.set(world_path.name)
        inferred = infer_map_id(world_path)
        if inferred is not None:
            self.map_id_var.set(str(inferred))
        object_guess = guess_object_folder(world_path)
        if object_guess is not None:
            self.object_dir_var.set(str(object_guess))
        else:
            self.object_dir_var.set("")

    def _current_world_path(self) -> Optional[Path]:
        selected = self.world_var.get()
        if not selected:
            return None
        data_path = Path(self.data_dir_var.get())
        return data_path / selected

    def visualize(self) -> None:
        try:
            self._run(show=True)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Erro", str(exc))

    def save_png(self) -> None:
        try:
            (
                world_path,
                _,
                _,
                _,
                object_dir,
                view_mode,
                overlay,
                scene_focus,
                _,
                _,
                texture_detail,
                renderer,
                fog_density,
                fog_color,
                params,
            ) = self._gather_context()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Erro", str(exc))
            return
        output_path = filedialog.asksaveasfilename(
            title="Salvar visualização",
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
        )
        if not output_path:
            return
        try:
            destination = Path(output_path)
            if self._can_reuse_last(params) and self.last_result is not None:
                title = (
                    f"{world_path.name} (mapa {self.last_result.map_id}) —"
                    f" {len(self.last_result.objects)} objetos"
                )
                texture_library = None
                if renderer == "opengl" or overlay == "textures":
                    texture_library = TextureLibrary(
                        world_path,
                        detail_factor=max(1, texture_detail),
                        object_path=object_dir,
                        map_id=self.last_result.map_id,
                    )
                material_library = None
                bmd_library = None
                if renderer == "opengl":
                    material_roots: List[Path] = []
                    if object_dir:
                        material_roots.append(object_dir)
                    guessed = guess_object_folder(world_path)
                    if guessed:
                        material_roots.append(guessed)
                    material_library = MaterialStateLibrary(world_path, extra_roots=material_roots)
                    search_roots: List[Path] = material_roots.copy()
                    parent = world_path.parent
                    if parent not in search_roots:
                        search_roots.append(parent)
                    bmd_library = BMDLibrary(search_roots, material_library=material_library)
                render_scene(
                    self.last_result.data,
                    self.last_result.objects,
                    output=destination,
                    show=False,
                    title=title,
                    enable_object_edit=False,
                    view_mode=view_mode,
                    overlay=overlay,
                    scene_focus=scene_focus,
                    texture_library=texture_library,
                    material_library=material_library,
                    renderer=renderer,
                    bmd_library=bmd_library,
                    fog_density=fog_density,
                    fog_color=fog_color,
                )
            else:
                self._run(show=False, output=destination)
            messagebox.showinfo("Sucesso", f"PNG salvo em {output_path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Erro", str(exc))

    def export_objects(self) -> None:
        try:
            context = self._gather_context()
            params = context[-1]
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Erro", str(exc))
            return
        output_path = filedialog.asksaveasfilename(
            title="Exportar objetos",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not output_path:
            return
        try:
            destination = Path(output_path)
            if self._can_reuse_last(params) and self.last_result is not None:
                export_objects_csv(self.last_result, destination)
            else:
                self._run(show=False, export_objects=destination, render=False)
            messagebox.showinfo("Sucesso", f"Lista exportada para {output_path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Erro", str(exc))

    def export_json(self) -> None:
        try:
            context = self._gather_context()
            params = context[-1]
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Erro", str(exc))
            return
        output_path = filedialog.asksaveasfilename(
            title="Exportar JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not output_path:
            return
        try:
            destination = Path(output_path)
            if self._can_reuse_last(params) and self.last_result is not None:
                export_result_json(self.last_result, destination)
            else:
                self._run(show=False, export_json=destination, render=False)
            messagebox.showinfo("Sucesso", f"JSON salvo em {output_path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Erro", str(exc))

    def save_objects_dialog(self) -> None:
        try:
            context = self._gather_context()
            params = context[-1]
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Erro", str(exc))
            return
        output_path = filedialog.asksaveasfilename(
            title="Salvar arquivo EncTerrain",
            defaultextension=".obj",
            filetypes=[("EncTerrain", "*.obj")],
        )
        if not output_path:
            return
        destination = Path(output_path)
        try:
            if self._can_reuse_last(params) and self.last_result is not None:
                save_objects_file(self.last_result, destination)
            else:
                self._run(
                    show=False,
                    render=False,
                    save_objects_path=destination,
                )
            messagebox.showinfo("Sucesso", f"EncTerrain salvo em {output_path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Erro", str(exc))

    def show_summary(self) -> None:
        try:
            result = self._run(show=False, render=False)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Erro", str(exc))
            return
        messagebox.showinfo(
            "Resumo do mapa",
            format_detailed_summary(result, object_limit=5, attribute_limit=5),
        )

    def _run(
        self,
        *,
        show: bool,
        output: Optional[Path] = None,
        export_objects: Optional[Path] = None,
        export_json: Optional[Path] = None,
        save_objects_path: Optional[Path] = None,
        render: Optional[bool] = None,
    ) -> TerrainLoadResult:
        (
            world_path,
            map_id,
            height_scale,
            max_objects,
            object_dir,
            view_mode,
            overlay,
            scene_focus,
            include,
            exclude,
            texture_detail,
            renderer,
            fog_density,
            fog_color,
            params,
        ) = self._gather_context()
        include_filters = [include] if include else None
        exclude_filters = [exclude] if exclude else None
        result = run_viewer(
            world_path,
            map_id=map_id,
            object_path=object_dir,
            extended_height=self.extended_height_var.get(),
            height_scale=height_scale,
            output=output,
            show=show,
            max_objects=max_objects,
            enum_path=self.enum_path,
            log_summary=False,
            export_objects=export_objects,
            detailed_summary=False,
            summary_limit=5,
            render=(render if render is not None else (show or output is not None)),
            enable_object_edit=self.edit_objects_var.get(),
            view_mode=view_mode,
            overlay=overlay,
            scene_focus=scene_focus,
            renderer=renderer,
            include_filters=include_filters,
            exclude_filters=exclude_filters,
            export_json=export_json,
            save_objects=save_objects_path,
            texture_detail=texture_detail,
            fog_density=fog_density,
            fog_color=fog_color,
        )
        self.map_id_var.set(str(result.map_id))
        self.status_var.set(format_summary_line(result))
        self.last_result = result
        self.last_params = params
        if (show or output is not None) and result.warnings:
            messagebox.showwarning("Avisos", "\n".join(result.warnings))
        return result

    @staticmethod
    def _parse_float(value: str) -> Optional[float]:
        value = value.strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            raise ValueError("Height scale inválido.")

    @staticmethod
    def _parse_color(value: str) -> Optional[Tuple[float, float, float]]:
        value = value.strip()
        if not value:
            return None
        parts = [part for part in re.split(r"[;,\s]+", value) if part]
        if len(parts) != 3:
            raise ValueError("Cor da névoa deve ter três componentes (R G B).")
        try:
            floats = [float(part) for part in parts]
        except ValueError as exc:  # noqa: B904
            raise ValueError("Valores inválidos para a cor da névoa.") from exc
        return (floats[0], floats[1], floats[2])

    def run(self) -> None:
        self.root.mainloop()


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Visualizador de terreno legado")
    parser.add_argument("world_path", nargs="?", type=Path, help="Pasta Data/WorldX com os arquivos do mapa")
    parser.add_argument("--map-id", type=int, dest="map_id", help="ID numérico usado nos arquivos EncTerrain")
    parser.add_argument(
        "--object-path",
        type=Path,
        dest="object_path",
        help="Diretório ObjectX a ser usado para carregar EncTerrainXX.obj",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        dest="data_root",
        help="Pasta Data contendo subpastas WorldX/ObjectX (usada para a interface gráfica).",
    )
    parser.add_argument("--extended-height", action="store_true", help="Força o parse do formato estendido de altura")
    parser.add_argument("--output", type=Path, help="Arquivo de saída (PNG). Se omitido, não salva.")
    parser.add_argument("--no-show", action="store_true", help="Não abrir a janela interativa do Matplotlib")
    parser.add_argument("--max-objects", type=int, help="Limita quantidade de objetos renderizados")
    parser.add_argument(
        "--height-scale",
        type=float,
        help="Fator de escala aplicado às alturas no formato clássico (padrão 1.5).",
    )
    parser.add_argument("--gui", action="store_true", help="Abre a interface gráfica de seleção de mapas.")
    parser.add_argument(
        "--enum-path",
        type=Path,
        dest="enum_path",
        help="Arquivo _enum.h para nomear tipos de objeto (padrão: source/_enum.h).",
    )
    parser.add_argument(
        "--export-objects",
        type=Path,
        dest="export_objects",
        help="Salva um CSV com a lista completa de objetos posicionados no mapa.",
    )
    parser.add_argument(
        "--export-json",
        type=Path,
        dest="export_json",
        help="Exporta os dados visíveis em JSON para automação externa.",
    )
    parser.add_argument(
        "--edit-objects",
        action="store_true",
        help="Permite mover objetos na visualização interativa (setas do teclado).",
    )
    parser.add_argument(
        "--save-objects",
        type=Path,
        dest="save_objects",
        help="Gera um novo arquivo EncTerrainXX.obj com as posições atuais.",
    )
    parser.add_argument(
        "--detailed-summary",
        action="store_true",
        help="Mostra estatísticas adicionais (altura, atributos e texturas).",
    )
    parser.add_argument(
        "--summary-limit",
        type=int,
        default=8,
        help="Quantidade de entradas exibidas nos resumos de objetos.",
    )
    parser.add_argument(
        "--view-mode",
        choices=["3d", "2d"],
        default="3d",
        help="Define se a visualização será 3D tradicional ou 2D com heatmap.",
    )
    parser.add_argument(
        "--overlay",
        choices=["textures", "height", "attributes"],
        default="textures",
        help="Coloração aplicada ao terreno (texturas, altura ou atributos).",
    )
    parser.add_argument(
        "--renderer",
        choices=["matplotlib", "opengl"],
        default="opengl",
        help="Motor de renderização: Matplotlib clássico ou OpenGL com texturas reais.",
    )
    parser.add_argument(
        "--scene-focus",
        choices=["terrain", "full"],
        default="terrain",
        help="Define se a visualização prioriza apenas terreno e instâncias ou a cena completa.",
    )
    parser.add_argument(
        "--texture-detail",
        type=int,
        default=2,
        help="Subdivisões por tile ao rasterizar texturas reais (>=1).",
    )
    parser.add_argument(
        "--fog-density",
        type=float,
        dest="fog_density",
        help="Densidade da névoa no renderer OpenGL (padrão adaptativo).",
    )
    parser.add_argument(
        "--fog-color",
        nargs=3,
        type=float,
        metavar=("R", "G", "B"),
        dest="fog_color",
        help="Cor da névoa no renderer OpenGL (componentes 0-1).",
    )
    parser.add_argument(
        "--filter",
        dest="include_filters",
        action="append",
        help="Inclui apenas objetos cujo ID ou nome contenha o texto informado.",
    )
    parser.add_argument(
        "--exclude",
        dest="exclude_filters",
        action="append",
        help="Oculta objetos cujo ID ou nome contenha o texto informado.",
    )
    args = parser.parse_args(argv)

    if args.gui or args.world_path is None:
        app = TerrainViewerGUI(initial_data_dir=args.data_root, enum_path=args.enum_path)
        app.run()
        return

    run_viewer(
        args.world_path,
        map_id=args.map_id,
        object_path=args.object_path,
        extended_height=args.extended_height,
        height_scale=args.height_scale,
        output=args.output,
        show=not args.no_show,
        max_objects=args.max_objects,
        enum_path=args.enum_path,
        export_objects=args.export_objects,
        detailed_summary=args.detailed_summary,
        summary_limit=args.summary_limit,
        render=not args.no_show or args.output is not None,
        enable_object_edit=args.edit_objects,
        view_mode=args.view_mode,
        overlay=args.overlay,
        renderer=args.renderer,
        scene_focus=args.scene_focus,
        include_filters=args.include_filters,
        exclude_filters=args.exclude_filters,
        export_json=args.export_json,
        save_objects=args.save_objects,
        texture_detail=max(1, args.texture_detail),
        fog_density=args.fog_density,
        fog_color=tuple(args.fog_color) if args.fog_color else None,
    )


if __name__ == "__main__":
    main()
