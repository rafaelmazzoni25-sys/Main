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
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

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
    from pyglet.window import key as pyglet_key
except Exception:  # noqa: BLE001
    pyglet = None  # type: ignore[assignment]
    pyglet_key = None  # type: ignore[assignment]

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
    bone_indices: Optional[np.ndarray] = None


@dataclass
class BMDKeyframe:
    translation: np.ndarray
    rotation: np.ndarray
    time: float


@dataclass
class BMDAnimationChannel:
    bone_index: int
    keyframes: List[BMDKeyframe] = field(default_factory=list)


@dataclass
class BMDAnimation:
    name: str
    duration: float
    frames_per_second: float
    channels: Dict[int, BMDAnimationChannel] = field(default_factory=dict)


@dataclass
class BMDBone:
    name: str
    parent: int
    rest_translation: np.ndarray
    rest_rotation: np.ndarray
    rest_matrix: np.ndarray
    inverse_bind_matrix: np.ndarray


@dataclass
class BMDModel:
    name: str
    meshes: List[BMDMesh] = field(default_factory=list)
    version: int = 0
    bones: List[BMDBone] = field(default_factory=list)
    animations: Dict[str, BMDAnimation] = field(default_factory=dict)

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
    ) -> None:
        self.world_path = world_path
        self.detail_factor = max(1, detail_factor)
        self.object_path = object_path
        self._image_cache: Dict[int, Optional[np.ndarray]] = {}
        self._resized_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._missing: set[int] = set()
        self._fallback_cmap = cm.get_cmap("tab20")
        self.search_roots = self._build_search_roots()

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
        shading = LightSource(azdeg=315, altdeg=55).shade(refined_heights, vert_exag=1.0, fraction=0.6)
        facecolors[..., :3] *= np.clip(shading[:-1, :-1, :], 0.0, 1.0)
        facecolors = np.clip(facecolors, 0.0, 1.0)
        return xx, yy, refined_heights, facecolors

    @property
    def missing_indices(self) -> Sequence[int]:
        return sorted(self._missing)


def compute_tile_material_flags(
    layer1: np.ndarray, layer2: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    flags = np.zeros_like(layer1, dtype=np.uint32)
    if layer1.size == 0:
        return flags
    for tile_ids, flag in ((WATER_TILE_IDS, MATERIAL_WATER), (LAVA_TILE_IDS, MATERIAL_LAVA)):
        if tile_ids:
            mask1 = np.isin(layer1, list(tile_ids))
            mask2 = (alpha > 0.01) & np.isin(layer2, list(tile_ids))
            flags[mask1] |= flag
            flags[mask2] |= flag
    if TRANSPARENT_TILE_IDS:
        mask1 = np.isin(layer1, list(TRANSPARENT_TILE_IDS))
        mask2 = (alpha > 0.01) & np.isin(layer2, list(TRANSPARENT_TILE_IDS))
        flags[mask1] |= MATERIAL_TRANSPARENT
        flags[mask2] |= MATERIAL_TRANSPARENT
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


def _classify_mesh_material(texture_name: str) -> int:
    lowered = texture_name.lower()
    flags = 0
    if any(token in lowered for token in ("water", "river", "wave", "ocean", "sea")):
        flags |= MATERIAL_WATER | MATERIAL_TRANSPARENT
    if any(token in lowered for token in ("lava", "magma", "volcano", "fire")):
        flags |= MATERIAL_LAVA | MATERIAL_TRANSPARENT
    if any(token in lowered for token in ("alpha", "glass", "trans", "smoke", "light", "flare")):
        flags |= MATERIAL_TRANSPARENT
    return flags


def _lerp_vec(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + (b - a) * t


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


def load_bmd_model(path: Path) -> BMDModel:
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
            meshes.append(
                BMDMesh(
                    name=f"{name}_mesh{mesh_index}",
                    positions=positions,
                    normals=normals,
                    texcoords=uvs,
                    indices=indices,
                    texture_name=texture_name,
                    material_flags=_classify_mesh_material(texture_name),
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

            bone = BMDBone(
                name=bone_name or f"bone_{len(bones)}",
                parent=parent,
                rest_translation=rest_translation,
                rest_rotation=rest_rotation,
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
                        )
                    )
                if channel.keyframes:
                    animations[info.name].channels[bone_index] = channel

        rest_globals: List[np.ndarray] = []
        for idx, bone in enumerate(bones):
            local = _compose_transform_matrix(bone.rest_translation, bone.rest_rotation)
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
    return model


class BMDLibrary:
    def __init__(self, search_roots: Sequence[Path]) -> None:
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
                self._cache[path] = load_bmd_model(path)
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
    ) -> None:
        detail = texture_library.detail_factor
        heights = _upsample_height_map(data.height, detail)
        tile_flags = compute_tile_material_flags(data.mapping_layer1, data.mapping_layer2, data.mapping_alpha)
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
        spacing = float(TERRAIN_SCALE) / max(1, detail)
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
        self.texture = None
        self.bone_indices = (
            mesh.bone_indices.astype(np.int32, copy=True) if mesh.bone_indices is not None else None
        )
        self.has_skinning = bool(self.bone_indices is not None and np.any(self.bone_indices >= 0))
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

    def update_pose(self, bone_matrices: Sequence[np.ndarray]) -> None:
        if not self.has_skinning or not bone_matrices:
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


class BMDAnimationPlayer:
    def __init__(self, model: BMDModel, *, loop: bool = True) -> None:
        self.model = model
        self.loop = loop
        self.active: Optional[BMDAnimation] = None
        self.current_time = 0.0
        self._pose_cache: List[np.ndarray] = []
        self._dirty = True
        if model.animations:
            first = next(iter(model.animations))
            self.set_animation(first)

    def set_animation(self, name: Optional[str]) -> None:
        if not name:
            self.active = None
            self.current_time = 0.0
            self._dirty = True
            return
        if name not in self.model.animations:
            return
        self.active = self.model.animations[name]
        self.current_time = 0.0
        self._dirty = True

    def update(self, delta: float) -> None:
        if self.active is None:
            return
        if self.active.duration > 0.0:
            self.current_time += delta
            if self.loop:
                self.current_time = math.fmod(self.current_time, self.active.duration)
                if self.current_time < 0.0:
                    self.current_time += self.active.duration
            else:
                self.current_time = min(self.current_time, self.active.duration)
        else:
            self.current_time = 0.0
        self._dirty = True

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
            self._pose_cache = [pose_matrix @ bone.inverse_bind_matrix for pose_matrix, bone in zip(pose, self.model.bones)]
            self._dirty = False
            return self._pose_cache

        local_matrices: List[np.ndarray] = []
        for index, bone in enumerate(self.model.bones):
            if index in animation.channels:
                translation, rotation = self._sample_channel(animation.channels[index], animation)
            else:
                translation, rotation = bone.rest_translation, bone.rest_rotation
            local_matrices.append(_compose_transform_matrix(translation, rotation))

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

        self._pose_cache = [matrix.astype(np.float32) for matrix in skinning]
        self._dirty = False
        return self._pose_cache

    def _sample_channel(
        self, channel: BMDAnimationChannel, animation: BMDAnimation
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not channel.keyframes:
            zero = np.zeros(3, dtype=np.float32)
            return zero, zero
        if len(channel.keyframes) == 1:
            frame = channel.keyframes[0]
            return frame.translation, frame.rotation
        duration = max(animation.duration, channel.keyframes[-1].time)
        time_value = self.current_time
        if duration > 0.0 and self.loop:
            cycles = time_value / duration
            frac = cycles - math.floor(cycles)
            time_value = frac * duration
        prev = channel.keyframes[0]
        for frame in channel.keyframes[1:]:
            if time_value <= frame.time:
                span = frame.time - prev.time
                if span <= 0.0:
                    return frame.translation, frame.rotation
                t = (time_value - prev.time) / span
                translation = _lerp_vec(prev.translation, frame.translation, t)
                rotation = _lerp_vec(prev.rotation, frame.rotation, t)
                return translation, rotation
            prev = frame
        return channel.keyframes[-1].translation, channel.keyframes[-1].rotation


@dataclass
class BMDInstance:
    model_matrix: np.ndarray
    mesh_renderers: List[_BMDMeshRenderer]
    animation_player: Optional[BMDAnimationPlayer] = None


class OpenGLTerrainApp:
    def __init__(
        self,
        data: TerrainData,
        objects: Sequence[TerrainObject],
        *,
        texture_library: TextureLibrary,
        bmd_library: Optional[BMDLibrary],
        overlay: str,
        title: str,
        fog_color: Tuple[float, float, float] = (0.25, 0.33, 0.45),
        fog_density: float = 0.00025,
    ) -> None:
        self.data = data
        self.objects = list(objects)
        self.texture_library = texture_library
        self.bmd_library = bmd_library
        self.overlay = overlay
        self.title = title
        self.fog_color = np.array(fog_color, dtype=np.float32)
        self.fog_density = fog_density
        self.window_size = (1280, 720)
        self.ctx: Optional["moderngl.Context"] = None
        self.window: Optional["pyglet.window.Window"] = None
        self.terrain: Optional[_TerrainBuffers] = None
        self.terrain_program: Optional["moderngl.Program"] = None
        self.terrain_specular_program: Optional["moderngl.Program"] = None
        self.object_program: Optional["moderngl.Program"] = None
        self.object_specular_program: Optional["moderngl.Program"] = None
        self.sky_program: Optional["moderngl.Program"] = None
        self.particle_program: Optional["moderngl.Program"] = None
        self.camera: Optional[OrbitCamera] = None
        self._pressed_keys: set[int] = set()
        self._start_time = 0.0
        self._last_frame_time = 0.0
        self._object_mesh_cache: Dict[str, List[_BMDMeshRenderer]] = {}
        self._particle_vbo: Optional["moderngl.Buffer"] = None
        self._particle_count = 0
        self._particle_vao: Optional["moderngl.VertexArray"] = None
        self.object_instances: List[BMDInstance] = []

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
        self.terrain_program = self.ctx.program(
            vertex_shader=terrain_vertex_shader,
            fragment_shader=textwrap.dedent(
                """
                #version 330
                flat in int v_material;
                in vec3 v_normal;
                in vec3 v_world_pos;
                in vec2 v_uv;
                uniform sampler2D u_texture;
                uniform vec3 u_light_dir;
                uniform vec3 u_fog_color;
                uniform float u_fog_density;
                uniform float u_time;
                uniform vec3 u_camera_pos;
                out vec4 frag_color;
                void main() {
                    vec2 uv = v_uv;
                    float wave = 0.0;
                    if ((v_material & 1) != 0) {
                        vec2 flow = vec2(sin(u_time * 0.35), cos(u_time * 0.4)) * 0.035;
                        uv += flow;
                        wave += sin(u_time * 1.6 + v_world_pos.x * 0.002 + v_world_pos.z * 0.002) * 0.08;
                    }
                    if ((v_material & 2) != 0) {
                        float lava = sin(u_time * 2.0 + v_world_pos.x * 0.003) * 0.06;
                        uv += vec2(0.0, lava);
                        wave += sin(u_time * 3.1 + v_world_pos.z * 0.004) * 0.12;
                    }
                    vec4 color = texture(u_texture, uv);
                    vec3 normal = normalize(v_normal);
                    if (wave != 0.0) {
                        vec3 wave_normal = normalize(vec3(normal.x + wave, normal.y, normal.z + wave));
                        normal = mix(normal, wave_normal, 0.6);
                    }
                    float diff = max(dot(normal, normalize(-u_light_dir)), 0.0);
                    vec3 ambient = color.rgb * 0.25;
                    vec3 diffuse = color.rgb * diff * 0.75;
                    vec3 lighting = ambient + diffuse;
                    if ((v_material & 1) != 0) {
                        lighting *= vec3(0.9, 1.05, 1.1);
                    }
                    if ((v_material & 2) != 0) {
                        lighting *= vec3(1.4, 0.7, 0.5);
                    }
                    float distance = length(v_world_pos - u_camera_pos);
                    float fog = clamp(exp(-u_fog_density * distance), 0.0, 1.0);
                    vec3 final_color = mix(u_fog_color, lighting, fog);
                    frag_color = vec4(final_color, color.a);
                }
                """
            ),
        )

        self.terrain_specular_program = self.ctx.program(
            vertex_shader=terrain_vertex_shader,
            fragment_shader=textwrap.dedent(
                """
                #version 330
                flat in int v_material;
                in vec3 v_normal;
                in vec3 v_world_pos;
                in vec2 v_uv;
                uniform sampler2D u_texture;
                uniform vec3 u_light_dir;
                uniform vec3 u_camera_pos;
                uniform float u_time;
                out vec4 frag_color;
                void main() {
                    vec2 uv = v_uv;
                    if ((v_material & 1) != 0) {
                        uv += vec2(sin(u_time * 0.5) * 0.025, cos(u_time * 0.45) * 0.02);
                    }
                    if ((v_material & 2) != 0) {
                        uv += vec2(0.0, sin(u_time * 2.2 + v_world_pos.x * 0.004) * 0.08);
                    }
                    vec3 base = texture(u_texture, uv).rgb;
                    vec3 normal = normalize(v_normal);
                    vec3 light_dir = normalize(-u_light_dir);
                    vec3 view_dir = normalize(u_camera_pos - v_world_pos);
                    vec3 half_vec = normalize(light_dir + view_dir);
                    float shininess = ((v_material & 1) != 0) ? 24.0 : 16.0;
                    if ((v_material & 2) != 0) {
                        shininess = 12.0;
                    }
                    float spec = pow(max(dot(normal, half_vec), 0.0), shininess);
                    float boost = ((v_material & 1) != 0) ? 1.3 : 0.7;
                    if ((v_material & 2) != 0) {
                        boost = 1.6;
                    }
                    frag_color = vec4(base * spec * boost, 1.0);
                }
                """
            ),
        )

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
        self.object_program = self.ctx.program(
            vertex_shader=object_vertex_shader,
            fragment_shader=textwrap.dedent(
                """
                #version 330
                in vec3 v_normal;
                in vec3 v_world_pos;
                in vec2 v_uv;
                uniform sampler2D u_texture;
                uniform vec3 u_light_dir;
                uniform vec3 u_fog_color;
                uniform float u_fog_density;
                uniform vec3 u_camera_pos;
                uniform float u_time;
                uniform int u_material_flags;
                out vec4 frag_color;
                void main() {
                    vec2 uv = v_uv;
                    float wave = 0.0;
                    if ((u_material_flags & 1) != 0) {
                        vec2 flow = vec2(cos(u_time * 0.4), sin(u_time * 0.45)) * 0.035;
                        uv += flow;
                        wave += sin(u_time * 1.9 + v_world_pos.x * 0.003) * 0.09;
                    }
                    if ((u_material_flags & 2) != 0) {
                        float lava = sin(u_time * 2.5 + v_world_pos.x * 0.005) * 0.08;
                        uv += vec2(0.0, lava);
                        wave += sin(u_time * 3.5 + v_world_pos.z * 0.004) * 0.15;
                    }
                    vec4 color = texture(u_texture, uv);
                    if ((u_material_flags & 4) != 0) {
                        color.a = min(color.a + 0.2, 1.0);
                    }
                    vec3 normal = normalize(v_normal);
                    if (wave != 0.0) {
                        vec3 wave_normal = normalize(vec3(normal.x + wave, normal.y, normal.z + wave));
                        normal = mix(normal, wave_normal, 0.6);
                    }
                    float diff = max(dot(normal, normalize(-u_light_dir)), 0.0);
                    vec3 ambient = color.rgb * 0.28;
                    vec3 diffuse = color.rgb * diff * 0.72;
                    vec3 lighting = ambient + diffuse;
                    float distance = length(v_world_pos - u_camera_pos);
                    float fog = clamp(exp(-u_fog_density * distance), 0.0, 1.0);
                    vec3 final_color = mix(u_fog_color, lighting, fog);
                    frag_color = vec4(final_color, color.a);
                }
                """
            ),
        )

        self.object_specular_program = self.ctx.program(
            vertex_shader=object_vertex_shader,
            fragment_shader=textwrap.dedent(
                """
                #version 330
                in vec3 v_normal;
                in vec3 v_world_pos;
                in vec2 v_uv;
                uniform sampler2D u_texture;
                uniform vec3 u_light_dir;
                uniform vec3 u_camera_pos;
                uniform float u_time;
                uniform int u_material_flags;
                out vec4 frag_color;
                void main() {
                    vec2 uv = v_uv;
                    if ((u_material_flags & 1) != 0) {
                        uv += vec2(sin(u_time * 0.5) * 0.03, cos(u_time * 0.55) * 0.03);
                    }
                    if ((u_material_flags & 2) != 0) {
                        uv += vec2(0.0, sin(u_time * 2.4 + v_world_pos.x * 0.006) * 0.08);
                    }
                    vec3 base = texture(u_texture, uv).rgb;
                    vec3 normal = normalize(v_normal);
                    vec3 light_dir = normalize(-u_light_dir);
                    vec3 view_dir = normalize(u_camera_pos - v_world_pos);
                    vec3 half_vec = normalize(light_dir + view_dir);
                    float shininess = ((u_material_flags & 1) != 0) ? 30.0 : 22.0;
                    if ((u_material_flags & 2) != 0) {
                        shininess = 14.0;
                    }
                    float spec = pow(max(dot(normal, half_vec), 0.0), shininess);
                    float boost = ((u_material_flags & 1) != 0) ? 1.5 : 0.8;
                    if ((u_material_flags & 2) != 0) {
                        boost = 1.9;
                    }
                    frag_color = vec4(base * spec * boost, 1.0);
                }
                """
            ),
        )

        self.sky_program = self.ctx.program(
            vertex_shader=textwrap.dedent(
                """
                #version 330
                out vec2 v_pos;
                const vec2 positions[3] = vec2[](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
                void main() {
                    v_pos = positions[gl_VertexID];
                    gl_Position = vec4(v_pos, 0.999, 1.0);
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
                    float dist = length(gl_PointCoord - vec2(0.5));
                    float alpha = smoothstep(0.5, 0.0, dist) * v_alpha;
                    frag_color = vec4(u_fog_color, alpha);
                }
                """
            ),
        )

    def _build_particles(self) -> None:
        assert self.ctx is not None
        count = 400
        width = (TERRAIN_SIZE - 1) * TERRAIN_SCALE
        rng = np.random.default_rng(42)
        particle_data = np.zeros((count, 7), dtype=np.float32)
        particle_data[:, 0] = rng.uniform(0.0, width, count)
        particle_data[:, 2] = rng.uniform(0.0, width, count)
        particle_data[:, 1] = rng.uniform(800.0, 2200.0, count)
        particle_data[:, 3:6] = rng.uniform(-15.0, 15.0, (count, 3))
        particle_data[:, 6] = rng.uniform(0.0, 12.0, count)
        self._particle_vbo = self.ctx.buffer(particle_data.astype("f4").tobytes())
        self._particle_count = count
        self._particle_vao = self.ctx.vertex_array(
            self.particle_program,
            [
                (self._particle_vbo, "3f 3f 1f", "in_position", "in_velocity", "in_birth"),
            ],
        )

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
                    renderers.append(renderer)
                self._object_mesh_cache[cache_key] = renderers
            render_meshes = self._object_mesh_cache[cache_key]
            animation_player = BMDAnimationPlayer(model) if model.animations else None
            if animation_player:
                bone_matrices = animation_player.pose_matrices()
                for renderer in render_meshes:
                    renderer.update_pose(bone_matrices)
            instances.append(
                BMDInstance(
                    model_matrix=_compose_model_matrix(obj),
                    mesh_renderers=render_meshes,
                    animation_player=animation_player,
                )
            )
        return instances

    def _setup(self) -> None:
        if moderngl is None or pyglet is None:
            raise RuntimeError("O renderer OpenGL requer as dependências 'moderngl' e 'pyglet'.")
        config = None
        if pyglet:
            try:
                config = pyglet.gl.Config(sample_buffers=1, samples=4, depth_size=24, double_buffer=True)
            except Exception:  # noqa: BLE001
                config = None
        self.window = pyglet.window.Window(
            width=self.window_size[0],
            height=self.window_size[1],
            caption=self.title,
            resizable=True,
            config=config,
            visible=False,
        )
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self._init_programs()
        self.terrain = _TerrainBuffers(
            self.ctx,
            self.terrain_program,
            self.terrain_specular_program,
            self.data,
            self.texture_library,
            self.overlay,
        )
        self.object_instances = self._load_objects()
        self._build_particles()
        center = np.array(
            [
                (TERRAIN_SIZE - 1) * TERRAIN_SCALE / 2.0,
                float(np.max(self.data.height)) + 1000.0,
                (TERRAIN_SIZE - 1) * TERRAIN_SCALE / 2.0,
            ],
            dtype=np.float32,
        )
        self.camera = OrbitCamera(center)
        self._start_time = time.perf_counter()
        self._last_frame_time = self._start_time
        self.window.set_visible(True)
        self._bind_events()

    def _bind_events(self) -> None:
        assert self.window is not None

        @self.window.event
        def on_draw() -> None:  # noqa: ANN001
            self.render_frame()

        @self.window.event
        def on_close() -> None:  # noqa: ANN001
            pyglet.app.exit()

        @self.window.event
        def on_key_press(symbol: int, _modifiers: int) -> None:
            if symbol == pyglet_key.ESCAPE:
                pyglet.app.exit()
                return
            self._pressed_keys.add(symbol)

        @self.window.event
        def on_key_release(symbol: int, _modifiers: int) -> None:
            if symbol in self._pressed_keys:
                self._pressed_keys.remove(symbol)

        @self.window.event
        def on_mouse_scroll(_x: int, _y: int, _dx: float, dy: float) -> None:
            if self.camera:
                self.camera.zoom(-dy * 400.0)

        def _update(dt: float) -> None:
            if self.camera:
                self.camera.update(self._pressed_keys, dt)

        pyglet.clock.schedule_interval(_update, 1 / 60.0)

    def _render_sky(self, time_value: float) -> None:
        assert self.ctx is not None and self.sky_program is not None
        self.ctx.disable(moderngl.DEPTH_TEST)
        cycle = (math.sin(time_value * 0.05) + 1.0) * 0.5
        day_top = np.array([0.32, 0.45, 0.72], dtype=np.float32)
        night_top = np.array([0.05, 0.08, 0.18], dtype=np.float32)
        top = day_top * (1.0 - cycle) + night_top * cycle
        bottom = self.fog_color * (0.7 + 0.3 * cycle)
        self.sky_program["u_color_top"].value = tuple(np.clip(top, 0.0, 1.0).tolist())
        self.sky_program["u_color_bottom"].value = tuple(np.clip(bottom, 0.0, 1.0).tolist())
        self.ctx.screen.use()
        self.sky_program.run(vertices=3)
        self.ctx.enable(moderngl.DEPTH_TEST)

    def _render_particles(self, view: np.ndarray, projection: np.ndarray, time_value: float) -> None:
        if self._particle_vbo is None or self._particle_count == 0 or self._particle_vao is None:
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
        self._last_frame_time = current_time
        time_value = current_time - self._start_time
        if self.window is not None:
            width, height = self.window.get_framebuffer_size()
        else:
            width, height = self.window_size
        aspect = width / float(max(height, 1))
        projection = _perspective_matrix(60.0, aspect, 10.0, 60000.0)
        view = self.camera.view_matrix()
        eye = self.camera.position
        for instance in self.object_instances:
            if instance.animation_player is not None:
                instance.animation_player.update(delta_time)
                bone_matrices = instance.animation_player.pose_matrices()
                for renderer in instance.mesh_renderers:
                    renderer.update_pose(bone_matrices)
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.screen.clear(*self.fog_color.tolist(), 1.0)
        self._render_sky(time_value)

        terrain = self.terrain
        assert terrain is not None
        terrain.texture.use(location=0)
        model_identity = np.eye(4, dtype=np.float32)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.terrain_program["u_model"].write(model_identity.tobytes())
        self.terrain_program["u_view"].write(view.astype("f4").tobytes())
        self.terrain_program["u_projection"].write(projection.astype("f4").tobytes())
        self.terrain_program["u_texture"].value = 0
        self.terrain_program["u_light_dir"].value = (-0.35, -1.0, -0.45)
        self.terrain_program["u_fog_color"].value = tuple(self.fog_color.tolist())
        self.terrain_program["u_fog_density"].value = self.fog_density
        self.terrain_program["u_time"].value = time_value
        self.terrain_program["u_camera_pos"].value = tuple(eye.tolist())
        terrain.vao_diffuse.render()

        if self.terrain_specular_program is not None and terrain.vao_specular is not None:
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.ONE, moderngl.ONE
            self.terrain_specular_program["u_model"].write(model_identity.tobytes())
            self.terrain_specular_program["u_view"].write(view.astype("f4").tobytes())
            self.terrain_specular_program["u_projection"].write(projection.astype("f4").tobytes())
            self.terrain_specular_program["u_texture"].value = 0
            self.terrain_specular_program["u_light_dir"].value = (-0.35, -1.0, -0.45)
            self.terrain_specular_program["u_camera_pos"].value = tuple(eye.tolist())
            self.terrain_specular_program["u_time"].value = time_value
            terrain.vao_specular.render()
            self.ctx.disable(moderngl.BLEND)

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        for instance in self.object_instances:
            model_bytes = instance.model_matrix.astype(np.float32).tobytes()
            self.object_program["u_model"].write(model_bytes)
            self.object_program["u_view"].write(view.astype("f4").tobytes())
            self.object_program["u_projection"].write(projection.astype("f4").tobytes())
            self.object_program["u_texture"].value = 0
            self.object_program["u_light_dir"].value = (-0.35, -1.0, -0.45)
            self.object_program["u_fog_color"].value = tuple(self.fog_color.tolist())
            self.object_program["u_fog_density"].value = self.fog_density
            self.object_program["u_time"].value = time_value
            self.object_program["u_camera_pos"].value = tuple(eye.tolist())
            for mesh in instance.mesh_renderers:
                if mesh.texture is not None:
                    mesh.texture.use(location=0)
                self.object_program["u_material_flags"].value = mesh.material_flags
                mesh.vao_diffuse.render()
            if self.object_specular_program is not None:
                self.ctx.blend_func = moderngl.ONE, moderngl.ONE
                self.object_specular_program["u_model"].write(model_bytes)
                self.object_specular_program["u_view"].write(view.astype("f4").tobytes())
                self.object_specular_program["u_projection"].write(projection.astype("f4").tobytes())
                self.object_specular_program["u_texture"].value = 0
                self.object_specular_program["u_light_dir"].value = (-0.35, -1.0, -0.45)
                self.object_specular_program["u_camera_pos"].value = tuple(eye.tolist())
                self.object_specular_program["u_time"].value = time_value
                for mesh in instance.mesh_renderers:
                    if mesh.vao_specular is None:
                        continue
                    if mesh.texture is not None:
                        mesh.texture.use(location=0)
                    self.object_specular_program["u_material_flags"].value = mesh.material_flags
                    mesh.vao_specular.render()
                self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.disable(moderngl.BLEND)

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
        self._setup()
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
    renderer: str = "matplotlib",
    bmd_library: Optional[BMDLibrary] = None,
    fog_color: Optional[Tuple[float, float, float]] = None,
    fog_density: Optional[float] = None,
) -> None:
    if view_mode == "2d":
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
            bmd_library=bmd_library,
            overlay=overlay,
            title=title or "Visualização OpenGL",
            fog_color=fog_color or (0.25, 0.33, 0.45),
            fog_density=fog_density or 0.00025,
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
        facecolors = base_colors[:-1, :-1, :]
        shading = LightSource(azdeg=315, altdeg=55).shade(
            render_heights, vert_exag=1.0, fraction=0.6
        )
        facecolors[..., :3] *= np.clip(shading[:-1, :-1, :], 0.0, 1.0)
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

        ax.set_xlabel("X (tiles)")
        ax.set_ylabel("Y (tiles)")
        ax.set_zlabel("Altura")
        ax.view_init(elev=60, azim=45)
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
    include_filters: Optional[Sequence[str]] = None,
    exclude_filters: Optional[Sequence[str]] = None,
    export_json: Optional[Path] = None,
    save_objects: Optional[Path] = None,
    texture_detail: int = 2,
    renderer: str = "matplotlib",
    fog_density: Optional[float] = None,
    fog_color: Optional[Tuple[float, float, float]] = None,
) -> TerrainLoadResult:
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

    if enable_object_edit and not show:
        raise ValueError(
            "A edição de objetos requer a janela interativa. Remova --no-show para usar esta opção."
        )

    if enable_object_edit and view_mode != "3d":
        raise ValueError("A movimentação de objetos só está disponível no modo 3D.")

    if render:
        texture_library: Optional[TextureLibrary] = None
        if renderer == "opengl":
            texture_library = TextureLibrary(
                display_result.world_path,
                detail_factor=max(1, texture_detail),
                object_path=object_path,
            )
        elif view_mode == "3d" and overlay == "textures":
            texture_library = TextureLibrary(
                display_result.world_path,
                detail_factor=max(1, texture_detail),
                object_path=object_path,
            )

        bmd_library: Optional[BMDLibrary] = None
        if renderer == "opengl":
            search_roots: List[Path] = []
            if object_path:
                search_roots.append(object_path)
            guessed = guess_object_folder(world_path)
            if guessed:
                search_roots.append(guessed)
            parent = world_path.parent
            if parent not in search_roots:
                search_roots.append(parent)
            bmd_library = BMDLibrary(search_roots)

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
            renderer=renderer,
            bmd_library=bmd_library,
            fog_density=fog_density,
            fog_color=fog_color,
        )
        if texture_library is not None and texture_library.missing_indices:
            preview = ", ".join(map(str, texture_library.missing_indices[:10]))
            if len(texture_library.missing_indices) > 10:
                preview += ", ..."
            print(
                "Aviso: não foi possível localizar todas as texturas. Índices ausentes:",
                preview,
            )

    if truncated:
        print(
            "Aviso: limite de objetos aplicado. Apenas"
            f" {len(display_result.objects)} de {len(filtered_objects)} objetos foram renderizados."
        )

    if export_objects is not None:
        export_objects_csv(display_result, export_objects)
        print(f"Objetos exportados em {export_objects}")

    if export_json is not None:
        export_result_json(display_result, export_json)
        print(f"Dados exportados em {export_json}")

    if save_objects is not None:
        save_objects_file(result, save_objects)
        print(f"Arquivo EncTerrain salvo em {save_objects}")

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

        tk.Label(options_frame, text="Névoa densidade:").grid(row=3, column=0, sticky="w")
        tk.Entry(options_frame, textvariable=self.fog_density_var, width=10).grid(
            row=3, column=1, sticky="w"
        )

        tk.Label(options_frame, text="Cor névoa (R,G,B):").grid(row=3, column=2, sticky="w")
        tk.Entry(options_frame, textvariable=self.fog_color_var, width=20).grid(
            row=3, column=3, columnspan=3, sticky="ew"
        )

        tk.Label(options_frame, text="Mostrar apenas (ID/nome):").grid(row=4, column=0, sticky="w")
        tk.Entry(options_frame, textvariable=self.include_filter_var, width=20).grid(row=4, column=1, sticky="ew")

        tk.Label(options_frame, text="Ocultar (ID/nome):").grid(row=4, column=2, sticky="w")
        tk.Entry(options_frame, textvariable=self.exclude_filter_var, width=20).grid(row=4, column=3, sticky="ew")

        self.edit_checkbox = tk.Checkbutton(
            options_frame,
            text="Permitir mover objetos (janela interativa)",
            variable=self.edit_objects_var,
        )
        self.edit_checkbox.grid(row=5, column=0, columnspan=4, sticky="w")

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
                    )
                bmd_library = None
                if renderer == "opengl":
                    search_roots: List[Path] = []
                    if object_dir:
                        search_roots.append(object_dir)
                    guessed = guess_object_folder(world_path)
                    if guessed:
                        search_roots.append(guessed)
                    parent = world_path.parent
                    if parent not in search_roots:
                        search_roots.append(parent)
                    bmd_library = BMDLibrary(search_roots)
                render_scene(
                    self.last_result.data,
                    self.last_result.objects,
                    output=destination,
                    show=False,
                    title=title,
                    enable_object_edit=False,
                    view_mode=view_mode,
                    overlay=overlay,
                    texture_library=texture_library,
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
