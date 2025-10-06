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
import math
import re
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

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
class TerrainLoadResult:
    world_path: Path
    data: TerrainData
    objects: List[TerrainObject]
    map_id: int
    map_id_attribute: int
    map_id_mapping: int
    map_id_objects: int
    model_names: Mapping[int, str]


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


def open_objects_enc(path: Path, model_names: Mapping[int, str]) -> Tuple[int, List[TerrainObject]]:
    raw = _read_file(path)
    decrypted = map_file_decrypt(raw)
    ptr = 0
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
    return int(map_id), objects


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


def render_scene(
    data: TerrainData,
    objects: Sequence[TerrainObject],
    *,
    output: Optional[Path],
    show: bool,
    max_objects: Optional[int],
    title: Optional[str] = None,
) -> None:
    x = np.arange(TERRAIN_SIZE)
    y = np.arange(TERRAIN_SIZE)
    xx, yy = np.meshgrid(x, y)
    heights = data.height

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap("terrain")
    face_colors = cmap((data.mapping_layer1.astype(np.float32) / 255.0))

    ax.plot_surface(
        xx,
        yy,
        heights,
        rstride=2,
        cstride=2,
        facecolors=face_colors,
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    if objects:
        used_objects = objects[:max_objects] if max_objects is not None else objects
        ox = np.array([obj.tile_position[0] for obj in used_objects])
        oy = np.array([obj.tile_position[1] for obj in used_objects])
        oz = np.array([bilinear_height(heights, x, y) for x, y in zip(ox, oy)])
        type_ids = np.array([obj.type_id for obj in used_objects], dtype=float)
        if type_ids.size > 0 and type_ids.ptp() > 0:
            colors = (type_ids - type_ids.min()) / type_ids.ptp()
        else:
            colors = np.zeros_like(type_ids)
        ax.scatter(ox, oy, oz + 50.0, c=colors, cmap="tab20", s=10, depthshade=False)

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

    height = world_path / "TerrainHeight.OZB"
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
    obj_map_id, objects = open_objects_enc(objects_path, model_names)
    height = load_height_file(
        height_path,
        extended=extended_height or height_path.name.endswith("New.OZB"),
        scale_override=height_scale,
    )

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
        f"{len(result.objects)} objetos",
    ]
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
    result: TerrainLoadResult, destination: Path, *, include_tile_coords: bool = True
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
        for obj in result.objects:
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
) -> TerrainLoadResult:
    result = load_world_data(
        world_path,
        map_id=map_id,
        object_path=object_path,
        extended_height=extended_height,
        height_scale=height_scale,
        enum_path=enum_path,
    )

    if render:
        render_scene(
            result.data,
            result.objects,
            output=output,
            show=show,
            max_objects=max_objects,
            title=f"{world_path.name} (mapa {result.map_id}) — {len(result.objects)} objetos",
        )

    if export_objects is not None:
        export_objects_csv(result, export_objects)
        print(f"Objetos exportados em {export_objects}")

    if log_summary:
        if detailed_summary:
            print_detailed_summary(result, object_limit=summary_limit)
        else:
            print_summary(result, limit=summary_limit)
    return result


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

        self.object_dir_var = tk.StringVar()
        self.status_var = tk.StringVar()

        self.world_options: List[Path] = []
        self.enum_path = None
        if enum_path and enum_path.exists():
            self.enum_path = enum_path
        elif DEFAULT_ENUM_PATH.exists():
            self.enum_path = DEFAULT_ENUM_PATH

        self.last_result: Optional[TerrainLoadResult] = None

        if initial_data_dir and initial_data_dir.is_dir():
            self.data_dir_var.set(str(initial_data_dir))
            self._populate_worlds(initial_data_dir)

        self._build_layout()

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

        buttons_frame = tk.Frame(self.root)
        buttons_frame.grid(row=2, column=0, sticky="e", **padding)

        tk.Button(buttons_frame, text="Visualizar", command=self.visualize).grid(row=0, column=0, padx=4)
        tk.Button(buttons_frame, text="Salvar PNG", command=self.save_png).grid(row=0, column=1, padx=4)
        tk.Button(buttons_frame, text="Exportar objetos", command=self.export_objects).grid(row=0, column=2, padx=4)
        tk.Button(buttons_frame, text="Resumo", command=self.show_summary).grid(row=0, column=3, padx=4)
        tk.Button(buttons_frame, text="Sair", command=self.root.quit).grid(row=0, column=4, padx=4)

        status_label = tk.Label(self.root, textvariable=self.status_var, anchor="w")
        status_label.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 8))

        self.root.columnconfigure(0, weight=1)

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
        output_path = filedialog.asksaveasfilename(
            title="Salvar visualização",
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
        )
        if not output_path:
            return
        try:
            self._run(show=False, output=Path(output_path))
            messagebox.showinfo("Sucesso", f"PNG salvo em {output_path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Erro", str(exc))

    def export_objects(self) -> None:
        output_path = filedialog.asksaveasfilename(
            title="Exportar objetos",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not output_path:
            return
        try:
            self._run(show=False, export_objects=Path(output_path), render=False)
            messagebox.showinfo("Sucesso", f"Lista exportada para {output_path}")
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
        render: Optional[bool] = None,
    ) -> TerrainLoadResult:
        world_path = self._current_world_path()
        if world_path is None:
            raise ValueError("Selecione uma pasta World válida.")
        map_id = _safe_int(self.map_id_var.get())
        height_scale = self._parse_float(self.height_scale_var.get())
        max_objects = _safe_int(self.max_objects_var.get())
        object_dir = Path(self.object_dir_var.get()) if self.object_dir_var.get() else None
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
        )
        self.map_id_var.set(str(result.map_id))
        self.status_var.set(format_summary_line(result))
        self.last_result = result
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
    )


if __name__ == "__main__":
    main()
