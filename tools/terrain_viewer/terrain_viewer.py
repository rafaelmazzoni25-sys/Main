"""Terrain visualization utility.

Ferramenta em Python para visualizar o terreno e objetos estáticos em 3D.
Ela reutiliza os mesmos formatos do cliente legado para gerar uma malha
matricial e sobrepor os objetos com Matplotlib.
"""
from __future__ import annotations

import argparse

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

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


def load_attribute_file(path: Path) -> np.ndarray:
    raw = _read_file(path)
    decrypted = bytearray(map_file_decrypt(raw))
    bux_convert(decrypted)

    if len(decrypted) not in (131_076, 65_540):
        raise ValueError(
            "Tamanho inesperado para arquivo de atributos."
            f" Esperado 65540 ou 131076 bytes, recebi {len(decrypted)}."
        )

    version = decrypted[0]
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
    return attributes.reshape((TERRAIN_SIZE, TERRAIN_SIZE))


def load_mapping_file(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = _read_file(path)
    decrypted = map_file_decrypt(raw)
    ptr = 0
    ptr += 1  # versão
    ptr += 1  # número do mapa

    layer_count = TERRAIN_SIZE * TERRAIN_SIZE
    layer1 = np.frombuffer(decrypted, dtype=np.uint8, count=layer_count, offset=ptr)
    ptr += layer_count
    layer2 = np.frombuffer(decrypted, dtype=np.uint8, count=layer_count, offset=ptr)
    ptr += layer_count
    alpha_bytes = decrypted[ptr : ptr + layer_count]
    alpha = np.frombuffer(alpha_bytes, dtype=np.uint8).astype(np.float32) / 255.0
    return layer1.reshape((TERRAIN_SIZE, TERRAIN_SIZE)), layer2.reshape((TERRAIN_SIZE, TERRAIN_SIZE)), alpha.reshape((TERRAIN_SIZE, TERRAIN_SIZE))


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


def load_objects_file(path: Path) -> List[TerrainObject]:
    raw = _read_file(path)
    decrypted = map_file_decrypt(raw)
    ptr = 0
    ptr += 1  # versão
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
            )
        )
    return objects


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


def render_scene(data: TerrainData, objects: Sequence[TerrainObject], *, output: Optional[Path], show: bool, max_objects: Optional[int]) -> None:
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
        ox = np.array([obj.position[0] / TERRAIN_SCALE for obj in used_objects])
        oy = np.array([obj.position[1] / TERRAIN_SCALE for obj in used_objects])
        oz = np.array([bilinear_height(heights, x, y) for x, y in zip(ox, oy)])
        ax.scatter(ox, oy, oz + 50.0, c="red", s=10, depthshade=False)

    ax.set_xlabel("X (tiles)")
    ax.set_ylabel("Y (tiles)")
    ax.set_zlabel("Altura")
    ax.view_init(elev=60, azim=45)
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


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Visualizador de terreno legado")
    parser.add_argument("world_path", type=Path, help="Pasta Data/WorldX com os arquivos do mapa")
    parser.add_argument("--map-id", type=int, dest="map_id", help="ID numérico usado nos arquivos EncTerrain")
    parser.add_argument(
        "--object-path",
        type=Path,
        dest="object_path",
        help="Diretório ObjectX a ser usado para carregar EncTerrainXX.obj",
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
    args = parser.parse_args(argv)

    world_path = args.world_path
    if not world_path.is_dir():
        raise FileNotFoundError(f"Diretório inválido: {world_path}")

    attributes_path, mapping_path, objects_path, height_path = resolve_files(
        world_path, args.map_id, object_path=args.object_path
    )

    attributes = load_attribute_file(attributes_path)
    layer1, layer2, alpha = load_mapping_file(mapping_path)
    height = load_height_file(
        height_path,
        extended=args.extended_height or height_path.name.endswith("New.OZB"),
        scale_override=args.height_scale,
    )
    objects = load_objects_file(objects_path)

    terrain = TerrainData(
        height=height,
        mapping_layer1=layer1,
        mapping_layer2=layer2,
        mapping_alpha=alpha,
        attributes=attributes,
    )

    render_scene(
        terrain,
        objects,
        output=args.output,
        show=not args.no_show,
        max_objects=args.max_objects,
    )


if __name__ == "__main__":
    main()
