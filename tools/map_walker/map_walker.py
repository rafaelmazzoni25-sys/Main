#!/usr/bin/env python3
"""Ferramenta interativa para navegar pelos mapas do cliente MuOnline.

Este módulo implementa uma janela Qt com um painel de configurações e um
visualizador OpenGL que reconstrói o terreno a partir dos arquivos
`EncTerrain`. É possível escolher diferentes mapas, ajustar velocidade da
câmera, altura do observador e alternar elementos de depuração em tempo
real.
"""

import argparse
import math
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL import GL
from OpenGL.GL import shaders

# Constantes dos atributos
TW_NOMOVE = 0x0004
TW_NOGROUND = 0x0008
TW_HEIGHT = 0x0040

TERRAIN_SIZE = 256
TERRAIN_SCALE = 100.0
MIN_EXTENDED_HEIGHT = -500.0

MAP_NAMES = {
    0: "Lorencia",
    1: "Dungeon",
    2: "Devias",
    3: "Noria",
    4: "Lost Tower",
    5: "Exile",
    6: "Arena",
    7: "Atlans",
    8: "Tarkan",
    9: "Devil Square",
    10: "Icarus",
    11: "Blood Castle 1",
    12: "Blood Castle 2",
    13: "Blood Castle 3",
    14: "Blood Castle 4",
    15: "Blood Castle 5",
    16: "Blood Castle 6",
    17: "Blood Castle 7",
    18: "Chaos Castle 1",
    19: "Chaos Castle 2",
    20: "Chaos Castle 3",
    21: "Chaos Castle 4",
    22: "Chaos Castle 5",
    23: "Chaos Castle 6",
    24: "Kalima 1",
    25: "Kalima 2",
    26: "Kalima 3",
    27: "Kalima 4",
    28: "Kalima 5",
    29: "Kalima 6",
    30: "Battle Castle",
    31: "Hunting Ground",
    33: "Aida",
    34: "Crywolf (Campo)",
    35: "Crywolf (Fortaleza)",
    37: "Kanturu 1",
    38: "Kanturu 2",
    39: "Kanturu 3",
    40: "GM Area",
    41: "Changeup 3rd (1)",
    42: "Changeup 3rd (2)",
    45: "Cursed Temple 1",
    46: "Cursed Temple 2",
    47: "Cursed Temple 3",
    48: "Cursed Temple 4",
    49: "Cursed Temple 5",
    50: "Cursed Temple 6",
    51: "Home (6º personagem)",
    52: "Blood Castle Master",
    53: "Chaos Castle Master",
    54: "Character Scene",
    55: "Login Scene",
    56: "Swamp of Calmness",
    57: "Raklion",
    58: "Raklion Boss",
    62: "Santa Town",
    63: "PK Field",
    64: "Duel Arena",
    65: "Doppelganger 1",
    66: "Doppelganger 2",
    67: "Doppelganger 3",
    68: "Doppelganger 4",
    69: "Empire Guardian 1",
    70: "Empire Guardian 2",
    71: "Empire Guardian 3",
    72: "Empire Guardian 4",
    73: "New Login Scene",
    74: "New Character Scene",
    77: "New Login Scene (Alt)",
    78: "New Character Scene (Alt)",
    79: "Lorencia Market",
    80: "Karutan 1",
    81: "Karutan 2",
}


@dataclass
class MapDescriptor:
    map_index: int
    world_index: int
    display_name: str
    world_path: Path


@dataclass
class ObjectInstance:
    type_id: int
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    scale: float


class TerrainData:
    def __init__(self, descriptor: MapDescriptor, heights: np.ndarray,
                 attributes: Optional[np.ndarray], objects: Sequence[ObjectInstance]):
        self.descriptor = descriptor
        self.heights = heights.astype(np.float32)
        self.attributes = attributes
        self.objects = list(objects)
        self.scale = TERRAIN_SCALE
        self.size = heights.shape[0]
        self.min_height = float(self.heights.min(initial=0.0)) if self.heights.size else 0.0
        self.max_height = float(self.heights.max(initial=0.0)) if self.heights.size else 0.0
        self._normals = self._compute_normals()
        self._positions = self._build_positions()
        self._indices = self._build_indices()

    def _compute_normals(self) -> np.ndarray:
        if self.heights.size == 0:
            return np.zeros((0, 0, 3), dtype=np.float32)
        h = self.heights
        padded = np.pad(h, 1, mode="edge")
        dz = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / (2.0 * self.scale)
        dx = (padded[1:-1, 2:] - padded[1:-1, :-2]) / (2.0 * self.scale)
        normals = np.dstack((-dx, np.ones_like(h), -dz)).astype(np.float32)
        lengths = np.linalg.norm(normals, axis=2, keepdims=True)
        lengths[lengths == 0] = 1.0
        normals /= lengths
        return normals

    def _build_positions(self) -> np.ndarray:
        size = self.size
        positions = np.zeros((size * size, 3), dtype=np.float32)
        normals = np.zeros_like(positions)
        idx = 0
        for y in range(size):
            for x in range(size):
                positions[idx] = (x * self.scale, self.heights[y, x], y * self.scale)
                normals[idx] = self._normals[y, x]
                idx += 1
        self._flat_normals = normals
        return positions

    def _build_indices(self) -> np.ndarray:
        size = self.size
        faces = []
        for y in range(size - 1):
            for x in range(size - 1):
                i0 = y * size + x
                i1 = i0 + 1
                i2 = (y + 1) * size + x
                i3 = i2 + 1
                faces.extend((i0, i2, i1, i1, i2, i3))
        return np.array(faces, dtype=np.uint32)

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @property
    def normals(self) -> np.ndarray:
        return self._flat_normals

    @property
    def indices(self) -> np.ndarray:
        return self._indices

    def world_extent(self) -> float:
        return (self.size - 1) * self.scale

    def sample_height(self, x: float, z: float) -> float:
        if self.heights.size == 0:
            return 0.0
        xf = np.clip(x / self.scale, 0.0, self.size - 1.0001)
        zf = np.clip(z / self.scale, 0.0, self.size - 1.0001)
        xi = int(math.floor(xf))
        zi = int(math.floor(zf))
        xd = xf - xi
        zd = zf - zi
        h00 = self.heights[zi, xi]
        h10 = self.heights[zi, min(xi + 1, self.size - 1)]
        h01 = self.heights[min(zi + 1, self.size - 1), xi]
        h11 = self.heights[min(zi + 1, self.size - 1), min(xi + 1, self.size - 1)]
        h0 = h00 * (1 - xd) + h10 * xd
        h1 = h01 * (1 - xd) + h11 * xd
        return h0 * (1 - zd) + h1 * zd

    def attribute_at(self, x: float, z: float) -> Optional[int]:
        if self.attributes is None:
            return None
        xf = int(x / self.scale)
        zf = int(z / self.scale)
        if xf < 0 or zf < 0 or xf >= self.size or zf >= self.size:
            return None
        return int(self.attributes[zf, xf])

    def is_walkable(self, x: float, z: float) -> bool:
        attr = self.attribute_at(x, z)
        if attr is None:
            return True
        if attr & (TW_NOMOVE | TW_NOGROUND):
            return False
        if attr & TW_HEIGHT:
            return False
        return True


def map_file_decrypt(data: bytes) -> bytes:
    key = [0xD1, 0x73, 0x52, 0xF6, 0xD2, 0x9A, 0xCB, 0x27,
           0x3E, 0xAF, 0x59, 0x31, 0x37, 0xB3, 0xE7, 0xA2]
    result = bytearray(len(data))
    w_key = 0x5E
    for i, value in enumerate(data):
        result[i] = (value ^ key[i % 16]) - w_key & 0xFF
        w_key = (value + 0x3D) & 0xFF
    return bytes(result)


def bux_convert(buffer: bytearray) -> None:
    code = (0xFC, 0xCF, 0xAB)
    for i in range(len(buffer)):
        buffer[i] ^= code[i % 3]


def resolve_data_root(path: Optional[str]) -> Optional[Path]:
    if path is None:
        return None
    candidate = Path(path).expanduser().resolve()
    if (candidate / "World1").exists():
        return candidate
    if (candidate / "Data").exists():
        nested = candidate / "Data"
        if (nested / "World1").exists():
            return nested
    return None


class TerrainRepository:
    def __init__(self) -> None:
        self._data_root: Optional[Path] = None

    @property
    def data_root(self) -> Optional[Path]:
        return self._data_root

    def set_data_root(self, path: Optional[str]) -> Optional[Path]:
        resolved = resolve_data_root(path)
        self._data_root = resolved
        return resolved

    def list_maps(self) -> List[MapDescriptor]:
        if self._data_root is None:
            return []
        descriptors: List[MapDescriptor] = []
        for folder in sorted(self._data_root.glob("World*")):
            digits = ''.join(ch for ch in folder.name if ch.isdigit())
            if not digits:
                continue
            world_index = int(digits)
            map_index = world_index - 1
            name = MAP_NAMES.get(map_index, f"World {map_index}")
            descriptors.append(MapDescriptor(map_index, world_index, name, folder))
        return descriptors

    def load_map(self, descriptor: MapDescriptor, variant: str) -> TerrainData:
        if self._data_root is None:
            raise RuntimeError("Pasta de dados não configurada.")
        heights = self._load_heights(descriptor)
        attributes = self._load_attributes(descriptor, variant)
        objects = self._load_objects(descriptor)
        return TerrainData(descriptor, heights, attributes, objects)

    def _load_heights(self, descriptor: MapDescriptor) -> np.ndarray:
        world_path = descriptor.world_path
        classic = world_path / "TerrainHeight.OZB"
        extended = world_path / "TerrainHeightNew.OZB"
        bmp = world_path / "TerrainHeight.bmp"
        if extended.exists():
            return self._read_extended(extended)
        if classic.exists():
            scale = 3.0 if descriptor.map_index in {55, 73, 77} else 1.5
            return self._read_classic(classic, scale)
        if bmp.exists():
            return self._read_bmp(bmp)
        raise FileNotFoundError(f"Arquivos de altura não encontrados em {world_path}.")

    def _read_classic(self, path: Path, scale: float) -> np.ndarray:
        data = path.read_bytes()
        offset = 4 + 1080
        expected = offset + TERRAIN_SIZE * TERRAIN_SIZE
        if len(data) < expected:
            raise ValueError(f"Arquivo {path} muito pequeno para o formato clássico.")
        payload = data[offset:offset + TERRAIN_SIZE * TERRAIN_SIZE]
        height = np.frombuffer(payload, dtype=np.uint8, count=TERRAIN_SIZE * TERRAIN_SIZE)
        height = height.reshape((TERRAIN_SIZE, TERRAIN_SIZE)).astype(np.float32)
        return height * scale

    def _read_extended(self, path: Path) -> np.ndarray:
        data = path.read_bytes()
        offset = 4 + 54
        expected = offset + TERRAIN_SIZE * TERRAIN_SIZE * 3
        if len(data) < expected:
            raise ValueError(f"Arquivo {path} muito pequeno para o formato estendido.")
        payload = data[offset:offset + TERRAIN_SIZE * TERRAIN_SIZE * 3]
        values = np.frombuffer(payload, dtype=np.uint8)
        values = values.reshape((TERRAIN_SIZE, TERRAIN_SIZE, 3))
        r = values[:, :, 2].astype(np.uint32)
        g = values[:, :, 1].astype(np.uint32)
        b = values[:, :, 0].astype(np.uint32)
        height = (r | (g << 8) | (b << 16)).astype(np.float32)
        height += MIN_EXTENDED_HEIGHT
        return height

    def _read_bmp(self, path: Path) -> np.ndarray:
        with path.open('rb') as fp:
            header = fp.read(54)
            if len(header) != 54:
                raise ValueError(f"Cabeçalho inválido em {path}")
            width = struct.unpack_from('<I', header, 18)[0]
            height = struct.unpack_from('<I', header, 22)[0]
            if width != TERRAIN_SIZE or height != TERRAIN_SIZE:
                raise ValueError("Dimensão inesperada no BMP de terreno.")
            payload = fp.read(TERRAIN_SIZE * TERRAIN_SIZE)
        data = np.frombuffer(payload, dtype=np.uint8)
        data = data.reshape((TERRAIN_SIZE, TERRAIN_SIZE)).astype(np.float32)
        return data * 1.5

    def _attribute_candidates(self, descriptor: MapDescriptor, variant: str) -> Iterable[int]:
        base = descriptor.world_index
        if variant == 'event1':
            return [base * 10 + 1]
        if variant == 'event2':
            return [base * 10 + 2]
        if variant == 'base':
            return [base]
        return [base, base * 10 + 1, base * 10 + 2]

    def _load_attributes(self, descriptor: MapDescriptor, variant: str) -> Optional[np.ndarray]:
        world_path = descriptor.world_path
        for candidate in self._attribute_candidates(descriptor, variant):
            name = f"EncTerrain{candidate}.att"
            path = world_path / name
            if path.exists():
                return self._read_attribute(path)
        return None

    def _read_attribute(self, path: Path) -> np.ndarray:
        raw = path.read_bytes()
        decrypted = map_file_decrypt(raw)
        buffer = bytearray(decrypted)
        bux_convert(buffer)
        payload = buffer[4:]
        if len(payload) == TERRAIN_SIZE * TERRAIN_SIZE:
            data = np.frombuffer(payload, dtype=np.uint8)
            return data.reshape((TERRAIN_SIZE, TERRAIN_SIZE)).astype(np.uint16)
        if len(payload) == TERRAIN_SIZE * TERRAIN_SIZE * 2:
            data = np.frombuffer(payload, dtype='<u2')
            return data.reshape((TERRAIN_SIZE, TERRAIN_SIZE))
        raise ValueError(f"Arquivo de atributo com tamanho inesperado: {path}")

    def _load_objects(self, descriptor: MapDescriptor) -> List[ObjectInstance]:
        path = descriptor.world_path / f"EncTerrain{descriptor.world_index}.obj"
        if not path.exists():
            return []
        raw = path.read_bytes()
        data = map_file_decrypt(raw)
        if len(data) < 4:
            return []
        count = struct.unpack_from('<H', data, 2)[0]
        offset = 4
        objects: List[ObjectInstance] = []
        for _ in range(count):
            if offset + 2 + 12 + 12 + 4 > len(data):
                break
            type_id = struct.unpack_from('<h', data, offset)[0]
            offset += 2
            position = struct.unpack_from('<3f', data, offset)
            offset += 12
            rotation = struct.unpack_from('<3f', data, offset)
            offset += 12
            scale = struct.unpack_from('<f', data, offset)[0]
            offset += 4
            objects.append(ObjectInstance(type_id, position, rotation, scale))
        return objects


class ControlPanel(QtWidgets.QWidget):
    dataDirectoryChanged = QtCore.Signal(str)
    mapChanged = QtCore.Signal(int)
    attributeVariantChanged = QtCore.Signal(str)
    eyeHeightChanged = QtCore.Signal(float)
    moveSpeedChanged = QtCore.Signal(float)
    baseColorChanged = QtCore.Signal(float, float, float)
    lockToGroundChanged = QtCore.Signal(bool)
    showGridChanged = QtCore.Signal(bool)
    showObjectsChanged = QtCore.Signal(bool)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._descriptors: List[MapDescriptor] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.path_edit = QtWidgets.QLineEdit(self)
        browse_btn = QtWidgets.QPushButton("Escolher pasta de dados", self)
        browse_btn.clicked.connect(self._browse_data)

        layout.addWidget(QtWidgets.QLabel("Pasta de dados"))
        layout.addWidget(self.path_edit)
        layout.addWidget(browse_btn)

        self.map_combo = QtWidgets.QComboBox(self)
        self.map_combo.currentIndexChanged.connect(self._map_changed)
        layout.addWidget(QtWidgets.QLabel("Mapa"))
        layout.addWidget(self.map_combo)

        self.variant_combo = QtWidgets.QComboBox(self)
        self.variant_combo.addItem("Auto", userData='auto')
        self.variant_combo.addItem("Padrão", userData='base')
        self.variant_combo.addItem("Evento x10+1", userData='event1')
        self.variant_combo.addItem("Evento x10+2", userData='event2')
        self.variant_combo.currentIndexChanged.connect(self._variant_changed)
        layout.addWidget(QtWidgets.QLabel("Variante de atributo"))
        layout.addWidget(self.variant_combo)

        self.eye_spin = QtWidgets.QDoubleSpinBox(self)
        self.eye_spin.setRange(10.0, 500.0)
        self.eye_spin.setValue(120.0)
        self.eye_spin.setSuffix(" u")
        self.eye_spin.valueChanged.connect(self.eyeHeightChanged)
        layout.addWidget(QtWidgets.QLabel("Altura da câmera"))
        layout.addWidget(self.eye_spin)

        self.speed_spin = QtWidgets.QDoubleSpinBox(self)
        self.speed_spin.setRange(50.0, 5000.0)
        self.speed_spin.setValue(800.0)
        self.speed_spin.setSuffix(" u/s")
        self.speed_spin.valueChanged.connect(self.moveSpeedChanged)
        layout.addWidget(QtWidgets.QLabel("Velocidade"))
        layout.addWidget(self.speed_spin)

        color_layout = QtWidgets.QGridLayout()
        self.color_spins: List[QtWidgets.QDoubleSpinBox] = []
        for i, channel in enumerate("RGB"):
            spin = QtWidgets.QDoubleSpinBox(self)
            spin.setRange(0.05, 1.0)
            spin.setSingleStep(0.05)
            spin.setValue(0.6 if channel == 'G' else 0.45)
            spin.valueChanged.connect(self._color_changed)
            self.color_spins.append(spin)
            color_layout.addWidget(QtWidgets.QLabel(channel), 0, i)
            color_layout.addWidget(spin, 1, i)
        color_group = QtWidgets.QGroupBox("Cor base do terreno", self)
        color_group.setLayout(color_layout)
        layout.addWidget(color_group)

        self.lock_check = QtWidgets.QCheckBox("Travar câmera ao solo", self)
        self.lock_check.setChecked(True)
        self.lock_check.toggled.connect(self.lockToGroundChanged)
        layout.addWidget(self.lock_check)

        self.grid_check = QtWidgets.QCheckBox("Mostrar grade", self)
        self.grid_check.setChecked(True)
        self.grid_check.toggled.connect(self.showGridChanged)
        layout.addWidget(self.grid_check)

        self.object_check = QtWidgets.QCheckBox("Mostrar objetos", self)
        self.object_check.setChecked(True)
        self.object_check.toggled.connect(self.showObjectsChanged)
        layout.addWidget(self.object_check)

        layout.addStretch(1)

        self.path_edit.editingFinished.connect(self._path_changed)

    def _path_changed(self) -> None:
        self.dataDirectoryChanged.emit(self.path_edit.text())

    def _browse_data(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Selecione a pasta que contém 'World1'")
        if directory:
            self.path_edit.setText(directory)
            self.dataDirectoryChanged.emit(directory)

    def _map_changed(self, index: int) -> None:
        if index < 0 or index >= len(self._descriptors):
            return
        self.mapChanged.emit(index)

    def _variant_changed(self, index: int) -> None:
        variant = self.variant_combo.itemData(index)
        if variant:
            self.attributeVariantChanged.emit(variant)

    def _color_changed(self) -> None:
        r, g, b = (spin.value() for spin in self.color_spins)
        self.baseColorChanged.emit(r, g, b)

    def set_data_path(self, path: Optional[Path]) -> None:
        if path is None:
            return
        self.path_edit.setText(str(path))

    def set_maps(self, descriptors: Sequence[MapDescriptor]) -> None:
        self._descriptors = list(descriptors)
        self.map_combo.blockSignals(True)
        self.map_combo.clear()
        for desc in self._descriptors:
            label = f"[{desc.map_index:02d}] {desc.display_name}"
            self.map_combo.addItem(label)
        self.map_combo.blockSignals(False)
        if self._descriptors:
            self.map_combo.setCurrentIndex(0)

    def current_descriptor(self) -> Optional[MapDescriptor]:
        index = self.map_combo.currentIndex()
        if 0 <= index < len(self._descriptors):
            return self._descriptors[index]
        return None

    def set_lock_state(self, locked: bool) -> None:
        if self.lock_check.isChecked() != locked:
            self.lock_check.blockSignals(True)
            self.lock_check.setChecked(locked)
            self.lock_check.blockSignals(False)

    def set_grid_visible(self, visible: bool) -> None:
        if self.grid_check.isChecked() != visible:
            self.grid_check.blockSignals(True)
            self.grid_check.setChecked(visible)
            self.grid_check.blockSignals(False)

    def set_objects_visible(self, visible: bool) -> None:
        if self.object_check.isChecked() != visible:
            self.object_check.blockSignals(True)
            self.object_check.setChecked(visible)
            self.object_check.blockSignals(False)

    def current_variant(self) -> str:
        return self.variant_combo.currentData()

    def current_eye_height(self) -> float:
        return self.eye_spin.value()

    def current_speed(self) -> float:
        return self.speed_spin.value()

    def current_color(self) -> Tuple[float, float, float]:
        return tuple(spin.value() for spin in self.color_spins)


class TerrainView(QOpenGLWidget):
    lockStateChanged = QtCore.Signal(bool)
    gridVisibilityChanged = QtCore.Signal(bool)
    objectVisibilityChanged = QtCore.Signal(bool)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._terrain: Optional[TerrainData] = None
        self._eye_height = 120.0
        self._base_color = np.array([0.45, 0.6, 0.45], dtype=np.float32)
        self._move_speed = 800.0
        self._lock_to_ground = True
        self._show_grid = True
        self._show_objects = True
        self._keys: set[int] = set()
        self._mouse_pos = QtCore.QPoint()
        self._rotate_active = False
        self._yaw = math.radians(-90)
        self._pitch = math.radians(-20)
        self._camera_pos = np.array([0.0, 300.0, 0.0], dtype=np.float32)
        self._last_time = time.perf_counter()
        self._terrain_vbo = 0
        self._terrain_nbo = 0
        self._terrain_ibo = 0
        self._terrain_index_count = 0
        self._terrain_program = None
        self._grid_program = None
        self._grid_vbo = 0
        self._grid_vertex_count = 0
        self._objects_vbo = 0
        self._objects_vertex_count = 0

    def set_eye_height(self, value: float) -> None:
        self._eye_height = float(value)

    def set_move_speed(self, value: float) -> None:
        self._move_speed = float(value)

    def set_base_color(self, r: float, g: float, b: float) -> None:
        self._base_color = np.array([r, g, b], dtype=np.float32)

    def set_lock_to_ground(self, locked: bool) -> None:
        self._lock_to_ground = bool(locked)
        self.lockStateChanged.emit(self._lock_to_ground)

    def set_show_grid(self, visible: bool) -> None:
        self._show_grid = bool(visible)
        self.gridVisibilityChanged.emit(self._show_grid)

    def set_show_objects(self, visible: bool) -> None:
        self._show_objects = bool(visible)
        self.objectVisibilityChanged.emit(self._show_objects)

    def terrain(self) -> Optional[TerrainData]:
        return self._terrain

    def camera_position(self) -> Tuple[float, float, float]:
        return tuple(float(v) for v in self._camera_pos)

    def initializeGL(self) -> None:
        self._init_programs()
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_BACK)
        GL.glClearColor(0.05, 0.07, 0.10, 1.0)

    def resizeGL(self, width: int, height: int) -> None:
        GL.glViewport(0, 0, width, max(1, height))

    def paintGL(self) -> None:
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if self._terrain_program is None or self._terrain_vbo == 0:
            return
        projection = self._projection_matrix()
        view = self._view_matrix()
        mvp = projection @ view
        normal_matrix = np.identity(3, dtype=np.float32)
        GL.glUseProgram(self._terrain_program)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self._terrain_program, "u_mvp"), 1, GL.GL_FALSE, mvp)
        GL.glUniformMatrix3fv(GL.glGetUniformLocation(self._terrain_program, "u_normal"), 1, GL.GL_FALSE, normal_matrix)
        GL.glUniform3fv(GL.glGetUniformLocation(self._terrain_program, "u_base_color"), 1, self._base_color)
        GL.glUniform3f(GL.glGetUniformLocation(self._terrain_program, "u_light_direction"), -0.4, 0.8, -0.3)
        GL.glUniform1f(GL.glGetUniformLocation(self._terrain_program, "u_ambient"), 0.25)

        pos_loc = GL.glGetAttribLocation(self._terrain_program, "a_position")
        norm_loc = GL.glGetAttribLocation(self._terrain_program, "a_normal")
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._terrain_vbo)
        GL.glEnableVertexAttribArray(pos_loc)
        GL.glVertexAttribPointer(pos_loc, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._terrain_nbo)
        GL.glEnableVertexAttribArray(norm_loc)
        GL.glVertexAttribPointer(norm_loc, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._terrain_ibo)
        GL.glDrawElements(GL.GL_TRIANGLES, self._terrain_index_count, GL.GL_UNSIGNED_INT, None)
        GL.glDisableVertexAttribArray(pos_loc)
        GL.glDisableVertexAttribArray(norm_loc)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)
        GL.glUseProgram(0)

        if self._show_grid and self._grid_program and self._grid_vbo:
            GL.glUseProgram(self._grid_program)
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(self._grid_program, "u_mvp"), 1, GL.GL_FALSE, mvp)
            GL.glUniform3f(GL.glGetUniformLocation(self._grid_program, "u_color"), 0.15, 0.4, 0.6)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._grid_vbo)
            pos_loc = GL.glGetAttribLocation(self._grid_program, "a_position")
            GL.glEnableVertexAttribArray(pos_loc)
            GL.glVertexAttribPointer(pos_loc, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
            GL.glDrawArrays(GL.GL_LINES, 0, self._grid_vertex_count)
            GL.glDisableVertexAttribArray(pos_loc)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            GL.glUseProgram(0)

        if self._show_objects and self._objects_vbo:
            GL.glUseProgram(self._grid_program)
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(self._grid_program, "u_mvp"), 1, GL.GL_FALSE, mvp)
            GL.glUniform3f(GL.glGetUniformLocation(self._grid_program, "u_color"), 0.9, 0.3, 0.2)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._objects_vbo)
            pos_loc = GL.glGetAttribLocation(self._grid_program, "a_position")
            GL.glEnableVertexAttribArray(pos_loc)
            GL.glVertexAttribPointer(pos_loc, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
            GL.glDrawArrays(GL.GL_LINES, 0, self._objects_vertex_count)
            GL.glDisableVertexAttribArray(pos_loc)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            GL.glUseProgram(0)

    def set_terrain(self, data: Optional[TerrainData]) -> None:
        self.makeCurrent()
        self._terrain = data
        self._release_buffers()
        if data is None:
            self.doneCurrent()
            return
        self._upload_terrain(data)
        self._upload_grid(data)
        self._upload_objects(data)
        center = data.world_extent() * 0.5
        self._camera_pos = np.array([center, data.sample_height(center, center) + self._eye_height, center], dtype=np.float32)
        self.doneCurrent()

    def _upload_terrain(self, data: TerrainData) -> None:
        vertices = data.positions.astype(np.float32)
        normals = data.normals.astype(np.float32)
        indices = data.indices.astype(np.uint32)
        self._terrain_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._terrain_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)
        self._terrain_nbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._terrain_nbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, normals.nbytes, normals, GL.GL_STATIC_DRAW)
        self._terrain_ibo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._terrain_ibo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL.GL_STATIC_DRAW)
        self._terrain_index_count = indices.size
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

    def _upload_grid(self, data: TerrainData) -> None:
        extent = data.world_extent()
        step = int(TERRAIN_SIZE / 16)
        positions: List[Tuple[float, float, float]] = []
        base_height = data.min_height - 5.0
        for i in range(0, TERRAIN_SIZE, step):
            x = i * TERRAIN_SCALE
            positions.append((x, base_height, 0.0))
            positions.append((x, base_height, extent))
            positions.append((0.0, base_height, x))
            positions.append((extent, base_height, x))
        if positions:
            array = np.array(positions, dtype=np.float32)
            self._grid_vbo = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._grid_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, array.nbytes, array, GL.GL_STATIC_DRAW)
            self._grid_vertex_count = len(positions)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        else:
            self._grid_vbo = 0
            self._grid_vertex_count = 0

    def _upload_objects(self, data: TerrainData) -> None:
        if not data.objects:
            self._objects_vbo = 0
            self._objects_vertex_count = 0
            return
        vertices: List[Tuple[float, float, float]] = []
        for obj in data.objects:
            x, y, z = obj.position
            vertices.append((x, y, z))
            vertices.append((x, y + 150.0, z))
        array = np.array(vertices, dtype=np.float32)
        self._objects_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._objects_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, array.nbytes, array, GL.GL_STATIC_DRAW)
        self._objects_vertex_count = len(vertices)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def _release_buffers(self) -> None:
        if self._terrain_vbo:
            GL.glDeleteBuffers(1, [self._terrain_vbo])
            self._terrain_vbo = 0
        if self._terrain_nbo:
            GL.glDeleteBuffers(1, [self._terrain_nbo])
            self._terrain_nbo = 0
        if self._terrain_ibo:
            GL.glDeleteBuffers(1, [self._terrain_ibo])
            self._terrain_ibo = 0
        if self._grid_vbo:
            GL.glDeleteBuffers(1, [self._grid_vbo])
            self._grid_vbo = 0
        if self._objects_vbo:
            GL.glDeleteBuffers(1, [self._objects_vbo])
            self._objects_vbo = 0

    def _init_programs(self) -> None:
        terrain_vs = """
            #version 120
            attribute vec3 a_position;
            attribute vec3 a_normal;
            uniform mat4 u_mvp;
            uniform mat3 u_normal;
            varying float v_light;
            void main() {
                vec3 n = normalize(u_normal * a_normal);
                vec3 lightDir = normalize(vec3(0.4, 0.8, 0.3));
                v_light = max(dot(n, lightDir), 0.1);
                gl_Position = u_mvp * vec4(a_position, 1.0);
            }
        """
        terrain_fs = """
            #version 120
            uniform vec3 u_base_color;
            uniform float u_ambient;
            uniform vec3 u_light_direction;
            varying float v_light;
            void main() {
                float diffuse = max(v_light, u_ambient);
                gl_FragColor = vec4(u_base_color * diffuse, 1.0);
            }
        """
        grid_vs = """
            #version 120
            attribute vec3 a_position;
            uniform mat4 u_mvp;
            void main() {
                gl_Position = u_mvp * vec4(a_position, 1.0);
            }
        """
        grid_fs = """
            #version 120
            uniform vec3 u_color;
            void main() {
                gl_FragColor = vec4(u_color, 1.0);
            }
        """
        self._terrain_program = shaders.compileProgram(
            shaders.compileShader(terrain_vs, GL.GL_VERTEX_SHADER),
            shaders.compileShader(terrain_fs, GL.GL_FRAGMENT_SHADER),
        )
        self._grid_program = shaders.compileProgram(
            shaders.compileShader(grid_vs, GL.GL_VERTEX_SHADER),
            shaders.compileShader(grid_fs, GL.GL_FRAGMENT_SHADER),
        )

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.isAutoRepeat():
            return
        key = event.key()
        if key == QtCore.Qt.Key_Space:
            self.set_lock_to_ground(not self._lock_to_ground)
            return
        if key == QtCore.Qt.Key_G:
            self.set_show_grid(not self._show_grid)
            return
        if key == QtCore.Qt.Key_O:
            self.set_show_objects(not self._show_objects)
            return
        self._keys.add(key)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.isAutoRepeat():
            return
        key = event.key()
        self._keys.discard(key)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.RightButton:
            self._rotate_active = True
            self._mouse_pos = event.pos()
            self.setCursor(QtCore.Qt.BlankCursor)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._rotate_active:
            delta = event.pos() - self._mouse_pos
            sensitivity = 0.005
            self._yaw += delta.x() * sensitivity
            self._pitch += delta.y() * sensitivity
            self._pitch = max(math.radians(-89), min(math.radians(89), self._pitch))
            self._mouse_pos = event.pos()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.RightButton and self._rotate_active:
            self._rotate_active = False
            self.unsetCursor()

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        if self._rotate_active:
            self._rotate_active = False
            self.unsetCursor()

    def tick(self) -> None:
        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now
        if self._terrain is None or dt <= 0:
            return
        speed = self._move_speed
        if QtCore.Qt.Key_Shift in self._keys:
            speed *= 1.8
        move_vec = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        forward = np.array([math.cos(self._yaw), 0.0, math.sin(self._yaw)], dtype=np.float32)
        right = np.array([-math.sin(self._yaw), 0.0, math.cos(self._yaw)], dtype=np.float32)
        if QtCore.Qt.Key_W in self._keys:
            move_vec += forward
        if QtCore.Qt.Key_S in self._keys:
            move_vec -= forward
        if QtCore.Qt.Key_A in self._keys:
            move_vec -= right
        if QtCore.Qt.Key_D in self._keys:
            move_vec += right
        if np.linalg.norm(move_vec) > 0:
            move_vec = move_vec / np.linalg.norm(move_vec)
        displacement = move_vec * speed * dt
        new_pos = self._camera_pos + displacement
        extent = self._terrain.world_extent()
        new_pos[0] = np.clip(new_pos[0], 0.0, extent)
        new_pos[2] = np.clip(new_pos[2], 0.0, extent)
        if self._terrain.is_walkable(new_pos[0], new_pos[2]):
            if self._lock_to_ground:
                ground = self._terrain.sample_height(new_pos[0], new_pos[2])
                new_pos[1] = ground + self._eye_height
            else:
                if QtCore.Qt.Key_Q in self._keys:
                    new_pos[1] -= speed * 0.5 * dt
                if QtCore.Qt.Key_E in self._keys:
                    new_pos[1] += speed * 0.5 * dt
            self._camera_pos = new_pos
        self.update()

    def _projection_matrix(self) -> np.ndarray:
        aspect = self.width() / max(1.0, float(self.height()))
        fov = math.radians(60.0)
        near = 50.0
        far = 120000.0
        f = 1.0 / math.tan(fov / 2.0)
        mat = np.zeros((4, 4), dtype=np.float32)
        mat[0, 0] = f / aspect
        mat[1, 1] = f
        mat[2, 2] = (far + near) / (near - far)
        mat[2, 3] = (2.0 * far * near) / (near - far)
        mat[3, 2] = -1.0
        return mat

    def _view_matrix(self) -> np.ndarray:
        direction = np.array([
            math.cos(self._pitch) * math.cos(self._yaw),
            math.sin(self._pitch),
            math.cos(self._pitch) * math.sin(self._yaw),
        ], dtype=np.float32)
        center = self._camera_pos + direction
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        f = center - self._camera_pos
        f = f / np.linalg.norm(f)
        s = np.cross(f, up)
        s = s / np.linalg.norm(s)
        u = np.cross(s, f)
        view = np.identity(4, dtype=np.float32)
        view[0, :3] = s
        view[1, :3] = u
        view[2, :3] = -f
        view[:3, 3] = -view[:3, :3] @ self._camera_pos
        return view


class MapWalkerWindow(QtWidgets.QMainWindow):
    def __init__(self, repository: TerrainRepository, initial_path: Optional[str] = None) -> None:
        super().__init__()
        self.setWindowTitle("MU Map Walker")
        self.resize(1280, 720)
        self.repository = repository
        self.view = TerrainView(self)
        self.panel = ControlPanel(self)
        central = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.panel)
        layout.addWidget(self.view, 1)
        self.setCentralWidget(central)

        self.status_label = QtWidgets.QLabel("Nenhum mapa carregado", self)
        self.statusBar().addPermanentWidget(self.status_label, 1)

        self.panel.dataDirectoryChanged.connect(self._on_data_directory)
        self.panel.mapChanged.connect(self._on_map_selected)
        self.panel.attributeVariantChanged.connect(self._reload_current_map)
        self.panel.eyeHeightChanged.connect(self.view.set_eye_height)
        self.panel.moveSpeedChanged.connect(self.view.set_move_speed)
        self.panel.baseColorChanged.connect(self.view.set_base_color)
        self.panel.lockToGroundChanged.connect(self.view.set_lock_to_ground)
        self.panel.showGridChanged.connect(self.view.set_show_grid)
        self.panel.showObjectsChanged.connect(self.view.set_show_objects)

        self.view.lockStateChanged.connect(self.panel.set_lock_state)
        self.view.gridVisibilityChanged.connect(self.panel.set_grid_visible)
        self.view.objectVisibilityChanged.connect(self.panel.set_objects_visible)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)

        if initial_path:
            resolved = self.repository.set_data_root(initial_path)
            if resolved is not None:
                self.panel.set_data_path(resolved)
                self._refresh_map_list()

    def _tick(self) -> None:
        self.view.tick()
        self._update_status()

    def _on_data_directory(self, path: str) -> None:
        resolved = self.repository.set_data_root(path)
        if resolved is None:
            QtWidgets.QMessageBox.warning(self, "Dados", "Não foi possível localizar a pasta 'World1' no caminho informado.")
            return
        self._refresh_map_list()

    def _refresh_map_list(self) -> None:
        descriptors = self.repository.list_maps()
        if not descriptors:
            QtWidgets.QMessageBox.information(self, "Dados", "Nenhum mapa encontrado na pasta selecionada.")
        self.panel.set_maps(descriptors)
        if descriptors:
            self._load_descriptor(descriptors[0])

    def _on_map_selected(self, index: int) -> None:
        descriptor = self.panel.current_descriptor()
        if descriptor:
            self._load_descriptor(descriptor)

    def _reload_current_map(self, variant: str) -> None:
        descriptor = self.panel.current_descriptor()
        if descriptor:
            self._load_descriptor(descriptor)

    def _load_descriptor(self, descriptor: MapDescriptor) -> None:
        try:
            variant = self.panel.current_variant()
            terrain = self.repository.load_map(descriptor, variant)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Erro ao carregar mapa", str(exc))
            return
        self.view.set_terrain(terrain)
        self.statusBar().showMessage(f"Mapa carregado: {descriptor.display_name} (World{descriptor.world_index})")

    def _update_status(self) -> None:
        terrain = self.view.terrain()
        if terrain is None:
            self.status_label.setText("Nenhum mapa carregado")
            return
        x, y, z = self.view.camera_position()
        attr = terrain.attribute_at(x, z)
        attr_text = f"0x{attr:04X}" if attr is not None else "-"
        self.status_label.setText(
            f"Pos: ({x:8.1f}, {y:8.1f}, {z:8.1f})  |  Altura terreno: {terrain.sample_height(x, z):.1f}  |  Attr: {attr_text}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualizador interativo de mapas do MuOnline")
    parser.add_argument("--data", help="Caminho para a pasta Data ou para seu diretório pai", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    QtCore.QCoreApplication.setOrganizationName("ProjectRebirthMu")
    QtCore.QCoreApplication.setApplicationName("Map Walker")
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    app = QtWidgets.QApplication(sys.argv)
    repo = TerrainRepository()
    window = MapWalkerWindow(repo, initial_path=args.data)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
