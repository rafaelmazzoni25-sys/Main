#!/usr/bin/env python3
"""Mini jogo para navegar pelos mapas utilizando a lógica do projeto Main."""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import List, Optional, Sequence, Tuple

CONFIG_NAME = "teste_mapa_config.json"
WORLD_PATTERN = re.compile(r"^world\d+$", re.IGNORECASE)


@functools.lru_cache()
def _load_terrain_module() -> ModuleType:
    """Carrega o módulo ``terrain_viewer`` como biblioteca compartilhada."""

    module_path = Path(__file__).resolve().parents[1] / "terrain_viewer" / "terrain_viewer.py"
    if not module_path.exists():
        raise FileNotFoundError(
            "Não foi possível localizar tools/terrain_viewer/terrain_viewer.py. "
            "Garanta que o repositório está completo."
        )
    spec = importlib.util.spec_from_file_location("teste_mapa._terrain_viewer", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Falha ao preparar o carregamento dinâmico do terrain_viewer.")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def _default_config_path() -> Path:
    """Retorna o caminho padrão do arquivo de configuração."""

    local_path = Path(__file__).resolve().parent / CONFIG_NAME
    return local_path


@dataclass
class WorldSelection:
    """Informações sobre o mundo selecionado."""

    world_path: Path
    map_id: Optional[int]
    object_path: Optional[Path]


class TesteMapaLauncher:
    """Coordenador de carregamento dos recursos e inicialização do viewer."""

    def __init__(self, data_root: Optional[Path], *, remember_root: bool, config_path: Optional[Path] = None) -> None:
        self._terrain = _load_terrain_module()
        self._config_path = config_path or _default_config_path()
        self.data_root = self._resolve_data_root(data_root)
        if remember_root:
            self._store_data_root(self.data_root)

    # ------------------------------------------------------------------
    # Configuração
    def _load_config(self) -> dict:
        if self._config_path.exists():
            try:
                with self._config_path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                    if isinstance(data, dict):
                        return data
            except Exception:  # noqa: BLE001
                pass
        return {}

    def _store_data_root(self, path: Path) -> None:
        path = path.resolve()
        payload = self._load_config()
        payload["data_root"] = str(path)
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        with self._config_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def _resolve_data_root(self, candidate: Optional[Path]) -> Path:
        """Determina a pasta Data raiz utilizando CLI, config ou defaults."""

        if candidate is not None:
            candidate = candidate.expanduser().resolve()
            if not candidate.is_dir():
                raise FileNotFoundError(f"Pasta Data informada é inválida: {candidate}")
            return candidate

        config = self._load_config()
        stored = config.get("data_root")
        if stored:
            stored_path = Path(stored).expanduser()
            if stored_path.is_dir():
                return stored_path.resolve()

        repo_guess = Path(__file__).resolve().parents[2] / "data"
        if repo_guess.is_dir():
            return repo_guess.resolve()

        raise FileNotFoundError(
            "Nenhuma pasta Data encontrada. Utilize --data-root para informar o caminho."
        )

    # ------------------------------------------------------------------
    # Descoberta de mundos
    def _looks_like_world(self, path: Path) -> bool:
        if not path.is_dir():
            return False
        if WORLD_PATTERN.match(path.name):
            return True
        return any(path.glob("EncTerrain*.map"))

    def available_worlds(self) -> List[Path]:
        worlds: List[Path] = []
        if not self.data_root.exists():
            return worlds
        for candidate in sorted(self.data_root.iterdir()):
            if self._looks_like_world(candidate):
                worlds.append(candidate)
        return worlds

    def show_world_list(self) -> None:
        worlds = self.available_worlds()
        if not worlds:
            print("Nenhum diretório de mundo foi encontrado na pasta Data informada.")
            return
        print("Mundos disponíveis:")
        for idx, world in enumerate(worlds):
            print(f"  [{idx}] {world.name}")

    def select_world(
        self,
        *,
        map_name: Optional[str],
        map_index: Optional[int],
        map_id: Optional[int],
        object_root: Optional[Path],
    ) -> WorldSelection:
        worlds = self.available_worlds()
        if not worlds:
            raise FileNotFoundError("Nenhum mundo foi localizado. Verifique a pasta Data.")

        selected: Optional[Path] = None
        if map_name:
            target_name = map_name.lower()
            for world in worlds:
                if world.name.lower() == target_name:
                    selected = world
                    break
            if selected is None:
                manual = (self.data_root / map_name).resolve()
                if manual.is_dir() and self._looks_like_world(manual):
                    selected = manual
                else:
                    raise FileNotFoundError(f"Mapa '{map_name}' não foi encontrado em {self.data_root}.")
        elif map_index is not None:
            if map_index < 0 or map_index >= len(worlds):
                raise IndexError(
                    f"Índice de mapa inválido ({map_index}). Existem {len(worlds)} opções disponíveis."
                )
            selected = worlds[map_index]
        else:
            if len(worlds) == 1 or not sys.stdin.isatty():
                selected = worlds[0]
            else:
                print("Selecione o mapa para carregar:")
                for idx, world in enumerate(worlds):
                    print(f"  [{idx}] {world.name}")
                while True:
                    choice = input("Informe o índice desejado: ").strip()
                    if not choice:
                        continue
                    if not choice.isdigit():
                        print("Digite apenas números correspondentes ao índice listado.")
                        continue
                    idx = int(choice)
                    if 0 <= idx < len(worlds):
                        selected = worlds[idx]
                        break
                    print("Índice fora do intervalo. Tente novamente.")
        assert selected is not None

        custom_object = object_root.expanduser().resolve() if object_root else None
        if custom_object is not None and not custom_object.is_dir():
            raise FileNotFoundError(f"Diretório de objetos inválido: {custom_object}")

        guess_object_path = custom_object
        if guess_object_path is None:
            guess = getattr(self._terrain, "guess_object_folder")(selected)
            guess_object_path = guess

        return WorldSelection(
            world_path=selected,
            map_id=map_id,
            object_path=guess_object_path,
        )

    # ------------------------------------------------------------------
    # Execução principal
    def prepare_summary(
        self,
        selection: WorldSelection,
        *,
        extended_height: bool,
        height_scale: Optional[float],
        enum_path: Optional[Path],
    ) -> str:
        result = getattr(self._terrain, "load_world_data")(
            selection.world_path,
            map_id=selection.map_id,
            object_path=selection.object_path,
            extended_height=extended_height,
            height_scale=height_scale,
            enum_path=enum_path,
        )
        detailed = getattr(self._terrain, "format_detailed_summary")
        return detailed(result, object_limit=6, attribute_limit=6)

    def run(
        self,
        selection: WorldSelection,
        *,
        extended_height: bool,
        height_scale: Optional[float],
        enum_path: Optional[Path],
        renderer: str,
        overlay: str,
        texture_detail: int,
        fog_density: Optional[float],
        fog_color: Optional[Tuple[float, float, float]],
        no_show: bool,
        output: Optional[Path],
        max_objects: Optional[int],
    ) -> None:
        run_viewer = getattr(self._terrain, "run_viewer")
        run_viewer(
            selection.world_path,
            map_id=selection.map_id,
            object_path=selection.object_path,
            extended_height=extended_height,
            height_scale=height_scale,
            output=output,
            show=not no_show,
            max_objects=max_objects,
            enum_path=enum_path,
            detailed_summary=True,
            summary_limit=10,
            render=not no_show or output is not None,
            export_objects=None,
            enable_object_edit=False,
            view_mode="3d",
            overlay=overlay,
            renderer=renderer,
            scene_focus="full",
            include_filters=None,
            exclude_filters=None,
            export_json=None,
            save_objects=None,
            texture_detail=max(1, texture_detail),
            fog_density=fog_density,
            fog_color=fog_color,
        )


# ----------------------------------------------------------------------
# Argumentos de linha de comando

def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Carrega mapas do cliente utilizando a lógica da Main.exe e abre um mini jogo exploratório.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, help="Pasta Data raiz contendo os diretórios WorldXX.")
    parser.add_argument("--remember-root", action="store_true", help="Persiste a pasta Data utilizada no arquivo de configuração.")
    parser.add_argument("--list", action="store_true", help="Lista os mundos disponíveis e encerra.")
    parser.add_argument("--map", help="Nome do diretório World que será carregado.")
    parser.add_argument("--map-index", type=int, help="Índice do mundo na listagem automática.")
    parser.add_argument("--map-id", type=int, help="Sobrescreve o ID do mapa informado nos arquivos EncTerrain.")
    parser.add_argument("--object-root", type=Path, help="Diretório ObjectXX alternativo.")
    parser.add_argument("--enum-path", type=Path, help="Caminho para o arquivo _enum.h original (opcional).")
    parser.add_argument("--renderer", choices=["opengl", "matplotlib"], default="opengl", help="Motor de renderização utilizado.")
    parser.add_argument("--overlay", choices=["textures", "height", "attributes"], default="textures", help="Overlay aplicado sobre o terreno.")
    parser.add_argument("--texture-detail", type=int, default=2, help="Subdivisões por tile para texturização.")
    parser.add_argument("--extended-height", action="store_true", help="Força leitura de TerrainHeightNew.OZB.")
    parser.add_argument("--height-scale", type=float, help="Sobrescreve o fator de escala vertical clássico.")
    parser.add_argument("--fog-density", type=float, help="Ajusta a densidade da névoa (apenas renderer OpenGL).")
    parser.add_argument(
        "--fog-color",
        nargs=3,
        type=float,
        metavar=("R", "G", "B"),
        help="Define a cor da névoa (componentes entre 0 e 1).",
    )
    parser.add_argument("--no-show", action="store_true", help="Não abre janela; apenas prepara os dados.")
    parser.add_argument("--output", type=Path, help="Arquivo de saída para captura da visualização.")
    parser.add_argument("--max-objects", type=int, help="Limita a quantidade de objetos carregados na cena.")
    parser.add_argument(
        "--config-path",
        type=Path,
        help="Sobrescreve o caminho padrão do arquivo de configuração.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Gera apenas o resumo textual do mapa e encerra.",
    )
    return parser.parse_args(argv)


# ----------------------------------------------------------------------
# Ponto de entrada principal

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_arguments(argv)

    try:
        launcher = TesteMapaLauncher(
            args.data_root,
            remember_root=args.remember_root,
            config_path=args.config_path,
        )
    except FileNotFoundError as exc:
        print(exc)
        return 1

    if args.list:
        launcher.show_world_list()
        return 0

    try:
        selection = launcher.select_world(
            map_name=args.map,
            map_index=args.map_index,
            map_id=args.map_id,
            object_root=args.object_root,
        )
    except (FileNotFoundError, IndexError) as exc:
        print(exc)
        return 1

    fog_color = tuple(args.fog_color) if args.fog_color else None

    try:
        summary = launcher.prepare_summary(
            selection,
            extended_height=args.extended_height,
            height_scale=args.height_scale,
            enum_path=args.enum_path,
        )
    except Exception as exc:  # noqa: BLE001
        print("Falha ao carregar os dados do mapa:", exc)
        return 1

    header = textwrap.dedent(
        f"""
        === Resumo do mapa {selection.world_path.name} ===
        Pasta Data : {launcher.data_root}
        Mundo      : {selection.world_path}
        Objetos    : {selection.object_path or 'inferido automaticamente'}
        """
    ).strip()
    print(header)
    print(summary)

    if args.summary_only:
        return 0

    try:
        launcher.run(
            selection,
            extended_height=args.extended_height,
            height_scale=args.height_scale,
            enum_path=args.enum_path,
            renderer=args.renderer,
            overlay=args.overlay,
            texture_detail=args.texture_detail,
            fog_density=args.fog_density,
            fog_color=fog_color,
            no_show=args.no_show,
            output=args.output,
            max_objects=args.max_objects,
        )
    except Exception as exc:  # noqa: BLE001
        print("Erro durante a renderização:", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
