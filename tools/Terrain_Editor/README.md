# Terrain_Editor

O `Terrain_Editor` é a reimplementação em C++ do antigo `terrain_viewer` em Python.
Ele carrega os mesmos arquivos `EncTerrain` do cliente original e apresenta o
mapa em 3D com malha sombreada, objetos estáticos posicionados sobre o solo e
controles de câmera orbitais.

## Pré-requisitos

* [GLFW 3.3+](https://www.glfw.org/) — biblioteca de janelas e entrada.
* OpenGL (qualquer implementação com suporte ao perfil compatível 2.1).
* Compilador C++20.
* `glm` (já incluso em `dependencies/include`).

No Windows, recomenda-se instalar o GLFW por meio do vcpkg ou dos pacotes
pré-compilados oficiais e garantir que as DLLs estejam disponíveis no `PATH`.

## Compilação

O projeto fornece um `CMakeLists.txt`. A sequência básica é:

```bash
cmake -S tools/Terrain_Editor -B build/Terrain_Editor
cmake --build build/Terrain_Editor --config Release
```

Se o CMake não localizar o GLFW automaticamente, defina a variável
`CMAKE_PREFIX_PATH` apontando para a pasta contendo o pacote.

## Uso

Depois de compilar, execute:

```bash
./terrain_editor --world /caminho/para/Data/World1 --objects /caminho/para/Data/Object1
```

Opções principais:

* `--map <id>`: força o ID do mapa (caso o nome da pasta não siga o padrão `WorldX`).
* `--objects <pasta>`: diretório `ObjectX` com o `EncTerrainXX.obj` correspondente.
* `--height-scale <valor>`: substitui o fator padrão para `TerrainHeight.OZB` clássico.
* `--extended-height`: força a leitura da variação `TerrainHeightNew.OZB`.
* `--enum <arquivo>`: caminho para o `_enum.h` original, permitindo exibir o nome
  textual dos modelos.

## Controles

* Botão esquerdo do mouse: orbitar.
* Botão do meio: pan.
* Tecla `Q`: aproximar (zoom in).
* Tecla `E`: afastar (zoom out).

Os objetos são representados como eixos coloridos posicionados sobre o terreno.
Essa abordagem facilita diferenciar posição e orientação mesmo antes da futura
integração do pipeline completo de modelos BMD.
