# Estado do `Terrain_Editor`

A antiga ferramenta em Python foi substituída por uma aplicação C++ (`tools/Terrain_Editor`)
que mantém o pipeline de leitura dos arquivos `EncTerrain` e adiciona um renderer
compatível com OpenGL clássico.

## Implementação atual

### Carregamento de terreno e atributos
* O carregador C++ reproduz as rotinas `MapFileDecrypt` e `BuxConvert`, valida os
  cabeçalhos dos arquivos `EncTerrain*.att`/`.map` e reconstrói camadas de tile,
  alpha e atributos com o mesmo tamanho de grade do cliente original.【F:tools/Terrain_Editor/src/WorldLoader.cpp†L23-L125】【F:tools/Terrain_Editor/src/WorldLoader.cpp†L209-L271】
* Os dois formatos de altura (`TerrainHeight.OZB` e `TerrainHeightNew.OZB`) são
  suportados, incluindo o fator de escala clássico e a conversão RGB→altura do
  formato estendido.【F:tools/Terrain_Editor/src/WorldLoader.cpp†L273-L317】

### Objetos estáticos (`EncTerrain*.obj`)
* O parser nativo descriptografa `EncTerrain*.obj`, lê versão, ID do mapa,
  posições, rotações e escala, e associa nomes simbólicos a partir do
  `_enum.h` original quando disponível.【F:tools/Terrain_Editor/src/WorldLoader.cpp†L319-L380】
* Após o carregamento, cada objeto é reposicionado usando amostragem bilinear da
  malha de altura para garantir alinhamento visual ao terreno renderizado.【F:tools/Terrain_Editor/src/WorldLoader.cpp†L167-L199】

### Renderização e controles
* A malha do terreno é triangulada em C++, calculando normais suaves a partir da
  grade de alturas e aplicando uma paleta determinística de cores por tile.【F:tools/Terrain_Editor/src/TerrainMesh.cpp†L13-L88】【F:tools/Terrain_Editor/src/TilePalette.cpp†L5-L36】
* O renderer utiliza o pipeline compatível do OpenGL 2.1, configurando câmera
  orbital, iluminação direcional simples e desenhando os objetos como eixos
  coloridos para facilitar inspeção rápida antes da futura integração do
  pipeline BMD.【F:tools/Terrain_Editor/src/Renderer.cpp†L12-L134】
* A aplicação expõe uma janela interativa via GLFW, com orbit/pan via mouse e
  zoom por teclado (`Q`/`E`), replicando o fluxo de inspeção dos mapas sem
  depender do cliente original.【F:tools/Terrain_Editor/src/Application.cpp†L11-L75】【F:tools/Terrain_Editor/src/Renderer.cpp†L36-L91】

## Itens ainda pendentes
* Texturas de tile, materiais especiais (água/lava) e partículas atmosféricas
  ainda não foram reimplementados na versão C++; o visual atual usa coloração
  sintética para manter a legibilidade da topografia.【F:tools/Terrain_Editor/src/Renderer.cpp†L96-L134】【F:tools/Terrain_Editor/src/TilePalette.cpp†L17-L34】
* Modelos BMD, animações e efeitos de pós-processamento continuam pendentes para
  atingir paridade total com o cliente. As estruturas de carregamento já
  preservam metadados suficientes para essa etapa futura.【F:tools/Terrain_Editor/src/WorldLoader.cpp†L333-L380】

## Conclusão

O `Terrain_Editor` fornece um visualizador nativo multiplataforma capaz de ler os
mesmos arquivos `EncTerrain` do projeto base, gerar a malha do mapa, alinhar os
objetos e permitir inspeção interativa sem dependências externas de Python. A
fidelidade visual será ampliada em iterações futuras com a integração do sistema
completo de materiais e modelos animados.
