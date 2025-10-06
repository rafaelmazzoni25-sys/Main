# Visão geral da montagem do terreno

Este documento resume como o cliente monta e renderiza o terreno no projeto, com base no arquivo `source/ZzzLodTerrain.cpp`.

## Estruturas principais

O módulo de terreno mantém buffers globais para os dados de malha, atributos e iluminação, como altura, normais, camadas de texturas e marcações de colisão. Esses arrays são dimensionados para `TERRAIN_SIZE * TERRAIN_SIZE` e incluem buffers específicos para gramado e atributos especiais do mapa.【F:source/ZzzLodTerrain.cpp†L31-L57】

Os índices dos tiles são calculados por utilitários como `TERRAIN_INDEX` (indexação direta) e `TERRAIN_INDEX_REPEAT` (indexação com wrap), enquanto `TERRAIN_ATTRIBUTE` traduz coordenadas do mundo para o valor de atributo da célula (`TerrainWall`).【F:source/ZzzLodTerrain.cpp†L72-L87】

## Carregamento dos dados do mapa

### Atributos

`OpenTerrainAttribute` lê o arquivo de atributos (com encriptação Bux e `MapFileDecrypt`), preenche `TerrainWall` e valida algumas posições conhecidas para garantir integridade. Cada célula guarda flags como zonas de água, altura especial, bloqueios de movimento etc.【F:source/ZzzLodTerrain.cpp†L114-L210】 As funções `AddTerrainAttribute` e correlatas permitem editar essas flags em runtime.【F:source/ZzzLodTerrain.cpp†L228-L258】

### Mapeamento de texturas

`OpenTerrainMapping` abre o arquivo de mapeamento de texturas, também encriptado, preenchendo duas camadas (`TerrainMappingLayer1/2`) e um alpha por célula. Esse alpha decide se a segunda camada substitui a primeira e também controla a renderização de gramado. Há desativação opcional do gramado em mapas como Chaos Castle ou Battle Castle.【F:source/ZzzLodTerrain.cpp†L285-L329】

### Altura

`CreateTerrain` ativa o terreno e decide qual rotina de leitura usar (`OpenTerrainHeight` ou `OpenTerrainHeightNew`) conforme o formato do arquivo. No fim, chama `CreateSun` para preparar objetos visuais relacionados.【F:source/ZzzLodTerrain.cpp†L485-L499】

* **Formato clássico:** `OpenTerrainHeight` lê um BMP compactado (`*.OZB`), copia o cabeçalho e converte cada pixel em altura, com um fator de escala diferente para o mapa de login.【F:source/ZzzLodTerrain.cpp†L513-L563】
* **Formato estendido:** `OpenTerrainHeightNew` lê alturas codificadas em 24 bits, deslocando-as para o intervalo real adicionando `g_fMinHeight`. Esse formato cobre mapas novos/estendidos.【F:source/ZzzLodTerrain.cpp†L596-L639】

Com os arrays preenchidos, funções utilitárias como `RequestTerrainHeight` interpolam a altura em tempo real e respeitam flags especiais como `TW_HEIGHT`, que força um "platô" alto (usado para paredes invisíveis, por exemplo).【F:source/ZzzLodTerrain.cpp†L649-L684】

## Derivação de normais e iluminação

Depois de carregar a altura, `CreateTerrainNormal` percorre cada célula e monta normais a partir dos quatro vértices adjacentes, permitindo iluminação suave. Existe também a versão parcial para recalcular regiões locais.【F:source/ZzzLodTerrain.cpp†L371-L409】

`OpenTerrainLight` lê uma textura de luz (JPEG) e, após gerar as normais, chama `CreateTerrainLight` para combinar a textura com a direção da luz (diferente em alguns mapas, como Battle Castle). O resultado alimenta `BackTerrainLight`, usado diretamente na renderização.【F:source/ZzzLodTerrain.cpp†L411-L465】

Funções auxiliares (`SetTerrainLight`, `AddTerrainLight`, etc.) espalham contribuições de luz em um raio usando um fator radial, possibilitando efeitos dinâmicos como sombras locais.【F:source/ZzzLodTerrain.cpp†L725-L756】

## Pipeline de renderização

A chamada principal `RenderTerrain` coordena a renderização. Ela ajusta animações de água (`WaterMove`), prepara o modo de blending e define `TerrainFlag` para renderizar primeiro o solo padrão e depois, opcionalmente, o gramado. Também inicializa seleção/edição quando necessário.【F:source/ZzzLodTerrain.cpp†L2455-L2504】

`RenderTerrainFrustrum` percorre blocos 4x4 dentro dos limites do frustum pré-calculado, chamando `RenderTerrainBlock` para subdividir cada bloco em tiles individuais conforme o LOD atual. Isso evita desenhar tiles fora da visão.【F:source/ZzzLodTerrain.cpp†L2358-L2410】

`RenderTerrainTile` monta os quatro vértices de um tile com base em `BackTerrainHeight`, ajusta alturas especiais (`TW_HEIGHT`) e delega a renderização real a `RenderTerrainFace`. Em modo de edição, a função também trata seleção e depuração de atributos; em modo normal, desenha o tile com a textura adequada e iluminação interpolada.【F:source/ZzzLodTerrain.cpp†L1552-L1670】

Após o solo base, `RenderTerrainTile_After` e `RenderTerrainFace_After` desenham sobreposições como água transparente ou combinações de camadas, usando o alpha de mapeamento para decidir qual textura aplicar.【F:source/ZzzLodTerrain.cpp†L1526-L1697】 Finalmente, quando o gramado está ativo (`TerrainFlag == TERRAIN_MAP_GRASS`), os mesmos loops são reutilizados para renderizar o efeito animado de grama, deslocando vértices com `TerrainGrassWind` e texturas randômicas.【F:source/ZzzLodTerrain.cpp†L1499-L1517】【F:source/ZzzLodTerrain.cpp†L2494-L2499】

## Ferramenta de visualização

Para inspecionar rapidamente o resultado desse pipeline sem executar o cliente original, o repositório inclui o utilitário `tools/terrain_viewer/terrain_viewer.py`. Ele aplica as mesmas rotinas de descriptografia (`MapFileDecrypt` e `BuxConvert`) usadas nos carregadores de atributos, texturas e objetos, reconstrói o campo de altura e plota o terreno com os objetos estáticos em 3D utilizando Matplotlib.【F:tools/terrain_viewer/terrain_viewer.py†L1-L340】 O parser de objetos lê `EncTerrainXX.obj` exatamente como o cliente (`type_id`, posição XYZ, ângulos e escala), posicionando cada instância acima da altura interpolada no grid.【F:tools/terrain_viewer/terrain_viewer.py†L93-L151】【F:tools/terrain_viewer/README.md†L45-L70】 A ferramenta também oferece uma interface gráfica: basta selecionar a pasta `Data` e escolher o `WorldX` desejado em um menu drop-down, facilitando a exploração de múltiplos mapas sem lembrar caminhos específicos.【F:tools/terrain_viewer/terrain_viewer.py†L222-L338】【F:tools/terrain_viewer/README.md†L20-L44】 Consulte a documentação na própria pasta para detalhes de uso, incluindo o suporte a diretórios `ObjectX` externos para os arquivos `EncTerrainXX.obj`.

## Conclusão

O terreno é montado em três etapas principais: leitura dos dados (altura, atributos e texturas), geração de informações derivadas (normais e luz) e renderização otimizada com frustum culling. O uso de arrays globais para cada aspecto do terreno permite que sistemas diferentes (colisão, efeitos, UI de edição) consultem ou modifiquem os dados conforme necessário.
