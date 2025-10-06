# Entendendo o carregamento do `TerrainHeight.OZB`

Este documento detalha como o cliente converte o arquivo `TerrainHeight.OZB`
em um campo de alturas utilizável. Ele complementa a visão geral em
[`terrain_overview.md`](./terrain_overview.md) com foco específico no formato
"clássico" do terreno.

## Quando o formato clássico é usado

Ao ativar um mapa, o gerenciador monta o caminho `WorldX/TerrainHeight.bmp` e
invoca `CreateTerrain`. Somente alguns mundos marcados em
`IsTerrainHeightExtMap` utilizam o formato estendido (`TerrainHeightNew.OZB`);
todos os demais passam por `OpenTerrainHeight`, que espera o arquivo clássico
na pasta `Data/WorldX` com extensão `.OZB`.【F:source/MapManager.cpp†L1386-L1405】【F:source/ZzzLodTerrain.cpp†L500-L538】

## O que é um arquivo `.OZB`

Os utilitários internos geram `.OZB` acrescentando quatro bytes no início de um
BMP de 8 bits: primeiro gravam os quatro bytes originais do cabeçalho e, em
seguida, repetem o arquivo inteiro. A rotina de leitura ignora essa "capa"
extra com `fseek(fp, 4)` antes de processar o BMP propriamente dito.【F:source/GlobalBitmap.cpp†L816-L892】【F:source/ZzzLodTerrain.cpp†L525-L545】

## Conversão para o buffer de alturas

1. **Leitura em memória** – A função reserva um buffer com 1080 bytes (cabeçalho
   de um BMP indexado) + `256 * 256` bytes de pixels e carrega o conteúdo do
   arquivo após os quatro bytes extras. O cabeçalho é copiado para o array
   global `BMPHeader`, reaproveitado depois ao salvar alterações no terreno.
   【F:source/ZzzLodTerrain.cpp†L531-L563】
2. **Iteração por linhas** – O loop percorre cada linha `i` e obtém um ponteiro
   para os 256 pixels correspondentes. Cada valor é convertido para `float` e
   gravado em `BackTerrainHeight`, o array central de alturas do terreno. Não há
   inversão de linhas nessa etapa porque o BMP já foi serializado em ordem de
   cima para baixo quando exportado para `.OZB`.
3. **Fator de escala** – Para todos os mapas, exceto o cenário de login, o
   valor do pixel é multiplicado por `1.5`. O mapa de login usa `3.0` para
   compensar a escala reduzida do modelo. Essa conversão define as unidades
   finais usadas por física, posicionamento de objetos e renderização.【F:source/ZzzLodTerrain.cpp†L548-L563】

Ao final, o buffer temporário é liberado e `BackTerrainHeight` contém 65.536
amostras igualmente espaçadas (passo de `TERRAIN_SCALE` nas coordenadas X/Y).

## Como o valor é utilizado em jogo

`RequestTerrainHeight` consulta `BackTerrainHeight` com interpolação bilinear
para responder à altura de qualquer ponto (`xf`, `yf`). Flags especiais como
`TW_HEIGHT` podem forçar um patamar fixo, mas na maioria dos casos o valor
interpolado é devolvido diretamente. Essa função abastece praticamente todos os
sistemas de posicionamento (movimentação, partículas, monstros, etc.).
【F:source/ZzzLodTerrain.cpp†L640-L685】【F:source/ZzzLodTerrain.cpp†L1552-L1714】

## Comparativo com o formato estendido

Os mapas listados em `IsTerrainHeightExtMap` utilizam `OpenTerrainHeightNew`,
que lê pixels RGB de 24 bits e reconstrói a altura combinando os três canais e
somando `g_fMinHeight`. Essa diferença permite mapas mais altos ou com offset
global, mas não interfere no processo clássico descrito acima. O arquivo
`TerrainHeightNew.OZB` continua a carregar o BMP após quatro bytes extras, mas
usa um cabeçalho de 54 bytes típico de BMPs true color.【F:source/ZzzLodTerrain.cpp†L590-L639】

## Resumo

- `TerrainHeight.OZB` é um BMP indexado de 256×256 bytes, encapsulado com
  quatro bytes iniciais.
- O carregador lê o BMP, armazena o cabeçalho para futuras gravações e converte
  cada pixel para `float`, escalonando por `1.5` (ou `3.0` no login).
- O resultado alimenta diretamente a consulta de alturas e a renderização do
  terreno clássico; apenas mapas específicos exigem o formato estendido.

