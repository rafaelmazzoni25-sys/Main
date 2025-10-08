# Revisão da abordagem OpenGL no `terrain_viewer`

## Visão geral do pipeline atual

O modo OpenGL do utilitário Python é carregado apenas quando `moderngl` e `pyglet` estão
presentes, criando um contexto moderno para renderizar terreno, céu e objetos
com controles em tempo real.【F:tools/terrain_viewer/terrain_viewer.py†L50-L64】【F:tools/terrain_viewer/terrain_viewer.py†L2330-L2389】

A malha do terreno é gerada em CPU pela classe `_TerrainBuffers`, que
subdivide o grid original de 256×256 de acordo com o `detail_factor`, calcula
normais, máscaras de sombra e replica os flags de material por tile. Em
seguida, monta VBO/IBO/VAO com posição, normal, UV e o código de material que
aciona comportamentos especiais no shader.【F:tools/terrain_viewer/terrain_viewer.py†L1815-L1913】
As texturas dos tiles são pré-compostas na CPU a partir das camadas 1/2 e do
alpha do mapa, e o pacote ainda gera texturas derivadas para o light map,
normal map sintético e sombra projetada, que alimentam o shader difuso.【F:tools/terrain_viewer/terrain_viewer.py†L1915-L1959】

O shader de terreno combina essas texturas para iluminar o solo com luz
ambiental, direcional, pontos dinâmicos e ajustes de materiais como água ou
lava, além de fog exponencial. Já o shader de objetos aplica UV offset para
materiais animados, suporta normal map, emissivo e descartes por alpha test, e
utiliza as mesmas luzes calculadas para o terreno.【F:tools/terrain_viewer/terrain_viewer.py†L2391-L2527】【F:tools/terrain_viewer/terrain_viewer.py†L2529-L2607】

Instâncias BMD reutilizam o parser e animações do projeto, com interpolação de
curvas e skinning aplicado diretamente nos buffers antes da renderização.【F:tools/terrain_viewer/terrain_viewer.py†L1977-L2076】【F:tools/terrain_viewer/terrain_viewer.py†L2243-L2315】
Durante cada quadro o app atualiza animações, reconstrói luzes dinâmicas,
configura uniformes (incluindo fog, sombras e emissivo) e desenha primeiro o
terreno, depois objetos, partículas e céu, garantindo estados de blend/depth
adequados após cada etapa.【F:tools/terrain_viewer/terrain_viewer.py†L3158-L3294】

## Pontos fortes observados

* **Paridade com o carregamento original** – O viewer reutiliza os mesmos
  loaders de `EncTerrain*.att/.map/.obj` e o pipeline de animação BMD do
  cliente, o que reduz discrepâncias na geometria, materiais e movimentos em
  relação ao jogo.【F:tools/terrain_viewer/terrain_viewer.py†L3374-L3410】【F:tools/terrain_viewer/terrain_viewer.py†L1977-L2076】
* **Iluminação rica e efeitos atmosféricos** – O shader de terreno replica
  componentes direcional, pontos, emissivo e fog, enquanto o renderizador
  mantém um skybox/gradiente dinâmico e partículas volumétricas opcionais,
  aproximando a atmosfera do cliente original.【F:tools/terrain_viewer/terrain_viewer.py†L2391-L2527】【F:tools/terrain_viewer/terrain_viewer.py†L3110-L3156】
* **Materiais configuráveis** – O código carrega estados de material externos
  (blend, alpha test, emissivo, normal map) e aplica combinações de blend/depth
  por malha, respeitando propriedades individuais de modelos e tiles.【F:tools/terrain_viewer/terrain_viewer.py†L1815-L1913】【F:tools/terrain_viewer/terrain_viewer.py†L1977-L2100】

## Divergências e limitações

* **Mistura de camadas simplificada** – O viewer pré-compõe as texturas das
  camadas no CPU usando uma mistura linear do alpha.【F:tools/terrain_viewer/terrain_viewer.py†L1044-L1091】
  No cliente original a seleção da camada 2 é feita por vértice/passe e depende
  de condições discretas (por exemplo, alpha ≥ 1 nos quatro vértices ou regras
  especiais para água), com passes separados para blend aditivo ou alpha.【F:source/ZzzLodTerrain.cpp†L1514-L1558】
  Isso significa que transições que deveriam usar texturas distintas em múltiplos
  passes podem aparecer suavizadas demais no viewer.
* **Passe especular ausente** – Apesar da infraestrutura para um shader
  especular, `terrain_specular_program` não é inicializado, deixando reflexos
  em água/lava dependentes apenas do termo especular embutido no shader difuso.
  Para emular melhor o cliente (que faz múltiplos passes), seria necessário
  separar o passe especular ou reforçar o highlight atual.【F:tools/terrain_viewer/terrain_viewer.py†L2391-L2527】
* **Ordenação de transparências** – Objetos marcados como transparentes são
  enviados na ordem original sem sort por distância. Em mapas com muita
  vegetação ou efeitos translúcidos isso pode gerar artefatos de profundidade;
  implementar sort por profundidade mitigaria o problema.【F:tools/terrain_viewer/terrain_viewer.py†L3265-L3287】
* **Custo de pré-processamento** – A expansão de tiles por `detail_factor` e a
  geração de mapas auxiliares ocorrem a cada carga de mundo, o que pode ser
  caro em ambientes com Python puro. Cachear as texturas compostas por mapa ou
  permitir selecionar um `detail_factor` menor pode reduzir o tempo de setup.【F:tools/terrain_viewer/terrain_viewer.py†L1815-L1959】

## Recomendações

1. **Reproduzir a lógica de blending original** – Implementar a seleção de
   texturas por vértice/passe em GPU (duas samplers + branch por flag) ou ao
   menos aplicar as mesmas condições discretas antes de pré-compor reduzirá
   discrepâncias visuais com o cliente.【F:tools/terrain_viewer/terrain_viewer.py†L1915-L1959】【F:source/ZzzLodTerrain.cpp†L1514-L1558】
2. **Adicionar passe especular dedicado** – Aproveitar a infraestrutura já
   prevista (`terrain_specular_program`) para um segundo draw com blend aditivo
   forneceria reflexos mais próximos aos do jogo, principalmente em água/lava.
3. **Ordenar malhas transparentes** – Antes de desenhar objetos, ordenar as
   malhas com `MATERIAL_TRANSPARENT` pela distância à câmera melhora a
   consistência de blending.【F:tools/terrain_viewer/terrain_viewer.py†L3265-L3287】
4. **Cachear recursos derivados** – Persistir em disco ou em memória as
   texturas compostas/normal map gerados por `_TerrainBuffers` evitará repetir
   trabalho em recargas do mesmo mapa, beneficiando iterações rápidas.【F:tools/terrain_viewer/terrain_viewer.py†L1815-L1959】

## Conclusão

A arquitetura OpenGL atual do `terrain_viewer` é funcional e cobre a maior
parte do pipeline do cliente original, incluindo animações BMD e efeitos de luz
complexos. As principais oportunidades de melhoria estão na fidelidade da
mistura de camadas do terreno, na ausência de um passe especular dedicado e na
falta de ordenação para transparências. Ajustando esses pontos o visual ficará
mais alinhado com o jogo original, mantendo a base sólida já implementada.
