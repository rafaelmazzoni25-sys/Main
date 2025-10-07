# Estado do `terrain_viewer`

## Cobertura atual em relação ao cliente legado

### Carregamento de terreno e atributos
* O utilitário Python reproduz a rotina de decodificação `MapFileDecrypt`, aplica o mesmo XOR/Bux e valida cabeçalhos para `EncTerrain*.att` e `EncTerrain*.map`, reconstituindo as camadas de textura, alpha e atributos exatamente como o cliente C++.【F:tools/terrain_viewer/terrain_viewer.py†L973-L999】【F:tools/terrain_viewer/terrain_viewer.py†L3063-L3113】【F:source/ZzzLodTerrain.cpp†L131-L205】【F:source/ZzzLodTerrain.cpp†L285-L343】
* Os dois formatos de altura (`TerrainHeight.OZB` clássico e `TerrainHeightNew.OZB`) são suportados com os mesmos fatores de escala utilizados pelo jogo original.【F:tools/terrain_viewer/terrain_viewer.py†L3116-L3141】【F:source/ZzzLodTerrain.cpp†L513-L596】
* As texturas de tile são buscadas pelas mesmas convenções de nomes, com fallback para índices estendidos, replicando o carregamento feito pelo `MapManager` no cliente.【F:tools/terrain_viewer/terrain_viewer.py†L801-L915】【F:source/MapManager.cpp†L1440-L1499】

### Objetos estáticos (`EncTerrain*.obj`)
* O parser Python lê `EncTerrain*.obj`, preserva versão, ID de mapa, posições, rotações e escala, atribui nomes usando `_enum.h` e pode exportar novamente com o algoritmo inverso de encriptação, mantendo compatibilidade com o formato original.【F:tools/terrain_viewer/terrain_viewer.py†L3144-L3187】【F:tools/terrain_viewer/terrain_viewer.py†L3966-L4025】【F:tools/terrain_viewer/terrain_viewer.py†L4168-L4183】【F:source/ZzzObject.cpp†L5057-L5118】

### Renderização e recursos gráficos
* O modo OpenGL reconstrói a malha do terreno com normal map, máscara de sombras e atributos por tile, além de combinar texturas em múltiplas camadas como o cliente DirectX9, permitindo iluminação dinâmica e materiais especiais (água, lava, aditivos).【F:tools/terrain_viewer/terrain_viewer.py†L1684-L1814】【F:tools/terrain_viewer/terrain_viewer.py†L2586-L3033】
* Modelos BMD são carregados com ossos, animações, eventos e attachments; o `BMDAnimationPlayer` realiza blending e skinning idênticos ao pipeline do cliente.【F:tools/terrain_viewer/terrain_viewer.py†L544-L689】【F:tools/terrain_viewer/terrain_viewer.py†L1880-L2117】【F:source/ZzzBMD.cpp†L55-L166】
* O utilitário expõe o mesmo conjunto de saídas auxiliares (resumo estatístico, exportação CSV/JSON e regravação de `EncTerrain`) que facilitam auditoria e edição externa, algo inexistente no executável legado.【F:tools/terrain_viewer/terrain_viewer.py†L4028-L4200】【F:tools/terrain_viewer/terrain_viewer.py†L4680-L4973】
* A iluminação estática replica o carregamento de `TerrainLight*.jpg` e o cálculo de luminosidade do cliente: o `TextureLibrary` prioriza as variantes corretas para Battle Castle/Crywolf, usa o mesmo vetor direcional e multiplica tanto a malha Matplotlib quanto o pipeline OpenGL pelo mapa de luz e pelo dot product com as normais.【F:tools/terrain_viewer/terrain_viewer.py†L823-L1017】【F:tools/terrain_viewer/terrain_viewer.py†L1873-L1908】【F:source/MapManager.cpp†L1388-L1435】【F:source/ZzzLodTerrain.cpp†L411-L437】

## Pontos ainda diferentes do cliente original
* O viewer ignora rotinas de tempo real do cliente, como troca dinâmica de texturas de luz/neblina em eventos (`MapManager::CreateTerrain`, `SetTerrainWaterState`) ou geração procedural de grama animada, portanto o clima e certos efeitos dependentes do estado do servidor não são reproduzidos automaticamente.【F:source/ZzzLodTerrain.cpp†L260-L343】【F:source/MapManager.cpp†L1381-L1439】
* Sistemas acoplados ao runtime (spawn de monstros, partículas específicas de eventos, lógica de colisão baseada em `TerrainWall`) permanecem exclusivos do executável C++, já que o script atua como ferramenta offline de inspeção/edição e não substitui o loop do jogo.

## Conclusão
O `terrain_viewer` cobre integralmente o pipeline de leitura de terreno/objetos e fornece um renderer equivalente (com modos OpenGL e Matplotlib), suficiente para reproduzir a cena estática do jogo e editar `EncTerrain`. As diferenças residem apenas em efeitos em tempo real e integração com sistemas de gameplay, que estão fora do escopo da ferramenta, indicando que a implementação atual já está funcionalmente alinhada ao projeto para visualização e edição offline.
