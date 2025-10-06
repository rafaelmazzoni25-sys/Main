# Terrain viewer

Este utilitário em Python recria parte do pipeline de carregamento do terreno
(altura, atributos, mapeamento de texturas e objetos estáticos) e gera uma
visualização rápida usando o Matplotlib.

## Pré-requisitos

O script depende de Python 3.9+ com as bibliotecas abaixo:

```bash
pip install -r requirements.txt
```

O arquivo `requirements.txt` na pasta contém as dependências mínimas (NumPy e
Matplotlib).

## Como usar

1. Extraia os arquivos do cliente em uma pasta acessível. Você precisará do
   diretório `Data/WorldX` correspondente ao mapa que deseja visualizar com os
   arquivos `EncTerrain*.att`, `EncTerrain*.map` e `TerrainHeight.OZB` (ou
   `TerrainHeightNew.OZB`). Se os objetos estiverem separados em `Data/ObjectX`,
   mantenha essa pasta acessível também.

### Interface gráfica

Execute o script sem argumentos (ou com `--gui`) para abrir uma janela que
permite:

1. Escolher a pasta `Data` por meio de um seletor de diretórios.
2. Selecionar qual `WorldX` carregar usando um drop-down preenchido
   automaticamente com base na pasta escolhida.
3. Ajustar opções como `Map ID`, escala de altura, limite de objetos e, se
   necessário, apontar explicitamente para uma pasta `ObjectX`.

Ao clicar em **Visualizar**, o terreno e os objetos são carregados e exibidos.
Também é possível gerar uma imagem PNG diretamente pelo botão **Salvar PNG**.

### Linha de comando

Caso prefira o modo tradicional, execute o script apontando para a pasta
`WorldX` (e opcionalmente informe a pasta `ObjectX` com `--object-path`):

```bash
python terrain_viewer.py /caminho/para/Data/World1 --output world1.png
```

O comando acima gera um `PNG` com a visualização e também abre a janela do
Matplotlib. Para apenas salvar a imagem, adicione `--no-show`. Caso o ID do mapa
não esteja presente no nome da pasta, informe manualmente com `--map-id`. Se o
arquivo `EncTerrain*.obj` estiver armazenado em `Data/ObjectX`, use
`--object-path /caminho/para/Data/Object1`.

### Opções principais

- `--extended-height`: força o parser do formato novo de altura (24 bits) — a
  mesma rotina `OpenTerrainHeightNew` usada pelo cliente.
- `--max-objects`: limita a quantidade de objetos renderizados (útil em mapas
  com milhares de instâncias).
- `--object-path`: aponta diretamente para a pasta `ObjectX` quando os objetos
  não estão junto do `WorldX`.
- `--no-show`: evita abrir a janela interativa do Matplotlib.
- `--output`: caminho do arquivo PNG de saída.
- `--height-scale`: ajusta o fator aplicado ao formato clássico de altura (1.5
  por padrão, use 3.0 para o mapa de login).

## Exemplo

```bash
python terrain_viewer.py /caminho/para/Data/World7 --max-objects 500 --output world7.png --no-show
```

O script decodifica os arquivos criptografados usando as mesmas rotinas inline
(`MapFileDecrypt` e `BuxConvert`) vistas em `source/ZzzLodTerrain.cpp` e
`source/ZzzObject.cpp`. O arquivo `EncTerrainXX.obj` é interpretado exatamente
como no cliente: cada entrada informa o `type_id`, posição (`x`, `y`, `z`),
ângulos (`pitch`, `yaw`, `roll`) e escala. As posições são convertidas para o
mesmo espaço do terreno e plotadas acima da altura correspondente, garantindo
que a visualização reproduza o cenário esperado do jogo.
