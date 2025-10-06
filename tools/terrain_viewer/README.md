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

## Guia rápido para iniciantes

Se você nunca executou um script em Python, siga este passo a passo. Ele parte
do zero e usa apenas ações comuns no Windows:

1. **Instale o Python**
   - Baixe o instalador oficial em <https://www.python.org/downloads/>.
   - Execute o instalador e marque a opção "Add Python to PATH" antes de
     clicar em **Install Now**.

2. **Prepare uma pasta de trabalho**
   - Copie esta pasta `terrain_viewer` para um local fácil, por exemplo, sua
     área de trabalho.
   - Extraia os arquivos do cliente MuOnline para uma pasta como
     `C:\MuOnline`. Dentro dela deve existir o diretório `Data` com as subpastas
     `WorldX` (terreno) e, se disponível, `ObjectX` (objetos).

3. **Abra o Prompt de Comando**
   - Pressione `Win + R`, digite `cmd` e tecle **Enter**.
   - No prompt, mude para a pasta do visualizador. Exemplo:

     ```bat
     cd %USERPROFILE%\Desktop\terrain_viewer
     ```

4. **Instale as dependências**
   - Ainda no prompt, execute:

     ```bat
     python -m pip install -r requirements.txt
     ```

5. **Abra o Terrain Viewer**
   - Execute o comando abaixo para iniciar a interface gráfica:

     ```bat
     python terrain_viewer.py --gui
     ```
   - Clique em **Selecionar pasta Data** e aponte para `C:\MuOnline\Data`.
   - Use o menu suspenso para escolher o mundo (por exemplo, `World7`).
   - Pressione **Visualizar** para carregar o mapa; a janela exibirá o terreno
     e uma lista dos objetos encontrados.
   - Para gerar relatórios rápidos, utilize os botões **Resumo** (exibe
     estatísticas detalhadas) ou **Exportar objetos** (salva um CSV com a lista
     completa de instâncias do mapa).

> Dica: Se você quiser apenas gerar uma imagem, clique em **Salvar PNG** e
> escolha onde guardar o arquivo. O processo leva alguns segundos em mapas
> grandes; aguarde até a barra de progresso chegar ao fim.

6. **Feche o programa**
   - Após explorar o mapa, clique no `X` da janela ou pressione `Esc`.

## Como usar

Para usuários com experiência em linha de comando, as instruções completas
continuam abaixo.

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
O rodapé da janela mostra um resumo com a contagem total de objetos e os tipos
mais frequentes (com nomes extraídos do `_enum.h` do cliente). Também é
possível gerar uma imagem PNG diretamente pelo botão **Salvar PNG**. Caso
precise analisar os dados, use **Resumo** para visualizar estatísticas (altura,
atributos e texturas) e **Exportar objetos** para salvar um CSV com todas as
instâncias posicionadas no mapa.

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
`--object-path /caminho/para/Data/Object1`. O carregamento imprime um resumo dos
objetos encontrados e, se desejar, é possível apontar explicitamente para o
arquivo `_enum.h` do cliente com `--enum-path` para que os IDs sejam convertidos
em nomes legíveis.

### Exportando e inspecionando dados

- `--export-objects caminho.csv`: grava um CSV com todos os objetos, incluindo
  posição, ângulos, escala e as coordenadas em tiles. É útil para importar a
  lista em ferramentas externas ou planilhas.
- `--detailed-summary`: exibe estatísticas adicionais diretamente no terminal,
  como altura mínima/máxima, quantidade de texturas utilizadas e os atributos
  mais comuns.
- `--summary-limit N`: controla quantos tipos de objeto aparecem nos resumos
  (padrão 8).

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
- `--enum-path`: sobrescreve o caminho padrão de `_enum.h` usado para nomear os
  modelos lidos de `EncTerrainXX.obj`.
- `--export-objects`: salva a lista bruta de objetos em CSV.
- `--detailed-summary`: mostra estatísticas completas no terminal.
- `--summary-limit`: define quantos itens exibir nos resumos.

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
