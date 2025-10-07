# Terrain viewer

Este utilitário em Python recria parte do pipeline de carregamento do terreno
(altura, atributos, mapeamento de texturas e objetos estáticos) e oferece uma
visualização interativa com dois motores: o novo renderer OpenGL (via
`moderngl`/`pyglet`), que reproduz os azulejos originais do cliente com
iluminação dinâmica, névoa configurável e os modelos BMD posicionados no mapa;
e um modo compatível com o Matplotlib para ambientes sem aceleração gráfica.
Com o renderer OpenGL habilitado é possível percorrer o cenário com o mesmo
visual visto in-game sempre que as texturas `Tile*.jpg/.ozj/.ozt` e os arquivos
`Data/ObjectX/*.bmd` estiverem presentes.

### O que o renderer OpenGL reproduz do cliente

* **Iluminação multi-passos** – a malha do terreno recebe uma etapa difusa
  seguida por um passe especular aditivo que realça reflexos d'água, lava e
  materiais brilhantes, utilizando a mesma direção de luz do jogo original.
* **Animações BMD** – o carregador lê as curvas de translação/rotação dos
  ossos e aplica skinning nas malhas, fazendo com que portais, estruturas ou
  criaturas com keyframes ganhem movimento automaticamente.
* **Shaders especializados para água e lava** – os tiles com essas tags usam
  deslocamento UV, ondulação dinâmica e reforço de cores para simular o fluxo
  contínuo do jogo.
* **Céu e névoa dinâmicos** – o gradiente do céu alterna suavemente entre tons
  diurnos e noturnos de acordo com o relógio interno, sincronizando a névoa com
  a paleta de cada período.
* **Partículas atmosféricas** – pontos volumétricos com velocidades e tempos de
  vida randômicos criam poeira, fagulhas e neve leve dependendo do mapa,
  reforçando o clima geral.

## Pré-requisitos

O script depende de Python 3.9+ com as bibliotecas abaixo:

```bash
pip install -r requirements.txt
```

O arquivo `requirements.txt` na pasta contém as dependências mínimas (NumPy,
Matplotlib, moderngl e pyglet). Em ambientes sem suporte a OpenGL você pode
instalar apenas NumPy/Matplotlib e iniciar o visualizador com
`--renderer matplotlib`.

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
   - Use o menu suspenso para escolher o mundo (por exemplo, `World7`). O campo
     **Pasta Object** será preenchido automaticamente com a pasta `ObjectX`
     correspondente quando encontrada.
   - Pressione **Visualizar** para carregar o mapa; a janela exibirá o terreno
     e uma lista dos objetos encontrados.
   - Ajuste **Modo de visualização** para alternar entre a cena 3D padrão e a
     nova visão 2D (heatmap) — útil para posicionar instâncias com precisão.
   - Troque o **Overlay** para colorir o terreno pelas texturas, altura ou
     atributos. Os campos **Mostrar apenas** e **Ocultar** filtram os objetos
     renderizados por ID ou texto livre.
   - Ajuste **Detalhe textura** para definir quantas subdivisões cada tile
     recebe quando o overlay está em **Texturas** (valores maiores exibem mais
     definição, porém exigem mais tempo de renderização).
   - Escolha o **Renderer** entre *OpenGL* (visual mais próximo do cliente, com
     texturas reais e modelos BMD) e *Matplotlib* (compatibilidade total em
     ambientes sem aceleração gráfica).
   - Configure **Névoa densidade** e **Cor névoa (R,G,B)** para alterar o clima
     do renderer OpenGL; deixe em branco para usar os padrões sugeridos.
   - Marque **Permitir mover objetos** para habilitar a edição direta na janela
     do Matplotlib. Selecione um ponto com o mouse e use as setas para mover a
     instância (Shift acelera o passo, `[` e `]` ajustam a distância percorrida).
     Após fechar a janela, use **Salvar EncTerrain** para gerar um novo
     `EncTerrainXX.obj` com as posições atualizadas.
   - Use **WASD** para transladar a câmera pela cena 3D, **Q/E** para aproximar
     ou afastar, **I/K** para inclinar a vista e **J/L** para orbitar; o scroll
     do mouse também ajusta o zoom rapidamente.
   - Para gerar relatórios rápidos, utilize os botões **Resumo** (exibe
     estatísticas detalhadas), **Exportar objetos** (salva um CSV com a lista
     filtrada), **Exportar JSON** (gera um arquivo estruturado para scripts) ou
     **Salvar EncTerrain** (persiste as edições realizadas).

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
   `TerrainHeightNew.OZB`). Para que o terreno apareça texturizado como no
   jogo, mantenha também os arquivos `Tile*.jpg/.ozj/.ozt` fornecidos pelo
   cliente. Se os objetos estiverem separados em `Data/ObjectX`, mantenha essa
   pasta acessível também.

### Interface gráfica

Execute o script sem argumentos (ou com `--gui`) para abrir uma janela que
permite:

1. Escolher a pasta `Data` por meio de um seletor de diretórios.
2. Selecionar qual `WorldX` carregar usando um drop-down preenchido
   automaticamente com base na pasta escolhida.
3. Ajustar opções como `Map ID`, escala de altura, limite de objetos, modo de
   visualização (3D/2D), overlay (texturas/altura/atributos) e filtros para
   mostrar ou ocultar tipos específicos. Também é possível apontar
   explicitamente para uma pasta `ObjectX`.

Ao clicar em **Visualizar**, o terreno e os objetos são carregados e exibidos.
O rodapé da janela mostra um resumo com a contagem total de objetos e os tipos
mais frequentes (com nomes extraídos do `_enum.h` do cliente). Também é
possível gerar uma imagem PNG diretamente pelo botão **Salvar PNG**. Caso
precise analisar os dados, use **Resumo** para visualizar estatísticas
(altura/atributos/texturas), **Exportar objetos** ou **Exportar JSON** para
obter a lista filtrada em CSV/JSON, e **Salvar EncTerrain** para persistir as
edições feitas no editor interativo.

Quando a visualização 3D estiver aberta, utilize **WASD/QE/IJKL** para navegar
livremente pela cena. As setas do teclado continuam dedicadas ao movimento dos
objetos selecionados, enquanto o scroll do mouse oferece um zoom incremental.
Modelos BMD com animações têm suas curvas reproduzidas automaticamente no
renderer OpenGL, então portais e criaturas presentes no mapa exibirão o mesmo
movimento visto dentro do cliente.

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

- `--export-objects caminho.csv`: grava um CSV com os objetos visíveis na
  visualização atual, incluindo posição, ângulos, escala e as coordenadas em
  tiles. É útil para importar a lista em ferramentas externas ou planilhas.
- `--export-json caminho.json`: exporta o mesmo conjunto filtrado em formato
  JSON, junto com estatísticas básicas de altura e o histograma de atributos.
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
- `--view-mode`: alterna entre o modo 3D tradicional e a visualização 2D
  (heatmap) com sobreposição plana.
- `--overlay`: define o mapa de cores do terreno (texturas, altura ou
  atributos).
- `--renderer`: seleciona o motor de visualização (*opengl* por padrão). Use
  `--renderer matplotlib` para forçar o modo clássico.
- `--texture-detail`: controla a densidade da malha quando o overlay está em
  texturas. Valores altos (4, 8...) deixam os azulejos mais nítidos às custas de
  maior tempo de renderização.
- `--fog-density` / `--fog-color`: ajustam a névoa do renderer OpenGL; aceite
  valores em floats (ex.: `--fog-color 0.25 0.33 0.45`).
- `--filter` / `--exclude`: incluem ou ocultam objetos cujo ID ou nome contenha
  o texto informado (argumento pode ser repetido).
- `--save-objects`: grava um novo arquivo `EncTerrainXX.obj` criptografado com
  as posições atuais (ideal após mover objetos na interface interativa).
- `--edit-objects`: habilita a movimentação das instâncias exibidas (janela
  interativa obrigatória, remova `--no-show`).
- `--no-show`: evita abrir a janela interativa do Matplotlib.
- `--output`: caminho do arquivo PNG de saída.
- `--height-scale`: ajusta o fator aplicado ao formato clássico de altura (1.5
  por padrão, use 3.0 para o mapa de login).
- `--enum-path`: sobrescreve o caminho padrão de `_enum.h` usado para nomear os
  modelos lidos de `EncTerrainXX.obj`.
- `--export-objects`: salva a lista bruta de objetos em CSV.
- `--detailed-summary`: mostra estatísticas completas no terminal.
- `--summary-limit`: define quantos itens exibir nos resumos.

### Erros comuns

- **"Arquivo de altura (formato clássico) truncado."** – Esse aviso aparece
  quando o `TerrainHeight.OZB` encontrado é apenas um placeholder pequeno que
  acompanha alguns clientes modernos. Nesses casos o arquivo real está em
  `TerrainHeightNew.OZB`. A partir da versão atual o visualizador troca
  automaticamente para o formato novo sempre que detecta esse placeholder, mas
  se você estiver usando uma versão antiga basta marcar a opção **Forçar
  TerrainHeightNew** na interface (ou executar com `--extended-height`).

## Possíveis melhorias

Algumas ideias de evolução que podem deixar o visualizador mais prático no dia
a dia:

- **Histórico de edições com desfazer/refazer.** Registrar cada movimento e
  permitir desfazer etapas facilitaria experimentos mais longos sem medo de
  perder o progresso.
- **Snapping inteligente.** Adicionar alinhamento por grade, altura ou objetos
  vizinhos ajudaria a posicionar modelos complexos com mais precisão.
- **Edição de atributos e texturas.** Estender o editor para permitir ajustes
  diretos em `TerrainAttribute` e nas camadas de textura, atualizando os
  arquivos `.att/.map` no mesmo fluxo.
- **Comparação entre revisões.** Uma visualização lado a lado (ou modo diff)
  destacaria objetos adicionados/removidos entre duas pastas `WorldX`.
- **Renderização com recursos do jogo.** Carregar as texturas reais e aplicar
  shaders/iluminação aproximada deixaria a prévia ainda mais próxima do cliente.

## Exemplo

```bash
python terrain_viewer.py /caminho/para/Data/World7 --max-objects 500 --output world7.png --no-show
```

```bash
python terrain_viewer.py /caminho/para/Data/World7 --view-mode 2d --overlay height \
    --filter MODEL_TREE --export-json world7_trees.json --no-show
```

O script decodifica os arquivos criptografados usando as mesmas rotinas inline
(`MapFileDecrypt` e `BuxConvert`) vistas em `source/ZzzLodTerrain.cpp` e
`source/ZzzObject.cpp`. O arquivo `EncTerrainXX.obj` é interpretado exatamente
como no cliente: cada entrada informa o `type_id`, posição (`x`, `y`, `z`),
ângulos (`pitch`, `yaw`, `roll`) e escala. As posições são convertidas para o
mesmo espaço do terreno e plotadas acima da altura correspondente, garantindo
que a visualização reproduza o cenário esperado do jogo.
