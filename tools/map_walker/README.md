# Map Walker

`map_walker.py` é uma ferramenta independente escrita em Python que lê os
mesmos arquivos `EncTerrain` do cliente (altura, atributos e objetos
estáticos) e gera uma visualização 3D interativa dos mapas. A interface
permite escolher o mundo desejado, ajustar opções gráficas e de navegação e
caminhar pelo terreno com controles semelhantes aos do cliente original.

## Recursos

- **Seleção de mundos por interface**: escolha a pasta `Data` e o mapa a ser
  carregado a partir de uma lista que cruza o índice do mundo com o nome
  conhecido no cliente.
- **Renderização em tempo real via OpenGL**: reconstrói a malha 256×256 do
  terreno, calcula normais e aplica iluminação direcional simples para dar
  noção de profundidade.
- **Texturas originais de terreno**: lê `EncTerrainXX.map`, monta um atlas com
  `Tile*.jpg/.tga` e mistura as camadas e alfa da mesma forma que o cliente
  oficial.
- **Navegação com colisão**: mantém a câmera alinhada ao chão usando a mesma
  interpolação bilinear de `RequestTerrainHeight` e respeita os flags
  `TW_NOMOVE`/`TW_NOGROUND` do arquivo de atributos.
- **Marcadores de objetos**: exibe gizmos verticais nos pontos definidos em
  `EncTerrainXX.obj`, facilitando a inspeção de portais, estruturas e
  outros objetos estáticos.
- **Painel de configurações**: ajusta altura da câmera, velocidade, cores do
  terreno, grade de depuração e variantes de arquivos (`EncTerrainXX`,
  `EncTerrainXX1`, etc.) sem reiniciar o programa.

## Dependências

Instale os pacotes com `pip` em um ambiente Python 3.9+:

```bash
pip install PySide6 PyOpenGL numpy
```

Opcionalmente é possível instalar `PyOpenGL_accelerate` para melhorar a
performance em GPUs integradas.

### Requisitos de OpenGL

O `Map Walker` solicita um contexto **OpenGL 2.1 (perfil compatível)** ao Qt.
Drivers desatualizados ou que ofereçam apenas OpenGL 1.x/ES farão com que a
renderização seja desativada. Nesses casos, a janela exibirá uma mensagem em
vermelho com o motivo do erro e instruções para atualizar o driver ou utilizar
outro computador/placa de vídeo.

## Como usar

1. Execute o script apontando para a raiz do repositório ou para a pasta
   que contém `Data/`:
   ```bash
   python tools/map_walker/map_walker.py --data ../Cliente/Data
   ```
2. Caso o argumento `--data` não seja informado, use o botão **Escolher
   pasta de dados** no painel lateral para selecionar a raiz contendo as
   subpastas `WorldX` e arquivos `EncTerrain`.
3. Escolha o mapa na lista. A ferramenta procura os arquivos na seguinte
   ordem para cada tipo:
   - `EncTerrain{N}.att`, `EncTerrain{N*10+1}.att`, `EncTerrain{N*10+2}.att`
   - `EncTerrain{N}.map`, `EncTerraintest{N}.map`
   - `EncTerrain{N}.obj`
4. Use os controles de teclado e mouse para navegar (veja abaixo). A barra
   de status exibe a posição atual, altura e informações do tile.

## Controles

- **Mouse direito**: arrasta para girar a câmera.
- **W/S**: mover para frente/trás.
- **A/D**: mover lateralmente.
- **Q/E**: subir/descer manualmente (apenas quando o modo "travar no chão"
  estiver desativado).
- **Shift**: acelera temporariamente.
- **Barra de espaço**: alterna a trava da câmera no terreno.
- **G**: mostra/oculta a grade.
- **O**: mostra/oculta marcadores de objetos.

## Painel de configurações

O painel lateral oferece os seguintes ajustes:

| Opção | Descrição |
| ----- | --------- |
| Pasta de dados | Caminho base contendo a pasta `Data`. |
| Mapa | Lista dos mundos detectados, exibindo índice e nome legível. |
| Variante de atributo | Escolhe qual sufixo usar (`padrão`, `x10+1`, `x10+2`). |
| Altura da câmera | Offset em relação ao solo. |
| Velocidade | Velocidade base de movimento. |
| Cor base | Três controles RGB para ajustar o multiplicador aplicado às texturas. |
| Travar no chão | Define se a câmera acompanha o terreno. |
| Mostrar grade | Desenha uma grade 2D sobre o mapa. |
| Mostrar objetos | Habilita os marcadores carregados de `EncTerrainXX.obj`. |

Todas as alterações são aplicadas imediatamente.

## Estrutura do código

- `TerrainRepository` varre a pasta `Data` e resolve os caminhos de cada
  recurso (altura, atributos, mapeamento e objetos), aplicando `MapFileDecrypt`
  e `BuxConvert` como no cliente original antes de converter para `numpy`.
- `TerrainData` encapsula a malha reconstruída, atributos, funções de
  interpolação (`sample_height`) e utilidades para consulta de tiles.
- `TerrainView` é um `QOpenGLWidget` que compila um shader simples, cria VBOs
  e renderiza tanto a malha quanto os marcadores e o atlas de texturas.
- `ControlPanel` (subclasse de `QWidget`) expõe as opções e emite sinais Qt
  para atualizar a visualização.
- `MapWalkerWindow` coordena tudo, atualiza a barra de status e mantém o
  loop de movimentação com `QTimer`.

## Limitações conhecidas

- O shader atual aplica apenas uma luz direcional fixa. Reflexos, animacões
  de água e gramado dinâmico ainda não foram reproduzidos.
- Modelos BMD não são carregados; apenas um marcador vertical indica a
  posição de cada objeto.
- A ferramenta não substitui o cliente original para gameplay, servindo
  apenas para inspeção visual e validação de atributos.
- Mapas com arquivos ausentes serão ignorados na lista até que os recursos
  necessários sejam copiados para a pasta `Data`.
- Caso o driver não suporte OpenGL 2.1, a renderização é desativada e uma
  mensagem explicando o problema é apresentada diretamente no visualizador.

## Licença

O código segue a mesma licença MIT do repositório principal.
