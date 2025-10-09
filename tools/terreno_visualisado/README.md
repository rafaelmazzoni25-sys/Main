# Terreno Visualisado

O **Terreno Visualisado** é um utilitário em C# que reproduz o pipeline de
carregamento dos arquivos `EncTerrain` usado pela Main.exe. Ele descriptografa os
arquivos `.map`, `.att`, `.obj` e `TerrainHeight*.OZB`, reconstrói as camadas de
tiles e reposiciona os objetos estáticos para facilitar a inspeção rápida dos
dados sem executar o cliente completo.

## Pré-requisitos

* [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0) ou superior.

## Compilação

### Linha de comando (dotnet CLI)

Dentro da raiz do repositório execute:

```bash
# Linux/macOS (CLI)
dotnet build tools/terreno_visualisado/TerrenoVisualisado.csproj -c Release
# Interface gráfica WinForms (requer Windows)
dotnet build tools/terreno_visualisado/TerrenoVisualisado.Gui/TerrenoVisualisado.Gui.csproj -c Release

# Windows (PowerShell ou Prompt) — utilize barras invertidas para evitar que o MSBuild
# interprete o caminho como uma opção:
dotnet build "tools\terreno_visualisado\TerrenoVisualisado.csproj" -c Release
dotnet build "tools\terreno_visualisado\TerrenoVisualisado.Gui\TerrenoVisualisado.Gui.csproj" -c Release
```

Caso esteja abrindo o projeto a partir de um diretório com espaços no Windows,
lembre-se de envolver o caminho entre aspas (`"..."`). Exemplo:

```powershell
dotnet build "C:\\Users\\Rafael Mazzoni\\Downloads\\Projeto Unreal Estudos\\Ferramentas GFX\\Main-master\\Main-master\\tools\\terreno_visualisado\\TerrenoVisualisado.csproj" -c Release
dotnet build "C:\\Users\\Rafael Mazzoni\\Downloads\\Projeto Unreal Estudos\\Ferramentas GFX\\Main-master\\Main-master\\tools\\terreno_visualisado\\TerrenoVisualisado.Gui\\TerrenoVisualisado.Gui.csproj" -c Release
```

Antes de compilar pela primeira vez, execute um `dotnet restore` no mesmo
diretório do comando de build para que as dependências do SDK sejam baixadas.

Se o `dotnet restore` falhar dizendo que não encontrou pacotes como `OpenTK`
ou `OpenTK.WinForms`, confirme que o feed oficial do NuGet (`nuget.org`) está
habilitado:

```powershell
dotnet nuget list source
# Caso "nuget.org" não apareça, adicione-o:
dotnet nuget add source https://api.nuget.org/v3/index.json --name nuget.org
```

O arquivo `nuget.config` incluído na pasta da ferramenta já define o feed
oficial do NuGet como fonte padrão. Use os comandos acima apenas se sua
instalação do .NET ignorar esse arquivo ou ainda não reconhecer o `nuget.org`.

### Visual Studio 2022+

Foi adicionada a solução `TerrenoVisualisado.sln` dentro da pasta da
ferramenta. Abra-a no Visual Studio 2022 (ou superior) e aceite o diálogo para
restaurar os pacotes (`Build > Restore NuGet Packages`). Depois disso os três
projetos (`TerrenoVisualisado`, `TerrenoVisualisado.Core` e
`TerrenoVisualisado.Gui`) aparecerão carregados e prontos para compilar nas
configurações Debug/Release.

## Uso

```bash
# Exemplo mínimo apontando para a pasta Data/World1 do cliente

# Linux/macOS
dotnet run --project tools/terreno_visualisado/TerrenoVisualisado.csproj -- \
    --world /caminho/para/Data/World1

# Windows
dotnet run --project tools\terreno_visualisado\TerrenoVisualisado.csproj -- \
    --world C:\\caminho\\para\\Data\\World1
```

Opções principais:

* `--world <pasta>`: diretório que contém `EncTerrain*.map/.att` e `TerrainHeight.OZB`.
* `--objects <pasta>`: diretório `ObjectX` correspondente (opcional; na ausência deste parâmetro, o visualizador tenta localizar automaticamente uma pasta `Object*` irmã do `World*`).
* `--map <id>`: força o ID numérico usado nos arquivos `EncTerrain`.
* `--enum <arquivo>`: caminho para o `_enum.h` para exibir o nome textual dos objetos.
* `--height-scale <valor>`: fator aplicado ao `TerrainHeight.OZB` clássico.
* `--extended-height`: força o uso do `TerrainHeightNew.OZB` mesmo que o clássico exista.
* `--skip-attributes`: ignora o carregamento do `EncTerrain*.att` (útil para mundos com arquivos corrompidos ou ausentes).

A execução imprime um sumário contendo o ID detectado, o número total de tiles
por material, a contagem de atributos especiais (quando carregados) e um ranking com os tipos de
objetos estáticos mais comuns. O utilitário também gera um arquivo JSON
`terrainsummary.json` no diretório atual contendo todas as amostras de altura,
camadas e objetos já alinhados ao terreno, permitindo integrações com outras
ferramentas.

## Interface gráfica

O projeto `TerrenoVisualisado.Gui` oferece uma interface WinForms para
visualizar rapidamente as camadas carregadas, alternar entre altura, layers,
atributos e alfa, além de sobrepor os objetos estáticos nas posições corretas.
A tela principal permite selecionar as pastas do mundo e dos objetos, informar
um `EnumModelType.eum` opcional e ajustar escala/ID do mapa. Ao escolher um
`World*`, o campo de objetos é preenchido automaticamente com a pasta `Object*`
irmã correspondente (quando encontrada). Após o carregamento
é possível exportar o mesmo JSON gerado pelo utilitário de linha de comando.
O repositório inclui uma versão gerada automaticamente em `data/EnumModelType.eum`
— utilize-a diretamente ou regenere o arquivo executando
`python tools/terreno_visualisado/scripts/generate_enum_modeltype.py` a partir
da raiz do projeto. Um checkbox específico permite habilitar ou desabilitar a
leitura do `EncTerrain*.att`, refletindo imediatamente no sumário e nos modos
de visualização disponíveis.

Além do preview 2D tradicional, a interface inclui uma aba **Visualização 3D**
com um renderizador OpenGL. O heightmap é convertido para uma malha completa,
as camadas de tiles são mescladas em um atlas idêntico ao cliente e o resultado
é desenhado com iluminação difusa e câmera orbitando livremente, permitindo
inspecionar o terreno de forma fiel. A partir desta aba também é possível
instanciar os
modelos BMD (malhas e animações), desenhando cada objeto com seus materiais,
animações e estados corretos para aproximar ainda mais o comportamento do
cliente original. A iluminação direcional foi recalibrada com perfis por mapa,
ajustando céu, luz ambiente, especularidade e emissivos para reproduzir o clima
do jogo — inclusive com modo "voo"/primeira pessoa para caminhar pelo mapa
como dentro do cliente.

#### Controles da câmera 3D

* **Orbit (padrão)**
  * Botão esquerdo: orbitar o alvo.
  * Botão direito: deslocar lateralmente/verticalmente.
  * Roda do mouse: aproximação/afastamento.
  * Teclas `W`/`S` ou setas ↑/↓: ajustam a inclinação.
  * Teclas `A`/`D` ou setas ←/→: ajustam o azimute.
  * Teclas `Q`/`E` ou Page Up/Page Down: variam a distância.
* **Modo voo** (`F` para alternar)
  * Botão esquerdo: controla o olhar (yaw/pitch).
  * `W`/`S`: move para frente/trás.
  * `A`/`D`: strafe esquerda/direita.
  * `Q`/`E`: sobe/desce.
  * `Shift`: acelera, `Ctrl`: desacelera.
  * Roda do mouse: ajusta a velocidade base.
* `R`: redefine a câmera para o enquadramento padrão do mapa atual.

### Instanciação dos modelos BMD

A Visualização 3D oferece um pipeline completo de instanciação dos modelos BMD,
carregando meshes, animações e estados associados a cada objeto. Durante o
desenho, cada instância reaplica seus materiais, reproduz as animações e
respeita os estados definidos, garantindo uma representação fiel aos recursos
vistos dentro do cliente original.
