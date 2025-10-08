# Terreno Visualisado

O **Terreno Visualisado** é um utilitário em C# que reproduz o pipeline de
carregamento dos arquivos `EncTerrain` usado pela Main.exe. Ele descriptografa os
arquivos `.map`, `.att`, `.obj` e `TerrainHeight*.OZB`, reconstrói as camadas de
tiles e reposiciona os objetos estáticos para facilitar a inspeção rápida dos
dados sem executar o cliente completo.

## Pré-requisitos

* [.NET 6 SDK](https://dotnet.microsoft.com/download/dotnet/6.0) ou superior.

## Compilação

Dentro da raiz do repositório execute:

```bash
dotnet build tools/terreno_visualisado/TerrenoVisualisado.csproj -c Release
# Interface gráfica WinForms (requer Windows)
dotnet build tools/terreno_visualisado/TerrenoVisualisado.Gui/TerrenoVisualisado.Gui.csproj -c Release
```

## Uso

```bash
# Exemplo mínimo apontando para a pasta Data/World1 do cliente

dotnet run --project tools/terreno_visualisado/TerrenoVisualisado.csproj -- \
    --world /caminho/para/Data/World1
```

Opções principais:

* `--world <pasta>`: diretório que contém `EncTerrain*.map/.att` e `TerrainHeight.OZB`.
* `--objects <pasta>`: diretório `ObjectX` com o `EncTerrain*.obj` correspondente (opcional).
* `--map <id>`: força o ID numérico usado nos arquivos `EncTerrain`.
* `--enum <arquivo>`: caminho para o `_enum.h` para exibir o nome textual dos objetos.
* `--height-scale <valor>`: fator aplicado ao `TerrainHeight.OZB` clássico.
* `--extended-height`: força o uso do `TerrainHeightNew.OZB` mesmo que o clássico exista.

A execução imprime um sumário contendo o ID detectado, o número total de tiles
por material, a contagem de atributos especiais e um ranking com os tipos de
objetos estáticos mais comuns. O utilitário também gera um arquivo JSON
`terrainsummary.json` no diretório atual contendo todas as amostras de altura,
camadas e objetos já alinhados ao terreno, permitindo integrações com outras
ferramentas.

## Interface gráfica

O projeto `TerrenoVisualisado.Gui` oferece uma interface WinForms para
visualizar rapidamente as camadas carregadas, alternar entre altura, layers,
atributos e alfa, além de sobrepor os objetos estáticos nas posições corretas.
A tela principal permite selecionar as pastas do mundo e dos objetos, informar
um `EnumModelType.eum` opcional e ajustar escala/ID do mapa. Após o carregamento
é possível exportar o mesmo JSON gerado pelo utilitário de linha de comando.
