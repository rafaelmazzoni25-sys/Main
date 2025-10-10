# Map Walker (C#)

Ferramenta Windows Forms que reconstrói os cenários do cliente MU Online
utilizando C# e OpenGL (via OpenTK). O aplicativo carrega os arquivos de
terreno `EncTerrain`, reconstrói a malha com as alturas originais, aplica as
texturas de tiles do jogo e permite navegar livremente com controles ao estilo
do cliente original.

## Requisitos

* Windows 10 ou superior
* [.NET 6 SDK](https://dotnet.microsoft.com/download) ou Visual Studio 2022 com
  suporte a .NET 6
* Drivers gráficos compatíveis com OpenGL 3.3

## Como compilar

```bash
cd tools/map_walker
dotnet restore
dotnet build
```

Também é possível abrir a solução `Main.sln` no Visual Studio e adicionar o
projeto `MapWalker.csproj` à solução.

## Como usar

1. Copie a pasta `Data` do cliente original (ou aponte diretamente para a pasta
   que contém os diretórios `WorldXX`).
2. Inicie o aplicativo `MapWalker`.
3. Clique em **Escolher...** e selecione a pasta que contém os dados.
4. Selecione o mapa e, opcionalmente, escolha a variante de atributos (base ou
   eventos).
5. Utilize **WASD** para movimentar, **botão direito** para girar a câmera e
   **Espaço/G/O** para alternar travamento no chão, grade e objetos.

O painel lateral permite ajustar velocidade de movimento, altura da câmera,
cor base e visibilidade de elementos auxiliares.
