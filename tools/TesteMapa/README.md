# TesteMapa

O **TesteMapa** é uma ferramenta experimental incluída no diretório `tools`
para navegar pelos mapas do cliente de MuOnline utilizando exatamente a mesma
lógica de carregamento do projeto principal (`Main`). Ele reaproveita a
infraestrutura compartilhada pelo `terrain_viewer`/`Terrain_Editor` para abrir
arquivos `EncTerrain`, carregar todos os objetos estáticos, aplicar iluminação
direcional e gerar uma experiência interativa semelhante a um mini jogo.

## Recursos

- Seleção da pasta *Data* raiz, com persistência opcional da preferência em
  arquivo de configuração local.
- Descoberta automática dos diretórios `WorldXX` e dos respectivos `ObjectXX`
  associados.
- Pré-visualização textual com estatísticas do terreno, objetos e atributos
  antes de iniciar a renderização.
- Execução do mesmo pipeline de renderização OpenGL utilizado pela `Main`, com
  câmera livre (`WASD` + mouse), névoa dinâmica e materiais derivados dos
  arquivos originais.
- Fallback automático para o renderizador Matplotlib clássico caso as
  dependências de OpenGL não estejam disponíveis.

## Dependências

As dependências são idênticas às do `terrain_viewer`, portanto consulte o
arquivo [`requirements.txt`](requirements.txt) para instalá-las rapidamente com
`pip`:

```bash
python -m pip install -r requirements.txt
```

O backend OpenGL (recomendado para a experiência completa) requer `moderngl`
, `pyglet` e uma GPU compatível com OpenGL 3.3.

## Uso

1. Defina a pasta base que contém os diretórios `WorldXX` do cliente:

   ```bash
   python teste_mapa.py --data-root /caminho/para/Data --remember-root
   ```

2. Liste os mundos reconhecidos dentro da pasta informada:

   ```bash
   python teste_mapa.py --list
   ```

3. Abra um mapa específico (por nome ou índice) utilizando o renderizador
   OpenGL e todos os objetos carregados:

   ```bash
   python teste_mapa.py --map World1
   # ou
   python teste_mapa.py --map-index 0
   ```

Durante a execução, utilize `W`, `A`, `S`, `D`, `Space` e `Ctrl` para mover a
câmera livremente, o mouse para olhar ao redor e `Shift` para acelerar. As
mesmas regras de iluminação e materiais empregadas na `Main` são utilizadas
para renderizar o terreno e os objetos estáticos.

### Opções adicionais

- `--renderer matplotlib`: força o modo clássico 3D em Matplotlib.
- `--no-show`: apenas prepara os dados sem abrir janela, útil para gerar
  estatísticas em ambientes sem suporte gráfico.
- `--output captura.png`: exporta uma imagem da visualização atual.
- `--fog-density` / `--fog-color`: ajustam a névoa do modo OpenGL.

Execute `python teste_mapa.py --help` para consultar a lista completa de
opções disponíveis.

## Estrutura

O script `teste_mapa.py` carrega o módulo `terrain_viewer.py` por reflexão e
reutiliza diretamente as funções `load_world_data` e `run_viewer`. Dessa forma,
o processo de leitura dos arquivos é exatamente o mesmo da `Main`, garantindo
que os objetos, iluminação e camadas de terreno reflitam o comportamento do
cliente oficial.

Os dados de configuração persistidos ficam em
`tools/TesteMapa/teste_mapa_config.json`.

## Licença

Este utilitário segue a mesma licença MIT do restante do repositório.
