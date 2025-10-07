#include "terrain/Application.h"

#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

struct Arguments {
    std::optional<std::filesystem::path> worldPath;
    terrain::LoadOptions options;
};

void printUsage() {
    std::cout << "Terrain_Editor\n\n"
              << "Uso: terrain_editor --world <caminho_para_WorldX> [opções]\n\n"
              << "Opções:\n"
              << "  --world <pasta>          Diretório contendo EncTerrain*.map/.att e TerrainHeight.OZB\n"
              << "  --map <id>              Sobrescreve o ID do mapa inferido pelo nome da pasta\n"
              << "  --objects <pasta>       Diretório ObjectX com EncTerrain*.obj correspondente\n"
              << "  --height-scale <valor>  Aplica fator de escala para alturas no formato clássico\n"
              << "  --extended-height       Força leitura do formato TerrainHeightNew.OZB\n"
              << "  --enum <arquivo>        Caminho opcional para _enum.h para resolver nomes de objetos\n"
              << "\n";
}

std::optional<Arguments> parseArguments(int argc, char** argv) {
    Arguments args;
    std::vector<std::string> tokens(argv + 1, argv + argc);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        const std::string& token = tokens[i];
        if (token == "--world" && i + 1 < tokens.size()) {
            args.worldPath = std::filesystem::path(tokens[++i]);
        } else if (token == "--map" && i + 1 < tokens.size()) {
            args.options.mapId = std::stoi(tokens[++i]);
        } else if (token == "--objects" && i + 1 < tokens.size()) {
            args.options.objectRoot = std::filesystem::path(tokens[++i]);
        } else if (token == "--height-scale" && i + 1 < tokens.size()) {
            args.options.heightScale = std::stof(tokens[++i]);
        } else if (token == "--extended-height") {
            args.options.forceExtendedHeight = true;
        } else if (token == "--enum" && i + 1 < tokens.size()) {
            args.options.enumPath = std::filesystem::path(tokens[++i]);
        } else if (token == "--help" || token == "-h") {
            printUsage();
            return std::nullopt;
        } else {
            std::cerr << "Parâmetro desconhecido: " << token << "\n";
            printUsage();
            return std::nullopt;
        }
    }

    if (!args.worldPath) {
        std::cerr << "É necessário informar --world com o diretório do mapa.\n";
        printUsage();
        return std::nullopt;
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    const auto parsed = parseArguments(argc, argv);
    if (!parsed) {
        return 1;
    }

    terrain::Application app;
    if (!app.initialize()) {
        std::cerr << "Falha ao inicializar o Terrain_Editor." << std::endl;
        return 1;
    }

    if (!app.loadWorld(*parsed->worldPath, parsed->options)) {
        return 1;
    }

    app.run();
    return 0;
}
