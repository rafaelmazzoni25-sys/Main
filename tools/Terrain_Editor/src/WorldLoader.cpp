#include "terrain/WorldLoader.h"

#include "terrain/Types.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace terrain {

namespace {
constexpr float kDefaultClassicHeightScale = 1.5f;
constexpr float kMinHeightBias = -500.0f;

std::vector<std::uint8_t> readFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Não foi possível abrir " + path.string());
    }
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(file), {});
}

std::vector<std::uint8_t> mapFileDecrypt(const std::vector<std::uint8_t>& input) {
    static constexpr std::array<std::uint8_t, 16> kXorKey{
        0xD1, 0x73, 0x52, 0xF6, 0xD2, 0x9A, 0xCB, 0x27,
        0x3E, 0xAF, 0x59, 0x31, 0x37, 0xB3, 0xE7, 0xA2};
    std::vector<std::uint8_t> output(input.size());
    std::uint8_t wMapKey = 0x5E;
    for (std::size_t i = 0; i < input.size(); ++i) {
        const std::uint8_t byte = input[i];
        const std::uint8_t decrypted = static_cast<std::uint8_t>(((byte ^ kXorKey[i % kXorKey.size()]) - wMapKey) & 0xFF);
        output[i] = decrypted;
        wMapKey = static_cast<std::uint8_t>((byte + 0x3D) & 0xFF);
    }
    return output;
}

void buxConvert(std::vector<std::uint8_t>& data) {
    static constexpr std::array<std::uint8_t, 3> kBuxCode{0xFC, 0xCF, 0xAB};
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] ^= kBuxCode[i % kBuxCode.size()];
    }
}

std::optional<int> extractDigits(const std::string& text) {
    std::string digits;
    for (char ch : text) {
        if (std::isdigit(static_cast<unsigned char>(ch))) {
            digits.push_back(ch);
        }
    }
    if (digits.empty()) {
        return std::nullopt;
    }
    return std::stoi(digits);
}

std::filesystem::path findTerrainFile(
    const std::filesystem::path& directory,
    std::optional<int> desiredMapId,
    std::string_view extension) {
    if (!std::filesystem::is_directory(directory)) {
        throw std::runtime_error("Diretório inválido: " + directory.string());
    }

    std::vector<std::filesystem::path> matches;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto& path = entry.path();
        const auto filename = path.filename().string();
        if (filename.rfind("EncTerrain", 0) != 0) {
            continue;
        }
        if (!extension.empty() && path.extension() != extension) {
            continue;
        }
        matches.push_back(path);
    }

    if (matches.empty()) {
        throw std::runtime_error("Arquivos EncTerrain não encontrados em " + directory.string());
    }

    std::sort(matches.begin(), matches.end());

    if (desiredMapId) {
        for (const auto& candidate : matches) {
            const auto digits = extractDigits(candidate.filename().string());
            if (digits && *digits == *desiredMapId) {
                return candidate;
            }
        }
        throw std::runtime_error(
            "Nenhum arquivo EncTerrain correspondente ao mapa " + std::to_string(*desiredMapId) +
            " foi encontrado em " + directory.string());
    }

    return matches.front();
}

std::optional<std::filesystem::path> guessObjectFolder(const std::filesystem::path& worldDirectory) {
    const auto name = worldDirectory.filename().string();
    if (name.size() >= 5 && name.rfind("World", 0) == 0) {
        const std::string suffix = name.substr(5);
        auto candidate = worldDirectory.parent_path() / ("Object" + suffix);
        if (std::filesystem::is_directory(candidate)) {
            return candidate;
        }
        auto lower = worldDirectory.parent_path() / ("object" + suffix);
        if (std::filesystem::is_directory(lower)) {
            return lower;
        }
    }
    return std::nullopt;
}

float bilinearHeight(const TerrainData& terrain, float tileX, float tileY) {
    if (terrain.height.empty()) {
        return 0.0f;
    }

    const int maxIndex = static_cast<int>(kTerrainSize) - 1;
    float x = std::clamp(tileX, 0.0f, static_cast<float>(maxIndex));
    float y = std::clamp(tileY, 0.0f, static_cast<float>(maxIndex));

    const int xi = static_cast<int>(std::floor(x));
    const int yi = static_cast<int>(std::floor(y));
    const float xd = x - static_cast<float>(xi);
    const float yd = y - static_cast<float>(yi);

    auto sample = [&](int sx, int sy) {
        sx = std::clamp(sx, 0, maxIndex);
        sy = std::clamp(sy, 0, maxIndex);
        return terrain.height[static_cast<std::size_t>(sy) * kTerrainSize + static_cast<std::size_t>(sx)];
    };

    const float h00 = sample(xi, yi);
    const float h10 = sample(xi + 1, yi);
    const float h01 = sample(xi, yi + 1);
    const float h11 = sample(xi + 1, yi + 1);

    const float h0 = h00 * (1.0f - xd) + h10 * xd;
    const float h1 = h01 * (1.0f - xd) + h11 * xd;
    return h0 * (1.0f - yd) + h1 * yd;
}

}  // namespace

WorldLoader::WorldLoader() = default;

WorldData WorldLoader::load(const std::filesystem::path& worldDirectory, const LoadOptions& options) const {
    if (!std::filesystem::is_directory(worldDirectory)) {
        throw std::runtime_error("Diretório inválido: " + worldDirectory.string());
    }

    LoadOptions normalized = options;
    if (!normalized.mapId) {
        normalized.mapId = inferMapId(worldDirectory);
    }

    const auto attributesPath = resolveAttributesPath(worldDirectory, normalized.mapId);
    const auto mappingPath = resolveMappingPath(worldDirectory, normalized.mapId);
    const auto objectsPath = resolveObjectsPath(worldDirectory, normalized.objectRoot, normalized.mapId);
    const auto heightPath = resolveHeightPath(worldDirectory, normalized.forceExtendedHeight);

    const auto modelNames = loadModelNames(normalized.enumPath);

    int attributeMapId = -1;
    int mappingMapId = -1;
    TerrainData terrain = loadTerrain(
        attributesPath,
        mappingPath,
        heightPath,
        normalized.forceExtendedHeight,
        normalized.heightScale,
        attributeMapId,
        mappingMapId);

    int objectsVersion = 0;
    int objectsMapId = -1;
    std::vector<ObjectInstance> objects = loadObjects(objectsPath, modelNames, objectsVersion, objectsMapId);

    const int resolvedMapId = normalized.mapId.value_or(
        attributeMapId >= 0 ? attributeMapId : (mappingMapId >= 0 ? mappingMapId : objectsMapId));

    // Ajusta a posição vertical dos objetos para acompanhar o terreno carregado.
    for (auto& object : objects) {
        const float tileX = object.position[0] / kTerrainScale;
        const float tileY = object.position[1] / kTerrainScale;
        const float height = bilinearHeight(terrain, tileX, tileY);
        object.position = {object.position[0], height, object.position[1]};
    }

    WorldData world;
    world.worldPath = worldDirectory;
    world.objectsPath = objectsPath;
    world.mapId = resolvedMapId;
    world.objectVersion = objectsVersion;
    world.terrain = std::move(terrain);
    world.objects = std::move(objects);
    return world;
}

WorldLoader::ModelNameTable WorldLoader::loadModelNames(const std::optional<std::filesystem::path>& enumPath) {
    ModelNameTable table;
    if (!enumPath || !std::filesystem::exists(*enumPath)) {
        return table;
    }

    std::ifstream file(*enumPath);
    if (!file) {
        return table;
    }

    std::string line;
    std::regex pattern(R"(^\s*(MODEL_[A-Z0-9_]+)\s*=\s*([^,]+).*$)");
    while (std::getline(file, line)) {
        std::smatch match;
        if (!std::regex_match(line, match, pattern)) {
            continue;
        }
        const std::string name = match[1].str();
        const std::string valueText = match[2].str();

        int value = 0;
        std::string trimmed = valueText;
        trimmed.erase(trimmed.begin(), std::find_if(trimmed.begin(), trimmed.end(), [](unsigned char ch) {
            return !std::isspace(ch);
        }));
        trimmed.erase(std::find_if(trimmed.rbegin(), trimmed.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), trimmed.end());

        if (trimmed.rfind("0x", 0) == 0 || trimmed.rfind("0X", 0) == 0) {
            value = std::stoi(trimmed, nullptr, 16);
        } else {
            value = std::stoi(trimmed);
        }
        table.emplace(value, name);
    }
    return table;
}

std::optional<int> WorldLoader::inferMapId(const std::filesystem::path& worldDirectory) {
    for (auto it = worldDirectory; !it.empty(); it = it.parent_path()) {
        const auto name = it.filename().string();
        if (name.rfind("World", 0) == 0) {
            if (const auto digits = extractDigits(name)) {
                return digits;
            }
        }
    }
    return std::nullopt;
}

std::filesystem::path WorldLoader::resolveAttributesPath(const std::filesystem::path& worldDirectory, std::optional<int> mapId) {
    return findTerrainFile(worldDirectory, mapId, ".att");
}

std::filesystem::path WorldLoader::resolveMappingPath(const std::filesystem::path& worldDirectory, std::optional<int> mapId) {
    return findTerrainFile(worldDirectory, mapId, ".map");
}

std::filesystem::path WorldLoader::resolveObjectsPath(
    const std::filesystem::path& worldDirectory,
    const std::optional<std::filesystem::path>& objectRoot,
    std::optional<int> mapId) {
    std::vector<std::filesystem::path> candidates;
    if (objectRoot) {
        candidates.push_back(*objectRoot);
    }
    if (const auto guess = guessObjectFolder(worldDirectory)) {
        candidates.push_back(*guess);
    }
    candidates.push_back(worldDirectory);

    for (const auto& dir : candidates) {
        try {
            return findTerrainFile(dir, mapId, ".obj");
        } catch (const std::exception&) {
            continue;
        }
    }
    throw std::runtime_error("Arquivo EncTerrain*.obj não encontrado. Informe --objects para especificar a pasta correta.");
}

std::filesystem::path WorldLoader::resolveHeightPath(const std::filesystem::path& worldDirectory, bool preferExtended) {
    const auto classic = worldDirectory / "TerrainHeight.OZB";
    const auto extended = worldDirectory / "TerrainHeightNew.OZB";

    const std::uintmax_t classicMin = 4 + 1080 + kTerrainSize * kTerrainSize;
    const std::uintmax_t extendedMin = 4 + 54 + kTerrainSize * kTerrainSize * 3;

    const bool classicValid = std::filesystem::exists(classic) && std::filesystem::file_size(classic) >= classicMin;
    const bool extendedValid = std::filesystem::exists(extended) && std::filesystem::file_size(extended) >= extendedMin;

    if (preferExtended && extendedValid) {
        return extended;
    }
    if (classicValid) {
        return classic;
    }
    if (extendedValid) {
        return extended;
    }
    throw std::runtime_error("Arquivos TerrainHeight.OZB ou TerrainHeightNew.OZB não encontrados ou inválidos.");
}

TerrainData WorldLoader::loadTerrain(
    const std::filesystem::path& attributesPath,
    const std::filesystem::path& mappingPath,
    const std::filesystem::path& heightPath,
    bool forceExtendedHeight,
    std::optional<float> heightScale,
    int& outAttributeMapId,
    int& outMappingMapId) {
    TerrainData terrain;
    terrain.height.resize(kTerrainSize * kTerrainSize);
    terrain.layer1.resize(kTerrainSize * kTerrainSize);
    terrain.layer2.resize(kTerrainSize * kTerrainSize);
    terrain.alpha.resize(kTerrainSize * kTerrainSize);
    terrain.attributes.resize(kTerrainSize * kTerrainSize);

    // Atributos
    {
        auto raw = readFile(attributesPath);
        auto decrypted = mapFileDecrypt(raw);
        buxConvert(decrypted);
        if (decrypted.size() != 131076 && decrypted.size() != 65540) {
            throw std::runtime_error("Tamanho inesperado para arquivo de atributos: " + attributesPath.string());
        }
        outAttributeMapId = static_cast<int>(decrypted[1]);
        const std::size_t offset = 4;
        if (decrypted.size() == 65540) {
            for (std::size_t i = 0; i < kTerrainSize * kTerrainSize; ++i) {
                terrain.attributes[i] = decrypted[offset + i];
            }
        } else {
            const auto* buffer = reinterpret_cast<const std::uint16_t*>(decrypted.data() + offset);
            for (std::size_t i = 0; i < kTerrainSize * kTerrainSize; ++i) {
                terrain.attributes[i] = buffer[i];
            }
        }
    }

    // Mapping
    {
        auto raw = readFile(mappingPath);
        auto decrypted = mapFileDecrypt(raw);
        if (decrypted.size() < 2 + kTerrainSize * kTerrainSize * 3) {
            throw std::runtime_error("Arquivo EncTerrain.map truncado: " + mappingPath.string());
        }
        outMappingMapId = static_cast<int>(decrypted[1]);
        std::size_t ptr = 2;
        for (std::size_t i = 0; i < kTerrainSize * kTerrainSize; ++i) {
            terrain.layer1[i] = decrypted[ptr++];
        }
        for (std::size_t i = 0; i < kTerrainSize * kTerrainSize; ++i) {
            terrain.layer2[i] = decrypted[ptr++];
        }
        for (std::size_t i = 0; i < kTerrainSize * kTerrainSize; ++i) {
            terrain.alpha[i] = static_cast<float>(decrypted[ptr++]) / 255.0f;
        }
    }

    // Altura
    {
        auto raw = readFile(heightPath);
        if (raw.size() < 4) {
            throw std::runtime_error("Arquivo de altura muito pequeno: " + heightPath.string());
        }
        const std::vector<std::uint8_t> payload(raw.begin() + 4, raw.end());
        const bool extended = forceExtendedHeight || (heightPath.filename() == "TerrainHeightNew.OZB");
        if (!extended) {
            const std::size_t expected = 1080 + kTerrainSize * kTerrainSize;
            if (payload.size() < expected) {
                throw std::runtime_error("Arquivo de altura clássico truncado: " + heightPath.string());
            }
            const std::uint8_t* heightBytes = payload.data() + 1080;
            const float scale = heightScale.value_or(kDefaultClassicHeightScale);
            for (std::size_t i = 0; i < kTerrainSize * kTerrainSize; ++i) {
                terrain.height[i] = static_cast<float>(heightBytes[i]) * scale;
            }
        } else {
            const std::size_t headerSize = 14 + 40;
            const std::size_t expected = headerSize + kTerrainSize * kTerrainSize * 3;
            if (payload.size() < expected) {
                throw std::runtime_error("Arquivo TerrainHeightNew.OZB truncado: " + heightPath.string());
            }
            const std::uint8_t* pixel = payload.data() + headerSize;
            for (std::size_t i = 0; i < kTerrainSize * kTerrainSize; ++i) {
                const std::uint32_t b = pixel[i * 3 + 0];
                const std::uint32_t g = pixel[i * 3 + 1];
                const std::uint32_t r = pixel[i * 3 + 2];
                const std::uint32_t value = (r << 16) | (g << 8) | b;
                terrain.height[i] = static_cast<float>(value) + kMinHeightBias;
            }
        }
    }

    return terrain;
}

std::vector<ObjectInstance> WorldLoader::loadObjects(
    const std::filesystem::path& objectsPath,
    const ModelNameTable& modelNames,
    int& outVersion,
    int& outMapId) {
    std::vector<ObjectInstance> objects;

    auto raw = readFile(objectsPath);
    auto decrypted = mapFileDecrypt(raw);
    if (decrypted.size() < 4) {
        throw std::runtime_error("Arquivo EncTerrain.obj truncado: " + objectsPath.string());
    }

    std::size_t ptr = 0;
    outVersion = decrypted[ptr++];
    outMapId = decrypted[ptr++];
    const std::int16_t count = static_cast<std::int16_t>(decrypted[ptr] | (decrypted[ptr + 1] << 8));
    ptr += 2;

    if (count < 0) {
        throw std::runtime_error("Contagem negativa de objetos em " + objectsPath.string());
    }

    objects.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        if (ptr + 34 > decrypted.size()) {
            throw std::runtime_error("Dados de objeto insuficientes em " + objectsPath.string());
        }
        const std::int16_t typeId = static_cast<std::int16_t>(decrypted[ptr] | (decrypted[ptr + 1] << 8));
        ptr += 2;

        std::array<float, 3> position{};
        std::memcpy(position.data(), decrypted.data() + ptr, sizeof(float) * 3);
        ptr += sizeof(float) * 3;

        std::array<float, 3> angles{};
        std::memcpy(angles.data(), decrypted.data() + ptr, sizeof(float) * 3);
        ptr += sizeof(float) * 3;

        float scale = 1.0f;
        std::memcpy(&scale, decrypted.data() + ptr, sizeof(float));
        ptr += sizeof(float);

        ObjectInstance object;
        object.typeId = typeId;
        object.position = position;
        object.rotation = angles;
        object.scale = scale;
        if (const auto it = modelNames.find(typeId); it != modelNames.end()) {
            object.typeName = it->second;
        }
        objects.emplace_back(std::move(object));
    }

    return objects;
}

}  // namespace terrain
