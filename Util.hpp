#pragma once

#include <concepts>
#include <array>
#include <vector>
#include <fstream>

template<std::invocable F>
struct ScopeExit {
    F f;

    ScopeExit(F&& f) : f(f) {}
    ~ScopeExit() { f(); }
};

inline std::vector<char> readFullFile(const std::string& name) {
    const auto fileSize = [&]() { std::ifstream fileStream(name, std::ios::ate | std::ios::binary);
                                  if(!fileStream) throw std::runtime_error("Could not find file!");
                                  return fileStream.tellg(); }();
    std::vector<char> values(fileSize);
    std::ifstream fileStream(name, std::ios::binary);
    fileStream.read(values.data(), fileSize);
    return values;
}
