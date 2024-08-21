#pragma once

#include <concepts>
#include <array>
#include <vector>

template<std::invocable F>
struct ScopeExit {
    F f;
    
    ScopeExit(F &&f) : f(f) {}
    ~ScopeExit() { f(); }
};