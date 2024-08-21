#pragma once

#include <concepts>

template<std::invocable F>
struct ScopeExit {
    F f;
    
    ScopeExit(F &&f) : f(f) {}
    ~ScopeExit() { f(); }
};