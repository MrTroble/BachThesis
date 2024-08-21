# CMakeList.txt : CMake project for BachThesis, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.12)
set(CXX_STANDARD 20)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("BachThesis")

include(FindVulkan)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2
)
FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG        3.4
)
FetchContent_MakeAvailable(googletest glfw)

add_library(imgui_lib "submodules/imgui/imgui.cpp" "submodules/imgui/imgui_demo.cpp" 
    "submodules/imgui/imgui_draw.cpp" "submodules/imgui/imgui_tables.cpp" "submodules/imgui/imgui_widgets.cpp"
    "submodules/imgui/backends/imgui_impl_vulkan.cpp" "submodules/imgui/backends/imgui_impl_glfw.cpp")
target_include_directories(imgui_lib PUBLIC "submodules/imgui")
target_link_libraries(imgui_lib PUBLIC glfw Vulkan::Vulkan)

add_executable(BachThesis "BachThesis.cpp" "BachThesis.h")
target_link_libraries(BachThesis PUBLIC imgui_lib)

