cmake_minimum_required (VERSION 3.24)

if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("BachThesis")
set(CMAKE_CXX_STANDARD_REQUIRED 20)
set(CMAKE_CXX_STANDARD 20)

include(FindVulkan)

if(false)
add_compile_options("/fsanitize=address")
add_link_options("/fsanitize=address")
if(true)
add_compile_definitions(_DISABLE_VECTOR_ANNOTATION=1)
add_compile_definitions(_DISABLE_STRING_ANNOTATION=1)
endif()
endif()

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

message(${Vulkan_GLSLC_EXECUTABLE})
add_custom_target(shaderTarget ALL ${Vulkan_GLSLC_EXECUTABLE} --target-env=vulkan1.2 -c "test.frag" "testMesh.mesh" "vertexWire.vert"
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/shader)
configure_file("shader/test.frag.spv" "shader/test.frag.spv" COPYONLY)
configure_file("shader/testMesh.spv" "shader/testMesh.spv" COPYONLY)
configure_file("shader/vertexWire.vert.spv" "shader/vertexWire.vert.spv" COPYONLY)
configure_file("assets/crystal.vtk" "assets/crystal.vtk" COPYONLY)

add_executable(BachThesis "BachThesis.cpp")
target_link_libraries(BachThesis PUBLIC imgui_lib)
