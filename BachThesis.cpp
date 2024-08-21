#include "Util.h"

#include <iostream>
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

using namespace std;

int main()
{
    glfwInitVulkanLoader(vkGetInstanceProcAddr);

    if (!glfwInit()) {
        std::cerr << "GLFW could not init!" << std::endl;
        return -1;
    }
    ScopeExit cleanupGLFW(&glfwTerminate);
    

    GLFWwindow*  window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
    if (!window)
    {
        std::cerr << "GLFW could not init!" << std::endl;
        return -1;
    }


    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
    }
    return 0;
}
