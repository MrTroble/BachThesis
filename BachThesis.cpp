#include "Util.h"

#include <iostream>
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

using namespace std;

struct IContext {
    vk::Instance instance;
    // Device Creation
    vk::Device device;
    vk::PhysicalDevice physicalDevice;
    std::vector<vk::QueueFamilyProperties> queueFamilyProperties;
    uint32_t primaryFamilyIndex;
    // Surface
    vk::SurfaceKHR surface;
};

int main()
{
    if (!glfwInit()) {
        std::cerr << "GLFW could not init!" << std::endl;
        return -1;
    }
    const ScopeExit cleanupGLFW(&glfwTerminate);

    const vk::ApplicationInfo applicationInfo("Test", 0, "Test", 0);

    uint32_t instanceExtensionCount;
    const auto listOfExtensions = glfwGetRequiredInstanceExtensions(&instanceExtensionCount);
    IContext icontext;

    const std::array layers{ "VK_LAYER_KHRONOS_validation" };

    const vk::InstanceCreateInfo createInstanceInfo({}, &applicationInfo, layers.size(), layers.data(),
        instanceExtensionCount, listOfExtensions);
    icontext.instance = vk::createInstance(createInstanceInfo);
    const ScopeExit cleanInstance([&]() { icontext.instance.destroy(); });

    const auto physicalDevices = icontext.instance.enumeratePhysicalDevices();
    for (const auto physicalDevice : physicalDevices)
    {
        auto queueFamilies = physicalDevice.getQueueFamilyProperties();
        size_t familyIndex = 0;
        bool found = false;
        for (; familyIndex < queueFamilies.size(); familyIndex++) {
            if ((queueFamilies[familyIndex].queueFlags & vk::QueueFlagBits::eGraphics) &&
                glfwGetPhysicalDevicePresentationSupport(icontext.instance, physicalDevice, familyIndex)) {
                found = true;
                icontext.physicalDevice = physicalDevice;
                icontext.primaryFamilyIndex = familyIndex;
                icontext.queueFamilyProperties = std::move(queueFamilies);
                break;
            }
        }
        if (found)
            break;
    }

    const std::array queuePriorities{ 1.0f };
    const vk::DeviceQueueCreateInfo queueCreateInfo({}, icontext.primaryFamilyIndex, queuePriorities);

    const vk::DeviceCreateInfo deviceCreateInfo({}, queueCreateInfo);
    icontext.device = icontext.physicalDevice.createDevice(deviceCreateInfo);
    const ScopeExit cleanDevice([&]() { icontext.device.destroy(); });

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
    if (!window)
    {
        std::cerr << "GLFW could not init!" << std::endl;
        return -1;
    }

    const vk::Result result = (vk::Result)glfwCreateWindowSurface(icontext.instance, window, nullptr, (VkSurfaceKHR*)&icontext.surface);
    if (result != vk::Result::eSuccess) {
        std::cerr << "GLFW Surface creation failed! With VkResult " << vk::to_string(result) << std::endl;
        return -1;
    }
    const ScopeExit cleanSurface([&]() { icontext.instance.destroySurfaceKHR(icontext.surface); });


    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
    }
    return 0;
}
