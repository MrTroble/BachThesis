#include "Util.hpp"
#include "Context.hpp"
#include "CommandBuffer.hpp"

#include <iostream>
#include <GLFW/glfw3.h>

using namespace std;

inline void recreateSwapchain(IContext& icontext) {
    if (icontext.swapchain)
        icontext.device.destroySwapchainKHR(icontext.swapchain);
    const std::array queueFamiliesInSwapchain = { icontext.primaryFamilyIndex };
    const vk::SwapchainCreateInfoKHR swapchainCreateInfo({}, icontext.surface, 3, vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear, icontext.currentExtent,
        1, vk::ImageUsageFlagBits::eColorAttachment, vk::SharingMode::eExclusive, queueFamiliesInSwapchain);
    icontext.swapchain = icontext.device.createSwapchainKHR(swapchainCreateInfo);
    icontext.swapchainImages = icontext.device.getSwapchainImagesKHR(icontext.swapchain);
}

inline void checkErrorOrRecreate(vk::Result result, IContext& context) {
    if (result == vk::Result::eSuboptimalKHR) {
        recreateSwapchain(context);
        return;
    }
    if (result != vk::Result::eSuccess) {
        std::cerr << "Vulkan Error with " << vk::to_string(result) << std::endl;
        throw std::runtime_error("Vulkan Error!");
    }
}

int main()
{
    if (!glfwInit()) {
        std::cerr << "GLFW could not init!" << std::endl;
        return -1;
    }
    const ScopeExit cleanupGLFW(&glfwTerminate);

    const vk::ApplicationInfo applicationInfo("Test", 0, "Test", 0, VK_API_VERSION_1_2);

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

    const std::array extensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_EXT_MESH_SHADER_EXTENSION_NAME };
    const vk::DeviceCreateInfo deviceCreateInfo({}, queueCreateInfo, {}, extensions);
    icontext.device = icontext.physicalDevice.createDevice(deviceCreateInfo);
    const ScopeExit cleanDevice([&]() { icontext.device.destroy(); });

    icontext.currentExtent = vk::Extent2D{ 640u, 480u };

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(icontext.currentExtent.width, icontext.currentExtent.height,
        applicationInfo.pApplicationName, NULL, NULL);
    if (!window)
    {
        std::cerr << "GLFW could not init!" << std::endl;
        return -1;
    }
    const ScopeExit cleanWindow([&]() { glfwDestroyWindow(window); });

    const vk::Result result = (vk::Result)glfwCreateWindowSurface(icontext.instance, window, nullptr, (VkSurfaceKHR*)&icontext.surface);
    if (result != vk::Result::eSuccess) {
        std::cerr << "GLFW Surface creation failed! With VkResult " << vk::to_string(result) << std::endl;
        return -1;
    }
    const ScopeExit cleanSurface([&]() { icontext.instance.destroySurfaceKHR(icontext.surface); });

    recreateSwapchain(icontext);
    const ScopeExit cleanSwapchain([&]() { icontext.device.destroySwapchainKHR(icontext.swapchain); });

    createPrimaryCommandBufferContext(icontext);
    const ScopeExit cleanCommandPools([&]() { destroyPrimaryCommandBufferContext(icontext); });

    const auto primaryQueue = icontext.device.getQueue(icontext.primaryFamilyIndex, 0);

    const auto waitSemaphore = icontext.device.createSemaphore({});

    while (!glfwWindowShouldClose(window))
    {
        const auto nextImage = icontext.device.acquireNextImageKHR(icontext.swapchain, std::numeric_limits<uint64_t>().max(), waitSemaphore);
        checkErrorOrRecreate(nextImage.result, icontext);

        glfwPollEvents();

        rerecordPrimary(icontext, nextImage.value);

        const vk::PresentInfoKHR presentInfo(waitSemaphore, icontext.swapchain, nextImage.value);
        checkErrorOrRecreate(primaryQueue.presentKHR(presentInfo), icontext);
    }
    return 0;
}
