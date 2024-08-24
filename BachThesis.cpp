#include "Util.hpp"
#include "Context.hpp"
#include "CommandBuffer.hpp"

#include <iostream>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

using namespace std;

inline void checkErrorOrRecreate(vk::Result result, IContext& context) {
    if (result == vk::Result::eSuboptimalKHR || result == vk::Result::eErrorOutOfDateKHR) {
        const auto capabilities = context.physicalDevice.getSurfaceCapabilitiesKHR(context.surface);
        context.currentExtent = capabilities.currentExtent;
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
    vk::PhysicalDeviceMeshShaderFeaturesEXT meshShaderFeatures;
    meshShaderFeatures.meshShader = true;
    meshShaderFeatures.taskShader = true;
    vk::PhysicalDeviceFeatures2 features;
    features.pNext = &meshShaderFeatures;
    const vk::DeviceCreateInfo deviceCreateInfo({}, queueCreateInfo, {}, extensions, {}, &features);
    icontext.device = icontext.physicalDevice.createDevice(deviceCreateInfo);
    const ScopeExit cleanDevice([&]() { icontext.device.destroy(); });

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    icontext.window = glfwCreateWindow(640u, 480u, applicationInfo.pApplicationName, NULL, NULL);
    if (!icontext.window)
    {
        std::cerr << "GLFW could not init!" << std::endl;
        return -1;
    }
    const ScopeExit cleanWindow([&]() { glfwDestroyWindow(icontext.window); });

    const vk::Result result = (vk::Result)glfwCreateWindowSurface(icontext.instance, icontext.window, nullptr, (VkSurfaceKHR*)&icontext.surface);
    if (result != vk::Result::eSuccess) {
        std::cerr << "GLFW Surface creation failed! With VkResult " << vk::to_string(result) << std::endl;
        return -1;
    }
    const ScopeExit cleanSurface([&]() { icontext.instance.destroySurfaceKHR(icontext.surface); });
    const auto capabilities = icontext.physicalDevice.getSurfaceCapabilitiesKHR(icontext.surface);
    icontext.currentExtent = capabilities.currentExtent;

    renderPassCreation(icontext);
    const ScopeExit cleanRenderPass([&]() { icontext.device.destroy(icontext.renderPass); });

    recreateSwapchain(icontext);
    const ScopeExit cleanSwapchain([&]() { destroySwapchain(icontext); });

    createPrimaryCommandBufferContext(icontext);
    const ScopeExit cleanCommandPools([&]() { destroyPrimaryCommandBufferContext(icontext); });

    createShaderPipelines(icontext);
    const ScopeExit cleanShaderPipes([&]() { destroyShaderPipelines(icontext); });

    const auto primaryQueue = icontext.device.getQueue(icontext.primaryFamilyIndex, 0);

    const auto waitSemaphore = icontext.device.createSemaphore({});
    const auto acquireSemaphore = icontext.device.createSemaphore({});
    const ScopeExit cleanupSemaphore([&]() {
        icontext.device.destroy(waitSemaphore);
        icontext.device.destroy(acquireSemaphore);
        });

    vk::DescriptorPoolSize pool_sizes[] = {
        { vk::DescriptorType::eSampler, 1000},
        { vk::DescriptorType::eCombinedImageSampler, 1000},
        { vk::DescriptorType::eSampledImage, 1000},
        { vk::DescriptorType::eStorageImage, 1000},
        { vk::DescriptorType::eUniformTexelBuffer, 1000},
        { vk::DescriptorType::eStorageTexelBuffer, 1000},
        { vk::DescriptorType::eUniformBuffer, 1000},
        { vk::DescriptorType::eStorageBuffer, 1000},
        { vk::DescriptorType::eUniformBufferDynamic, 1000},
        { vk::DescriptorType::eStorageBufferDynamic, 1000},
        { vk::DescriptorType::eInputAttachment, 1000} };
    vk::DescriptorPoolCreateInfo pool_info = {};
    pool_info.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
    pool_info.maxSets = 1000 * IM_ARRAYSIZE(pool_sizes);
    pool_info.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
    pool_info.pPoolSizes = pool_sizes;
    const auto imguiPool = icontext.device.createDescriptorPool(pool_info);
    const ScopeExit cleanupPool([&]() { icontext.device.destroy(imguiPool); });

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    ImGui_ImplGlfw_InitForVulkan(icontext.window, true);
    ImGui_ImplVulkan_InitInfo vulkanImguiInfo{};
    vulkanImguiInfo.Device = icontext.device;
    vulkanImguiInfo.ImageCount = icontext.amountOfImages;
    vulkanImguiInfo.MinImageCount = icontext.amountOfImages;
    vulkanImguiInfo.Instance = icontext.instance;
    vulkanImguiInfo.PhysicalDevice = icontext.physicalDevice;
    vulkanImguiInfo.Queue = primaryQueue;
    vulkanImguiInfo.QueueFamily = icontext.primaryFamilyIndex;
    vulkanImguiInfo.RenderPass = icontext.renderPass;
    vulkanImguiInfo.Subpass = 0;
    vulkanImguiInfo.DescriptorPool = imguiPool;
    vulkanImguiInfo.Allocator = nullptr;
    vulkanImguiInfo.MSAASamples = VkSampleCountFlagBits::VK_SAMPLE_COUNT_1_BIT;
    ImGui_ImplVulkan_Init(&vulkanImguiInfo);

    std::vector<vk::Fence> fencesToCheck(icontext.amountOfImages);
    for (auto& fence : fencesToCheck)
    {
        fence = icontext.device.createFence({ vk::FenceCreateFlagBits::eSignaled });
    }
    const ScopeExit cleanFences([&]() { for (auto fence : fencesToCheck) icontext.device.destroy(fence); });

    while (!glfwWindowShouldClose(icontext.window))
    {
        glfwPollEvents();
        int x, y;
        glfwGetWindowSize(icontext.window, (int*)&x, (int*)&y);
        if (icontext.currentExtent.width != x || icontext.currentExtent.height != y) {
            recreateSwapchain(icontext);
        }

        const auto nextImage = icontext.device.acquireNextImageKHR(icontext.swapchain, std::numeric_limits<uint64_t>().max(), acquireSemaphore);
        checkErrorOrRecreate(nextImage.result, icontext);

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::ShowDemoWindow();
        ImGui::Render();

        checkErrorOrRecreate(icontext.device.waitForFences(fencesToCheck[nextImage.value], true, std::numeric_limits<uint64_t>().max()), icontext);
        icontext.device.resetFences(fencesToCheck[nextImage.value]);

        rerecordPrimary(icontext, nextImage.value);
        const std::array pipelineFlagBits = { vk::PipelineStageFlagBits::eAllGraphics | vk::PipelineStageFlagBits::eMeshShaderEXT };
        const vk::SubmitInfo submitInfo(acquireSemaphore, pipelineFlagBits, icontext.commandBuffer.primaryBuffers[nextImage.value], waitSemaphore);
        primaryQueue.submit(submitInfo, fencesToCheck[nextImage.value]);

        const vk::PresentInfoKHR presentInfo(waitSemaphore, icontext.swapchain, nextImage.value);
        checkErrorOrRecreate((vk::Result)vkQueuePresentKHR((VkQueue)primaryQueue, (VkPresentInfoKHR*)&presentInfo), icontext);
    }
    checkErrorOrRecreate(icontext.device.waitForFences(fencesToCheck, true, std::numeric_limits<uint64_t>().max()), icontext);

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    return 0;
}
