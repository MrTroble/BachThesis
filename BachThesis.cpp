#include "Util.hpp"
#include "Context.hpp"
#include "CommandBuffer.hpp"
#include "LoadVTK.hpp"

#include <iostream>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <chrono>
#include <deque>
#include <algorithm>
#include <numeric>

using namespace std;

inline bool checkErrorOrRecreate(vk::Result result, IContext& context) {
    if (result == vk::Result::eSuboptimalKHR || result == vk::Result::eErrorOutOfDateKHR) {
        const auto capabilities = context.physicalDevice.getSurfaceCapabilitiesKHR(context.surface);
        context.currentExtent = capabilities.currentExtent;
        recreateSwapchain(context);
        context.device.waitIdle();
        recreatePipeline(context);
        return true;
    }
    if (result != vk::Result::eSuccess) {
        std::cerr << "Vulkan Error with " << vk::to_string(result) << std::endl;
        throw std::runtime_error("Vulkan Error!");
    }
    return false;
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
    bool found = false;
    for (const auto physicalDevice : physicalDevices)
    {
        auto queueFamilies = physicalDevice.getQueueFamilyProperties();
        size_t familyIndex = 0;
        for (; familyIndex < queueFamilies.size(); familyIndex++) {
            if ((queueFamilies[familyIndex].queueFlags & (vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eTransfer)) &&
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
    if (!found) {
        std::cerr << "Device or queue not supported!" << std::endl;
        return -1;
    }

    const auto extensionsPresent = icontext.physicalDevice.enumerateDeviceExtensionProperties();
    for (auto value : extensionsPresent)
    {
        const std::string extName((char*)value.extensionName);
        if (extName == VK_EXT_MESH_SHADER_EXTENSION_NAME) {
            icontext.meshShader = true;
            break;
        }
    }
    const std::array queuePriorities{ 1.0f };
    const vk::DeviceQueueCreateInfo queueCreateInfo({}, icontext.primaryFamilyIndex, queuePriorities);

    std::vector extensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    vk::PhysicalDeviceFeatures2 features;
    vk::PhysicalDeviceMeshShaderFeaturesEXT meshShaderFeatures;
    if (icontext.meshShader) {
        extensions.push_back(VK_EXT_MESH_SHADER_EXTENSION_NAME);
        meshShaderFeatures.meshShader = true;
        meshShaderFeatures.taskShader = true;
        features.pNext = &meshShaderFeatures;
    }

    features.features.fillModeNonSolid = true;
    const vk::DeviceCreateInfo deviceCreateInfo({}, queueCreateInfo, {}, extensions, {}, &features);
    icontext.device = icontext.physicalDevice.createDevice(deviceCreateInfo);
    const ScopeExit cleanDevice([&]() { icontext.device.destroy(); });

    if (icontext.meshShader) {
        icontext.dynamicLoader.vkCmdDrawMeshTasksEXT =
            (PFN_vkCmdDrawMeshTasksEXT)vkGetDeviceProcAddr(icontext.device, "vkCmdDrawMeshTasksEXT");
    }

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

    icontext.primaryQueue = icontext.device.getQueue(icontext.primaryFamilyIndex, 0);

    createBuffer(icontext);
    const ScopeExit cleanBuffers([&]() { destroyBuffer(icontext); });

    const auto waitSemaphore = icontext.device.createSemaphore({});
    auto acquireSemaphore = icontext.device.createSemaphore({});
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
    vulkanImguiInfo.Queue = icontext.primaryQueue;
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
        fence = icontext.device.createFence({});
    }
    const ScopeExit cleanFences([&]() { for (auto fence : fencesToCheck) icontext.device.destroy(fence); });

    const auto startTimeLoading = std::chrono::steady_clock::now();
    std::vector vtkNames = { "perf.vtk", "crystal.vtk", "cube.vtk", "bunny.vtk", 
        //"Armadillo.vtk" 
    };
    std::vector<VTKFile> loadedVtkFiles = { };
    for (const auto& value : vtkNames) {
        loadedVtkFiles.push_back(loadVTK(std::string("assets/") + value, icontext));
    }
    std::vector<VTKFile> vtkFiles = { loadedVtkFiles[0] };
    std::vector<char>& active = icontext.settings.activeModels;
    active[0] = true;
    const ScopeExit cleanCrystal([&]() { for (auto& file : loadedVtkFiles) file.unload(icontext); });
    const auto endTimeLoading = std::chrono::steady_clock::now();
    const auto durationLoading = std::chrono::duration_cast<std::chrono::nanoseconds>(endTimeLoading - startTimeLoading);
    std::cout << "Loading time " << durationLoading.count() / (1e6f) << std::endl;

    int64_t currentValue = 0;
    std::deque<float> smoothing;
    constexpr size_t MAX_SMOOTH = 1000;

    const auto updateVTKs = [&]() {
        vtkFiles.clear();
        for (size_t i = 0; i < vtkNames.size(); i++) {
            if (active[i]) {
                vtkFiles.push_back(loadedVtkFiles[i]);
            }
        }
    };

    while (!glfwWindowShouldClose(icontext.window))
    {
        glfwPollEvents();
        if (glfwGetWindowAttrib(icontext.window, GLFW_ICONIFIED)) {
            continue;
        }
        int x, y;
        glfwGetWindowSize(icontext.window, (int*)&x, (int*)&y);
        if (icontext.currentExtent.width != x || icontext.currentExtent.height != y) {
            recreateSwapchain(icontext);
        }

        const auto nextImage = icontext.device.acquireNextImageKHR(icontext.swapchain, std::numeric_limits<uint64_t>().max(), acquireSemaphore);
        if (checkErrorOrRecreate(nextImage.result, icontext)) {
            icontext.device.destroy(acquireSemaphore);
            acquireSemaphore = icontext.device.createSemaphore({});
            continue;
        }

        updateCamera(icontext);

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        //ImGui::ShowDemoWindow();
        if (ImGui::Begin("Debug Menue")) {
            const auto currentPreset = to_string(icontext.presetType);
            if (ImGui::BeginCombo("Preset", currentPreset.c_str())) {
                const size_t currentSelected = (size_t)icontext.presetType;
                for (size_t i = 0; i < PRESET_TYPE_AMOUNT; i++)
                {
                    const auto currentType = (PresetType)i;
                    const auto name = to_string(currentType);
                    const bool isSelected = (currentSelected == i);
                    if (ImGui::Selectable(name.c_str(), isSelected) && currentType != PresetType::Default) {
                        icontext.settings = getSettingFromType(currentType);
                        icontext.presetType = currentType;
                        updateVTKs();
                    }
                    if (isSelected) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            const auto currentName = std::to_string(icontext.settings.type);
            if (ImGui::BeginCombo("Pipeline", currentName.c_str())) {
                const size_t currentSelected = (size_t)icontext.settings.type;
                for (size_t i = 0; i < PIPELINE_TYPE_AMOUNT; i++)
                {
                    const auto currentType = (PipelineType)i;
                    const auto name = std::to_string(currentType);
                    const bool isSelected = (currentSelected == i);
                    if (ImGui::Selectable(name.c_str(), isSelected))
                        icontext.settings.type = currentType;
                    if (isSelected) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            const auto sum = std::accumulate(smoothing.begin(), smoothing.end(), 0.0f);
            ImGui::Text("Frametime smoothed: %.3f ms", sum / (1e6f * smoothing.size()));
            ImGui::Text("Frametime: %.3f ms", currentValue / (1e6f));
            if (ImGui::CollapsingHeader("Models")) {
                for (size_t i = 0; i < vtkNames.size(); i++)
                {
                    if (ImGui::Checkbox(vtkNames[i], (bool*)&active[i])) {
                        updateVTKs();
                    }
                }
            }
            if (ImGui::CollapsingHeader("Camera")) {
                ImGui::SliderFloat2("Planes", &icontext.settings.planes.x, 0.001f, 1000.0f);
                ImGui::SliderFloat("FOV", &icontext.settings.FOV, 0.1f, 3.0f);
                ImGui::SliderFloat3("Position", &icontext.settings.position.x, 0, 10.0f);
                ImGui::SliderFloat2("Rotation", &icontext.settings.rotationAndZoom.x, -3.0f, 3.0f);
                ImGui::SliderFloat("Zoom", &icontext.settings.rotationAndZoom.z, 0, 10.0f);
                ImGui::SliderFloat3("Color Factor", &icontext.settings.colorADepth.x, 0, 1.0f);
                ImGui::SliderFloat("Depth Factor", &icontext.settings.colorADepth.w, 0, 10.0f);
                if (ImGui::Button("Centre")) {
                    AABB aabb{};
                    for (size_t i = 0; i < vtkNames.size(); i++) {
                        if (active[i]) {
                            aabb = extendAABB(aabb, loadedVtkFiles[i].aabb);
                        }
                    }
                    const auto middle = (aabb.max + aabb.min) * 0.5f;
                    icontext.settings.position = middle;
                }
            }
            if (ImGui::CollapsingHeader("LOD")) {
                ImGui::SliderFloat("Current LOD", &icontext.settings.currentLOD, 0.0f, -0.1f + LOD_COUNT - 1.0f);
                ImGui::Checkbox("Use LOD", &icontext.settings.useLOD);
            }
            ImGui::Checkbox("Sort primitives", &icontext.settings.sortingOfPrimitives);
        }
        ImGui::End();
        ImGui::Render();

        rerecordPrimary(icontext, nextImage.value, vtkFiles);
        const auto startTime = std::chrono::steady_clock::now();
        const auto shaderStage = icontext.meshShader ? vk::PipelineStageFlagBits::eMeshShaderEXT : vk::PipelineStageFlagBits::eTopOfPipe;
        const std::array pipelineFlagBits = { vk::PipelineStageFlagBits::eAllGraphics | shaderStage };
        const vk::SubmitInfo submitInfo(acquireSemaphore, pipelineFlagBits, icontext.commandBuffer.primaryBuffers[nextImage.value], waitSemaphore);
        icontext.primaryQueue.submit(submitInfo, fencesToCheck[nextImage.value]);

        const vk::PresentInfoKHR presentInfo(waitSemaphore, icontext.swapchain, nextImage.value);
        checkErrorOrRecreate((vk::Result)vkQueuePresentKHR((VkQueue)icontext.primaryQueue, (VkPresentInfoKHR*)&presentInfo), icontext);

        checkErrorOrRecreate(icontext.device.waitForFences(fencesToCheck[nextImage.value], true, std::numeric_limits<uint64_t>().max()), icontext);
        icontext.device.resetFences(fencesToCheck[nextImage.value]);

        const auto afterTime = std::chrono::steady_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(afterTime - startTime);
        currentValue = duration.count();
        smoothing.push_back(currentValue);
        if (smoothing.size() > MAX_SMOOTH)
            smoothing.pop_front();

    }
    icontext.device.waitIdle();

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    return 0;
}
