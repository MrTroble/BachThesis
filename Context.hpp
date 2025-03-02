#pragma once
#include <unordered_map>
#include <vector>
#include <string>

#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

enum class DataCommandBuffer {
    DataUpload,
    Last = DataUpload
};

struct CommandBufferContext {
    vk::CommandPool primaryPool;
    vk::CommandPool uploadAndDataPool;
    std::vector<vk::CommandBuffer> primaryBuffers;
    std::array<vk::CommandBuffer, (size_t)DataCommandBuffer::Last + 1> dataCommandBuffer;
    std::array<vk::Fence, (size_t)DataCommandBuffer::Last + 1> dataCommandFences;

    template<DataCommandBuffer buffer>
    inline std::pair<vk::CommandBuffer, vk::Fence> get() {
        return std::make_pair(dataCommandBuffer[(size_t)buffer], dataCommandFences[(size_t)buffer]);
    }
};

enum class PipelineType {
    Wireframe, Proxy, ProxyABuffer, ColorNoDepth, Color
};
constexpr size_t PIPELINE_TYPE_AMOUNT = (size_t)PipelineType::Color + 1;

namespace std {
    inline std::string to_string(PipelineType type) {
        switch (type)
        {
        case PipelineType::Wireframe:
            return "Wireframe";
        case PipelineType::Proxy:
            return "Proxy";
        case PipelineType::ProxyABuffer:
            return "Proxy with ABuffer";
        case PipelineType::Color:
            return "Color";
        case PipelineType::ColorNoDepth:
            return "Color No Depth";
        default:
            throw std::runtime_error("Pipeline type not found");
        }
    }
}

struct ContextSetting {
    // General
    PipelineType type = PipelineType::Wireframe;
    bool sortingOfPrimitives = false;
    std::vector<char> activeModels = std::vector<char>(5u, false);
    // Camera
    glm::vec2 planes{ 0.01f, 100.0f };
    float FOV = glm::radians(45.0f);
    glm::vec3 position{ 0.0f, 0.0f, 0.0f };
    glm::vec3 rotationAndZoom{ 0.0f, 0.0f, 1.0f };
    glm::vec4 colorADepth{ 1.0f, 1.0f, 1.0f, 1.0f };
    // LOD
    float currentLOD = 0.0f;
    bool useLOD = false;
    bool animate = false;
    float speed = 0.1f;
};

enum class PresetType {
    Default, BunnyTest, SortingSmall
};
constexpr size_t PRESET_TYPE_AMOUNT = (size_t)PresetType::SortingSmall + 1;
inline std::string to_string(PresetType type) {
    switch (type)
    {
    case PresetType::Default:
        return "Default";
    case PresetType::BunnyTest:
        return "Bunny test";
    case PresetType::SortingSmall:
        return "Sorting";
    default:
        break;
    }
    throw std::runtime_error("Preset not found!");
}
inline ContextSetting getSettingFromType(PresetType type) {
    ContextSetting setting;
    switch (type)
    {
    case PresetType::BunnyTest:
        setting.colorADepth.w = 10.0f;
        setting.position = { -0.017f, 0.110, -0.001 };
        setting.rotationAndZoom = { 1.2f, 0.0f, 0.22f };
        setting.activeModels[3] = true;
        setting.type = PipelineType::Proxy;
        return setting;
    case PresetType::SortingSmall:
        setting.position = { 0.0f, 0.5f, 0.5f };
        setting.rotationAndZoom = { 0.0f, 0.0f, 3.0f };
        setting.activeModels[0] = true;
        setting.sortingOfPrimitives = true;
        setting.type = PipelineType::ColorNoDepth;
        return setting;
    default:
        setting.activeModels[0] = true;
        return setting;
    }
    return setting;
}

struct IContext {
    GLFWwindow* window;
    vk::DispatchLoaderDynamic dynamicLoader;
    vk::Instance instance;
    // Use mesh shader
    bool meshShader = false;
    // Device Creation
    vk::Device device;
    vk::PhysicalDevice physicalDevice;
    std::vector<vk::QueueFamilyProperties> queueFamilyProperties;
    uint32_t primaryFamilyIndex;
    // Surface
    vk::SurfaceKHR surface;
    // Swapchain
    uint32_t amountOfImages = 3;
    vk::SwapchainKHR swapchain;
    vk::Extent2D currentExtent;
    std::vector<vk::ImageView> swapchainImages;
    // Command Buffer
    CommandBufferContext commandBuffer;
    // Framebuffer/RenderPass
    vk::RenderPass renderPass;
    std::vector<vk::Framebuffer> frameBuffer;
    // Shader/Pipes
    std::unordered_map<std::string, vk::ShaderModule> shaderModule;
    vk::DescriptorSetLayout defaultDescriptorSetLayout;
    vk::DescriptorSetLayout lodDescriptorSetLayout;
    vk::PipelineLayout defaultPipelineLayout;
    vk::DescriptorPool descriptorPool;    
    vk::Pipeline wireframePipeline;
    vk::Pipeline proxyPipeline;
    vk::Pipeline proxyABuffer;
    vk::Pipeline colorPipeline;
    vk::Pipeline colorNoDepth;
    vk::Pipeline computeInitPipeline;
    vk::Pipeline computeSortPipeline;
    vk::Pipeline computeLODPipeline;
    vk::Pipeline computeLODUpdatePipeline;
    // Memory
    vk::DeviceMemory cameraStagingMemory;
    vk::DeviceMemory cameraMemory;
    vk::Buffer stagingCamera;
    vk::Buffer uniformCamera;
    // Queue
    vk::Queue primaryQueue;
    // Settings
    ContextSetting settings;
    PresetType presetType = PresetType::Default;
    uint32_t changedLOD = 0;
    float oldLOD = 0;

    inline vk::DeviceMemory requestMemory(vk::DeviceSize memorySize, vk::MemoryPropertyFlags flags) {
        const auto properties = physicalDevice.getMemoryProperties();
        uint32_t memoryTypeIndex = UINT32_MAX;
        for (uint32_t i = 0; i < properties.memoryTypeCount; i++)
        {
            const auto& type = properties.memoryTypes[i];
            if (type.propertyFlags & flags) {
                memoryTypeIndex = i;
                break;
            }
        }
        vk::MemoryAllocateInfo memoryAllocationInfo(memorySize, memoryTypeIndex);
        return device.allocateMemory(memoryAllocationInfo);
    }

};

inline vk::Pipeline getFromType(PipelineType type, const IContext& context) {
    switch (type)
    {
    case PipelineType::Wireframe:
        return context.wireframePipeline;
    case PipelineType::Proxy:
        return context.proxyPipeline;
    case PipelineType::ProxyABuffer:
        return context.proxyABuffer;
    case PipelineType::Color:
        return context.colorPipeline;
    case PipelineType::ColorNoDepth:
        return context.colorNoDepth;
    default:
        throw std::runtime_error("Pipeline type not found");
    }
}

