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
    Wireframe, Proxy
};
constexpr size_t PIPELINE_TYPE_AMOUNT = (size_t)PipelineType::Proxy + 1;

namespace std {
    inline std::string to_string(PipelineType type) {
        switch (type)
        {
        case PipelineType::Wireframe:
            return "Wireframe";
        case PipelineType::Proxy:
            return "Proxy";
        default:
            throw std::runtime_error("Pipeline type not found");
        }
    }
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
    vk::PipelineLayout defaultPipelineLayout;
    vk::DescriptorPool descriptorPool;    
    vk::Pipeline wireframePipeline;
    vk::Pipeline proxyPipeline;
    // Memory
    vk::DeviceMemory cameraStagingMemory;
    vk::DeviceMemory cameraMemory;
    vk::Buffer stagingCamera;
    vk::Buffer uniformCamera;
    // Camera
    glm::vec2 planes{ 0.01f, 100.0f};
    float FOV = glm::radians(45.0f);
    glm::vec3 position{ 0.0f, 0.0f, 0.0f };
    glm::vec3 lookAtPosition{ 0.0f, 0.0f, 1.0f };
    // Settings
    PipelineType type = PipelineType::Wireframe;
    // Queue
    vk::Queue primaryQueue;

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
