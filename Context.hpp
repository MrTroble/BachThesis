#pragma once
#include <unordered_map>
#include <vector>
#include <string>

#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

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

struct AllocationInfo {

};

struct IContext {
    GLFWwindow* window;
    vk::Instance instance;
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
    // Memory

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
