#pragma once
#include <vulkan/vulkan.hpp>

enum class DataCommandBuffer {
    DataUpload,
    Last = DataUpload
};

struct CommandBufferContext {
    vk::CommandPool primaryPool;
    vk::CommandPool uploadAndDataPool;
    std::vector<vk::CommandBuffer> primaryBuffers;
    std::array<vk::CommandBuffer, (size_t)DataCommandBuffer::Last + 1> dataCommandBuffer;
};

struct IContext {
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
    std::vector<vk::Image> swapchainImages;
    // Command Buffer
    CommandBufferContext commandBuffer;
};
