#pragma once
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
};
