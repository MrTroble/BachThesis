#pragma once

#include <filesystem>
#include "Context.hpp"
#include "backends/imgui_impl_vulkan.h"

inline void createPrimaryCommandBufferContext(IContext& context) {
    const vk::CommandPoolCreateInfo defaultPoolCreateInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        context.primaryFamilyIndex);
    context.commandBuffer.primaryPool = context.device.createCommandPool(defaultPoolCreateInfo);
    context.commandBuffer.uploadAndDataPool = context.device.createCommandPool(defaultPoolCreateInfo);

    vk::CommandBufferAllocateInfo allocateInfo(context.commandBuffer.primaryPool, vk::CommandBufferLevel::ePrimary, context.amountOfImages);
    context.commandBuffer.primaryBuffers = context.device.allocateCommandBuffers(allocateInfo);
    allocateInfo.commandBufferCount = context.commandBuffer.dataCommandBuffer.size();
    const auto dataBuffers = context.device.allocateCommandBuffers(allocateInfo);
    std::copy(dataBuffers.begin(), dataBuffers.end(), context.commandBuffer.dataCommandBuffer.begin());

    for (auto& fence : context.commandBuffer.dataCommandFences)
    {
        fence = context.device.createFence({});
    }
}

inline void destroyPrimaryCommandBufferContext(IContext& context) {
    context.device.destroy(context.commandBuffer.primaryPool);
    context.device.destroy(context.commandBuffer.uploadAndDataPool);
    for (const auto fence : context.commandBuffer.dataCommandFences)
    {
        context.device.destroy(fence);
    }
}

inline void rerecordPrimary(IContext& context, uint32_t currentImage) {
    auto& currentBuffer = context.commandBuffer.primaryBuffers[currentImage];
    const vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    currentBuffer.begin(beginInfo);
    const vk::ClearValue clearColor(vk::ClearColorValue{ 0.0f, 0.0f, 0.0f, 0.0f });
    const vk::RenderPassBeginInfo renderPassBegin(context.renderPass, context.frameBuffer[currentImage],
        { {0,0}, context.currentExtent }, clearColor);
    currentBuffer.beginRenderPass(renderPassBegin, vk::SubpassContents::eInline);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), currentBuffer);
    currentBuffer.endRenderPass();
    currentBuffer.end();
}

inline void recreateSwapchain(IContext& icontext) {
    if (icontext.swapchain)
        icontext.device.destroySwapchainKHR(icontext.swapchain);
    const std::array queueFamiliesInSwapchain = { icontext.primaryFamilyIndex };
    const vk::SwapchainCreateInfoKHR swapchainCreateInfo({}, icontext.surface, icontext.amountOfImages, vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear, icontext.currentExtent,
        1, vk::ImageUsageFlagBits::eColorAttachment, vk::SharingMode::eExclusive, queueFamiliesInSwapchain);
    icontext.swapchain = icontext.device.createSwapchainKHR(swapchainCreateInfo);
    const auto swapchainImages = icontext.device.getSwapchainImagesKHR(icontext.swapchain);

    for (auto imageView : icontext.swapchainImages) {
        icontext.device.destroy(imageView);
    }
    icontext.swapchainImages.resize(icontext.amountOfImages);
    icontext.frameBuffer.resize(icontext.amountOfImages);
    size_t nextFrame = 0;
    for (auto& frame : icontext.frameBuffer)
    {
        if (frame)
            icontext.device.destroy(frame);
        const vk::ImageViewCreateInfo imageViewCreateInfo({}, swapchainImages[nextFrame],
            vk::ImageViewType::e2D, vk::Format::eB8G8R8A8Unorm, {}, { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
        const auto imageView = icontext.device.createImageView(imageViewCreateInfo);
        icontext.swapchainImages[nextFrame] = imageView;
        nextFrame++;
        const vk::FramebufferCreateInfo frameBufferCreateInfo({}, icontext.renderPass, imageView, icontext.currentExtent.width,
            icontext.currentExtent.height, 1);
        frame = icontext.device.createFramebuffer(frameBufferCreateInfo);
    }
}

inline void destroySwapchain(IContext& icontext) {
    icontext.device.destroySwapchainKHR(icontext.swapchain);
    for (auto imageView : icontext.swapchainImages) {
        icontext.device.destroy(imageView);
    }
    for (auto frame : icontext.frameBuffer) {
        icontext.device.destroy(frame);
    }

}

inline void renderPassCreation(IContext& icontext) {
    const std::array attachements = { vk::AttachmentDescription({}, vk::Format::eB8G8R8A8Unorm, vk::SampleCountFlagBits::e1,
        vk::AttachmentLoadOp::eClear,vk::AttachmentStoreOp::eStore,vk::AttachmentLoadOp::eClear,vk::AttachmentStoreOp::eStore,
        vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR) };

    const std::array value{ vk::AttachmentReference(0, vk::ImageLayout::eColorAttachmentOptimal) };
    const vk::SubpassDescription subpassDescription({}, vk::PipelineBindPoint::eGraphics, {}, value);

    const vk::RenderPassCreateInfo  renderPassCreateInfo({}, attachements, subpassDescription);
    icontext.renderPass = icontext.device.createRenderPass(renderPassCreateInfo);
}

inline void loadAndAdd(IContext& context) {
    for (const auto& name : { "test.frag.spv", "testMesh.spv" }) {
        const auto fileName = (std::filesystem::path("shader") / name).string();
        const auto loadValues = readFullFile(fileName);
        vk::ShaderModuleCreateInfo shaderModuleCreateInfo({}, loadValues.size(), (uint32_t*)loadValues.data());
        const auto shaderModule = context.device.createShaderModule(shaderModuleCreateInfo);
        context.shaderModule[name] = shaderModule;
    }
}

inline void createShaderPipelines(IContext& context) {
    loadAndAdd(context);
}

inline void destroyShaderPipelines(IContext& context) {
    for (const auto& [name, shader] : context.shaderModule) {
        context.device.destroy(shader);
    }
}
