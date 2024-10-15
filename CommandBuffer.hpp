#pragma once

#include <filesystem>
#include <ranges>
#include "Context.hpp"
#include "backends/imgui_impl_vulkan.h"
#include "LoadVTK.hpp"
#include <glm/ext.hpp>

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

inline void recordMeshPipeline(const VTKFile& vtk, vk::CommandBuffer currentBuffer, IContext& context) {
    const size_t workGroups = vtk.amountOfTetrahedrons / MAX_WORK_GROUPS;
    const size_t lastGroupAmount = vtk.amountOfTetrahedrons - workGroups * MAX_WORK_GROUPS;
    for (size_t i = 0; i < workGroups; i++)
    {
        const uint32_t currentOffset = i * MAX_WORK_GROUPS;
        currentBuffer.pushConstants(context.defaultPipelineLayout, vk::ShaderStageFlagBits::eMeshEXT, 0, sizeof(uint32_t), &currentOffset);
        currentBuffer.drawMeshTasksEXT(MAX_WORK_GROUPS, 1, 1, context.dynamicLoader);
    }
    if (lastGroupAmount > 0) {
        const uint32_t currentOffset = workGroups * MAX_WORK_GROUPS;
        currentBuffer.pushConstants(context.defaultPipelineLayout, vk::ShaderStageFlagBits::eMeshEXT, 0, sizeof(uint32_t), &currentOffset);
        currentBuffer.drawMeshTasksEXT(lastGroupAmount, 1, 1, context.dynamicLoader);
    }
}

inline void recordVertexPipeline(const VTKFile& vtk, vk::CommandBuffer currentBuffer, IContext& context) {
    const uint32_t currentOffset = 0;
    currentBuffer.pushConstants(context.defaultPipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, sizeof(uint32_t), &currentOffset);
    currentBuffer.draw(vtk.amountOfTetrahedrons * 12, 1, 0, 0);
}

inline void rerecordPrimary(IContext& context, uint32_t currentImage, const std::vector<VTKFile>& vtkFiles) {
    auto& currentBuffer = context.commandBuffer.primaryBuffers[currentImage];
    const vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    currentBuffer.begin(beginInfo);
    const vk::ClearValue clearColor(vk::ClearColorValue{ 0.0f, 1.0f, 0.0f, 1.0f });
    const vk::RenderPassBeginInfo renderPassBegin(context.renderPass, context.frameBuffer[currentImage],
        { {0,0}, context.currentExtent }, clearColor);
    currentBuffer.beginRenderPass(renderPassBegin, vk::SubpassContents::eInline);

    for (const auto& vtk : vtkFiles)
    {
        currentBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, context.wireframePipeline);
        currentBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, context.defaultPipelineLayout, 0, vtk.descriptor, {});
        if (context.meshShader) {
            recordMeshPipeline(vtk, currentBuffer, context);
        }
        else {
            recordVertexPipeline(vtk, currentBuffer, context);
        }
    }

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
    std::vector shaderNames = { "test.frag.spv", "vertexWire.vert.spv" };
    const std::array meshShader = { "testMesh.spv" };
    if (context.meshShader) {
        std::ranges::copy(meshShader, std::back_inserter(shaderNames));
    }
    for (const auto& name : shaderNames) {
        const auto fileName = (std::filesystem::path("shader") / name).string();
        const auto loadValues = readFullFile(fileName);
        vk::ShaderModuleCreateInfo shaderModuleCreateInfo({}, loadValues.size(), (uint32_t*)loadValues.data());
        const auto shaderModule = context.device.createShaderModule(shaderModuleCreateInfo);
        context.shaderModule[name] = shaderModule;
    }
}

inline void createShaderPipelines(IContext& context) {
    loadAndAdd(context);

    std::array pipelineShaderStages = {
        vk::PipelineShaderStageCreateInfo{{}, vk::ShaderStageFlagBits::eFragment, context.shaderModule["test.frag.spv"], "main"},
        vk::PipelineShaderStageCreateInfo{{}, vk::ShaderStageFlagBits::eMeshEXT, context.shaderModule["testMesh.spv"], "main"}
    };

    vk::Rect2D rect2d{ {0,0}, context.currentExtent };
    vk::Viewport viewport(0, 0, (float)context.currentExtent.width, (float)context.currentExtent.height, 0.0f, 1.0f);
    vk::PipelineViewportStateCreateInfo viewportState({}, viewport, rect2d);
    vk::PipelineRasterizationStateCreateInfo rasterizationState({}, false, false, vk::PolygonMode::eLine,
        vk::CullModeFlagBits::eNone, vk::FrontFace::eClockwise, false, 0.0f, 0.0f, 0.0f, 1.0f);
    vk::PipelineMultisampleStateCreateInfo multiplesampleState({}, vk::SampleCountFlagBits::e1);
    vk::PipelineDepthStencilStateCreateInfo depthState({}, false, false, vk::CompareOp::eNever, false, false);
    std::array colorBlends = { vk::PipelineColorBlendAttachmentState(true, vk::BlendFactor::eOne, vk::BlendFactor::eZero) };
    vk::PipelineColorBlendStateCreateInfo colorBlend({}, false, vk::LogicOp::eCopy, colorBlends);

    const vk::ShaderStageFlagBits flagBitsForBindings = context.meshShader ? vk::ShaderStageFlagBits::eMeshEXT : vk::ShaderStageFlagBits::eVertex;

    std::array bindings = {
        vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer,
                    1, flagBitsForBindings),
        vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer,
                    1, flagBitsForBindings),
        vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer,
                    1, flagBitsForBindings) };
    vk::DescriptorSetLayoutCreateInfo descriptorSetCreateInfo({}, bindings);
    context.defaultDescriptorSetLayout = context.device.createDescriptorSetLayout(descriptorSetCreateInfo);

    const auto shaderStageFlags = context.meshShader ? vk::ShaderStageFlagBits::eMeshEXT : vk::ShaderStageFlagBits::eVertex;
    std::array descriptorSets = { context.defaultDescriptorSetLayout };
    std::array pushConstantRanges = { vk::PushConstantRange{shaderStageFlags, 0, 2 * sizeof(uint32_t)} };
    vk::PipelineLayoutCreateInfo pipelineLayoutCreate({}, descriptorSets, pushConstantRanges);
    const auto pipelineLayout = context.device.createPipelineLayout(pipelineLayoutCreate);
    context.defaultPipelineLayout = pipelineLayout;

    vk::GraphicsPipelineCreateInfo createWirelessCreateInfo({}, pipelineShaderStages);
    createWirelessCreateInfo.layout = pipelineLayout;
    createWirelessCreateInfo.pMultisampleState = &multiplesampleState;
    createWirelessCreateInfo.pDepthStencilState = &depthState;
    createWirelessCreateInfo.pColorBlendState = &colorBlend;
    createWirelessCreateInfo.pRasterizationState = &rasterizationState;
    createWirelessCreateInfo.pViewportState = &viewportState;
    createWirelessCreateInfo.renderPass = context.renderPass;
    if (context.meshShader) {
        const auto result = context.device.createGraphicsPipeline({}, createWirelessCreateInfo);
        context.wireframePipeline = result.value;
    }
    else {
        std::array noneMeshShaderStages = {
            pipelineShaderStages[0],
            vk::PipelineShaderStageCreateInfo{{}, vk::ShaderStageFlagBits::eVertex, context.shaderModule.at("vertexWire.vert.spv"), "main"}
        };
        createWirelessCreateInfo.setStages(noneMeshShaderStages);
        vk::PipelineVertexInputStateCreateInfo vertexInputState({}, {});
        createWirelessCreateInfo.setPVertexInputState(&vertexInputState);
        vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState({}, vk::PrimitiveTopology::eLineList);
        createWirelessCreateInfo.setPInputAssemblyState(&inputAssemblyState);
        const auto result = context.device.createGraphicsPipeline({}, createWirelessCreateInfo);
        context.wireframePipeline = result.value;
    }

    const vk::DescriptorPoolSize poolSize(vk::DescriptorType::eStorageBuffer, 100);
    const vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo({}, 1, poolSize);
    context.descriptorPool = context.device.createDescriptorPool(descriptorPoolCreateInfo);
}

inline void destroyShaderPipelines(IContext& context) {
    for (const auto& [name, shader] : context.shaderModule) {
        context.device.destroy(shader);
    }
    context.device.destroy(context.descriptorPool);
    context.device.destroy(context.defaultDescriptorSetLayout);
    context.device.destroy(context.defaultPipelineLayout);
    context.device.destroy(context.wireframePipeline);
}

inline void updateCamera(IContext& context) {
    glm::mat4* cameraMap = (glm::mat4*)context.device.mapMemory(context.cameraStagingMemory, 0, VK_WHOLE_SIZE);
    const float aspect = context.currentExtent.width / (float)context.currentExtent.height;
    auto projectionMatrix = glm::perspective(context.FOV, aspect, context.planes.x, context.planes.y);
    projectionMatrix[1][1] *= -1;
    *cameraMap = projectionMatrix * glm::lookAt(context.position, context.lookAtPosition, glm::vec3{ 0.0f, 1.0f, 0.0f }) 
                 * glm::scale(glm::identity<glm::mat4>(), glm::vec3(0.1f, 0.1f, 0.1f));
    context.device.unmapMemory(context.cameraStagingMemory);

    const auto [buffer, fence] = context.commandBuffer.get<DataCommandBuffer::DataUpload>();
    vk::CommandBufferBeginInfo beginInfo;
    buffer.begin(beginInfo);
    vk::BufferCopy bufferCopy(0, 0, sizeof(glm::mat4));
    buffer.copyBuffer(context.stagingCamera, context.uniformCamera, bufferCopy);
    buffer.end();

    std::array buffers = { buffer };
    vk::SubmitInfo submitInfo({}, {}, buffers, {});
    context.primaryQueue.submit(submitInfo, fence);
    const auto result = context.device.waitForFences(fence, true, std::numeric_limits<uint64_t>().max());
    if (result != vk::Result::eSuccess)
        throw std::runtime_error("Wait for fence failed!");
    context.device.resetFences(fence);
}

inline void createBuffer(IContext& context) {
    std::array queueFamily = { context.primaryFamilyIndex };
    vk::BufferCreateInfo bufferCreateInfo({}, sizeof(glm::mat4), vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive, queueFamily);
    context.stagingCamera = context.device.createBuffer(bufferCreateInfo);
    bufferCreateInfo.usage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eUniformBuffer;
    context.uniformCamera = context.device.createBuffer(bufferCreateInfo);
    const auto stagingSize = context.device.getBufferMemoryRequirements(context.stagingCamera).size;
    const auto uniformSize = context.device.getBufferMemoryRequirements(context.uniformCamera).size;

    context.cameraStagingMemory = context.requestMemory(stagingSize, vk::MemoryPropertyFlagBits::eHostVisible);
    context.cameraMemory = context.requestMemory(uniformSize, vk::MemoryPropertyFlagBits::eDeviceLocal);
    context.device.bindBufferMemory(context.stagingCamera, context.cameraStagingMemory, 0);
    context.device.bindBufferMemory(context.uniformCamera, context.cameraMemory, 0);

    updateCamera(context);
}

inline void destroyBuffer(IContext& context) {
    context.device.freeMemory(context.cameraMemory);
    context.device.freeMemory(context.cameraStagingMemory);
    context.device.destroy(context.uniformCamera);
    context.device.destroy(context.stagingCamera);
}
