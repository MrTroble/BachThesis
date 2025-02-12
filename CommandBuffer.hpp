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
    currentBuffer.drawMeshTasksEXT(vtk.amountOfTetrahedrons, 1, 1, context.dynamicLoader);
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

    if (context.settings.sortingOfPrimitives) {
        for (const auto& vtk : vtkFiles)
        {
            currentBuffer.executeCommands(vtk.sortSecondary);
        }
    }

    const size_t lodToUse = context.settings.useLOD ? ((size_t)context.settings.currentLOD + 1u) : 1u;
    if (context.settings.useLOD) {
        const size_t nextLOD = lodToUse + 1;
        for (const auto& vtk : vtkFiles)
        {
            const auto lodUpdateAmount = vtk.lodUpdateAmount[lodToUse];
            if (context.changedLOD && lodUpdateAmount != 0) {
                const std::array descriptorsToUse = { vtk.descriptor[0], vtk.descriptor[context.changedLOD == 1 ? lodToUse: nextLOD] };
                currentBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, context.defaultPipelineLayout, 0, descriptorsToUse, {});
                currentBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, context.computeLODUpdatePipeline);
                currentBuffer.dispatch(lodUpdateAmount, 1, 1);
            }
            const auto lodAmount = vtk.lodAmount[lodToUse];
            if (lodAmount == 0) continue;
            const std::array descriptorsToUse = { vtk.descriptor[0], vtk.descriptor[nextLOD] };
            currentBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, context.defaultPipelineLayout, 0, descriptorsToUse, {});
            currentBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, context.computeLODPipeline);
            currentBuffer.dispatch(lodAmount, 1, 1);
        }
    }


    const vk::ClearColorValue whiteValue{ 1.0f, 1.0f, 1.0f, 1.0f };
    const vk::ClearColorValue blackValue{ 0.0f, 0.0f, 0.0f, 1.0f };
    const vk::ClearValue clearColor(context.settings.type == PipelineType::ProxyABuffer ? blackValue : whiteValue);
    const vk::RenderPassBeginInfo renderPassBegin(context.renderPass, context.frameBuffer[currentImage],
        { {0,0}, context.currentExtent }, clearColor);
    currentBuffer.beginRenderPass(renderPassBegin, vk::SubpassContents::eInline);

    const vk::Pipeline currentPipeline = getFromType(context.settings.type, context);
    currentBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, currentPipeline);

    for (const auto& vtk : vtkFiles)
    {
        const std::array descriptorsToUse = { vtk.descriptor[0], vtk.descriptor[lodToUse] };
        currentBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, context.defaultPipelineLayout, 0, descriptorsToUse, {});
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
    std::vector shaderNames = { "test.frag.spv", "vertexWire.vert.spv", "debug.frag.spv", "color.frag.spv", "iota.comp.spv", "sort.comp.spv",
                                "lod.comp.spv", "colorNoDepth.frag.spv", "updateLOD.comp.spv" };
    const std::array meshShader = { "testMesh.mesh.spv", "proxyGen.mesh.spv", "dispatch.task.spv" };
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

inline void recreatePipeline(IContext& context) {
    for (size_t i = 0; i < PIPELINE_TYPE_AMOUNT; i++)
    {
        const auto pipe = getFromType((PipelineType)i, context);
        if (pipe)
            context.device.destroy(pipe);
    }

    std::array pipelineShaderStages = {
    vk::PipelineShaderStageCreateInfo{{}, vk::ShaderStageFlagBits::eFragment, context.shaderModule["test.frag.spv"], "main"},
    vk::PipelineShaderStageCreateInfo{{}, vk::ShaderStageFlagBits::eMeshEXT, context.shaderModule["testMesh.mesh.spv"], "main"}
    };

    std::array proxyPipelineShaderStages = {
    vk::PipelineShaderStageCreateInfo{{}, vk::ShaderStageFlagBits::eFragment, context.shaderModule["debug.frag.spv"], "main"},
    vk::PipelineShaderStageCreateInfo{{}, vk::ShaderStageFlagBits::eMeshEXT, context.shaderModule["proxyGen.mesh.spv"], "main"},
    vk::PipelineShaderStageCreateInfo{{}, vk::ShaderStageFlagBits::eTaskEXT, context.shaderModule["dispatch.task.spv"], "main"}
    };

    std::array colorPipelineShaderStages = {
    vk::PipelineShaderStageCreateInfo{{}, vk::ShaderStageFlagBits::eFragment, context.shaderModule["color.frag.spv"], "main"},
    vk::PipelineShaderStageCreateInfo{{}, vk::ShaderStageFlagBits::eMeshEXT, context.shaderModule["proxyGen.mesh.spv"], "main"},
    vk::PipelineShaderStageCreateInfo{{}, vk::ShaderStageFlagBits::eTaskEXT, context.shaderModule["dispatch.task.spv"], "main"}
    };

    std::array colorNoDepthPipelineShaderStages = {
    vk::PipelineShaderStageCreateInfo{{}, vk::ShaderStageFlagBits::eFragment, context.shaderModule["colorNoDepth.frag.spv"], "main"},
    vk::PipelineShaderStageCreateInfo{{}, vk::ShaderStageFlagBits::eMeshEXT, context.shaderModule["proxyGen.mesh.spv"], "main"},
    vk::PipelineShaderStageCreateInfo{{}, vk::ShaderStageFlagBits::eTaskEXT, context.shaderModule["dispatch.task.spv"], "main"}
    };

    vk::Rect2D rect2d{ {0,0}, context.currentExtent };
    vk::Viewport viewport(0, 0, (float)context.currentExtent.width, (float)context.currentExtent.height, 0.0f, 1.0f);
    vk::PipelineViewportStateCreateInfo viewportState({}, viewport, rect2d);
    vk::PipelineRasterizationStateCreateInfo rasterizationState({}, false, false, vk::PolygonMode::eLine,
        vk::CullModeFlagBits::eNone, vk::FrontFace::eClockwise, false, 0.0f, 0.0f, 0.0f, 1.0f);
    vk::PipelineMultisampleStateCreateInfo multiplesampleState({}, vk::SampleCountFlagBits::e1);
    vk::PipelineDepthStencilStateCreateInfo depthState({}, true, false, vk::CompareOp::eAlways);
    std::array colorBlends = { vk::PipelineColorBlendAttachmentState(true, vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
                               vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd, vk::FlagTraits<vk::ColorComponentFlagBits>::allFlags) };
    vk::PipelineColorBlendStateCreateInfo colorBlend({}, false, vk::LogicOp::eCopy, colorBlends);

    vk::GraphicsPipelineCreateInfo createWirelessCreateInfo({}, pipelineShaderStages);
    createWirelessCreateInfo.layout = context.defaultPipelineLayout;
    createWirelessCreateInfo.pMultisampleState = &multiplesampleState;
    createWirelessCreateInfo.pDepthStencilState = &depthState;
    createWirelessCreateInfo.pColorBlendState = &colorBlend;
    createWirelessCreateInfo.pRasterizationState = &rasterizationState;
    createWirelessCreateInfo.pViewportState = &viewportState;
    createWirelessCreateInfo.renderPass = context.renderPass;
    if (context.meshShader) {
        const auto result = context.device.createGraphicsPipeline({}, createWirelessCreateInfo);
        if (result.result != vk::Result::eSuccess)
            throw std::runtime_error("Pipeline error!");
        context.wireframePipeline = result.value;
        createWirelessCreateInfo.setStages(proxyPipelineShaderStages);
        rasterizationState.polygonMode = vk::PolygonMode::eFill;
        colorBlends[0].dstColorBlendFactor = vk::BlendFactor::eOne;
        colorBlends[0].colorBlendOp = vk::BlendOp::eReverseSubtract;
        const auto result2 = context.device.createGraphicsPipeline({}, createWirelessCreateInfo);
        if (result2.result != vk::Result::eSuccess)
            throw std::runtime_error("Pipeline error!");
        context.proxyPipeline = result2.value;
        colorBlends[0].colorBlendOp = vk::BlendOp::eAdd;
        const auto result3 = context.device.createGraphicsPipeline({}, createWirelessCreateInfo);
        if (result3.result != vk::Result::eSuccess)
            throw std::runtime_error("Pipeline error!");
        context.proxyABuffer = result3.value;
        colorBlends[0].colorBlendOp = vk::BlendOp::eReverseSubtract;
        createWirelessCreateInfo.setStages(colorPipelineShaderStages);
        const auto result4 = context.device.createGraphicsPipeline({}, createWirelessCreateInfo);
        if (result4.result != vk::Result::eSuccess)
            throw std::runtime_error("Pipeline error!");
        context.colorPipeline = result4.value;
        colorBlends[0].colorBlendOp = vk::BlendOp::eAdd;
        colorBlends[0].srcColorBlendFactor = vk::BlendFactor::eOne;
        colorBlends[0].dstColorBlendFactor = vk::BlendFactor::eZero;
        createWirelessCreateInfo.setStages(colorNoDepthPipelineShaderStages);
        const auto result5 = context.device.createGraphicsPipeline({}, createWirelessCreateInfo);
        if (result5.result != vk::Result::eSuccess)
            throw std::runtime_error("Pipeline error!");
        context.colorNoDepth = result5.value;
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

}

inline void createShaderPipelines(IContext& context) {
    loadAndAdd(context);

    vk::ShaderStageFlags flagBitsForBindings = context.meshShader ? (vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eTaskEXT) : vk::ShaderStageFlagBits::eVertex;
    flagBitsForBindings |= vk::ShaderStageFlagBits::eFragment;
    flagBitsForBindings |= vk::ShaderStageFlagBits::eCompute;

    const std::array bindings = {
        vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer,
                    1, flagBitsForBindings),
        vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer,
                    1, flagBitsForBindings),
        vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer,
                    1, flagBitsForBindings),
        vk::DescriptorSetLayoutBinding(3, vk::DescriptorType::eStorageBuffer,
                    1, flagBitsForBindings) };
    const vk::DescriptorSetLayoutCreateInfo descriptorSetCreateInfo({}, bindings);
    context.defaultDescriptorSetLayout = context.device.createDescriptorSetLayout(descriptorSetCreateInfo);

    const std::array lodBindings = {
        vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageBuffer,
                    1, flagBitsForBindings),
        vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer,
                    1, flagBitsForBindings),
        vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer,
                    1, flagBitsForBindings) };
    const vk::DescriptorSetLayoutCreateInfo lodBindingsSetCreateInfo({}, lodBindings);
    context.lodDescriptorSetLayout = context.device.createDescriptorSetLayout(lodBindingsSetCreateInfo);

    std::array pushConsts{ vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t) * 2} };

    std::array descriptorSets = { context.defaultDescriptorSetLayout, context.lodDescriptorSetLayout };
    vk::PipelineLayoutCreateInfo pipelineLayoutCreate({}, descriptorSets, pushConsts);
    const auto pipelineLayout = context.device.createPipelineLayout(pipelineLayoutCreate);
    context.defaultPipelineLayout = pipelineLayout;

    recreatePipeline(context);

    const auto iotaPipelineShaderStages = vk::PipelineShaderStageCreateInfo{ {}, vk::ShaderStageFlagBits::eCompute, context.shaderModule["iota.comp.spv"], "main" };
    const auto sortPipelineShaderStages = vk::PipelineShaderStageCreateInfo{ {}, vk::ShaderStageFlagBits::eCompute, context.shaderModule["sort.comp.spv"], "main" };
    const auto lodPipelineShaderStages = vk::PipelineShaderStageCreateInfo{ {}, vk::ShaderStageFlagBits::eCompute, context.shaderModule["lod.comp.spv"], "main" };
    const auto lodUpdatePipelineShaderStages = vk::PipelineShaderStageCreateInfo{ {}, vk::ShaderStageFlagBits::eCompute, context.shaderModule["updateLOD.comp.spv"], "main" };

    vk::ComputePipelineCreateInfo computePipeCreateInfo({}, iotaPipelineShaderStages, pipelineLayout);
    const auto result5 = context.device.createComputePipeline({}, computePipeCreateInfo);
    if (result5.result != vk::Result::eSuccess)
        throw std::runtime_error("Pipeline error!");
    context.computeInitPipeline = result5.value;
    computePipeCreateInfo.setStage(sortPipelineShaderStages);
    const auto result6 = context.device.createComputePipeline({}, computePipeCreateInfo);
    if (result6.result != vk::Result::eSuccess)
        throw std::runtime_error("Pipeline error!");
    context.computeSortPipeline = result6.value;
    computePipeCreateInfo.setStage(lodPipelineShaderStages);
    const auto result7 = context.device.createComputePipeline({}, computePipeCreateInfo);
    if (result7.result != vk::Result::eSuccess)
        throw std::runtime_error("Pipeline error!");
    context.computeLODPipeline = result7.value;
    computePipeCreateInfo.setStage(lodUpdatePipelineShaderStages);
    const auto result8 = context.device.createComputePipeline({}, computePipeCreateInfo);
    if (result8.result != vk::Result::eSuccess)
        throw std::runtime_error("Pipeline error!");
    context.computeLODUpdatePipeline = result8.value;

    const vk::DescriptorPoolSize poolStorageSize(vk::DescriptorType::eStorageBuffer, 3000);
    const vk::DescriptorPoolSize poolUniformSize(vk::DescriptorType::eUniformBuffer, 1000);
    std::array poolSizes{ poolStorageSize, poolUniformSize };
    const vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo({}, 1000, poolSizes);
    context.descriptorPool = context.device.createDescriptorPool(descriptorPoolCreateInfo);
}

inline void destroyShaderPipelines(IContext& context) {
    for (const auto& [name, shader] : context.shaderModule) {
        context.device.destroy(shader);
    }
    context.device.destroy(context.descriptorPool);
    context.device.destroy(context.defaultDescriptorSetLayout);
    context.device.destroy(context.lodDescriptorSetLayout);
    context.device.destroy(context.defaultPipelineLayout);
    for (size_t i = 0; i < PIPELINE_TYPE_AMOUNT; i++)
    {
        const auto pipe = getFromType((PipelineType)i, context);
        if (pipe)
            context.device.destroy(pipe);
    }
    context.device.destroy(context.computeInitPipeline);
    context.device.destroy(context.computeSortPipeline);
    context.device.destroy(context.computeLODPipeline);
    context.device.destroy(context.computeLODUpdatePipeline);
}

struct CameraInfo {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 whole;
    glm::mat4 inverse;
    glm::vec4 colorADepth;
    float lod;
    uint32_t type;
};

inline void updateCamera(IContext& context) {
    CameraInfo* cameraMap = (CameraInfo*)context.device.mapMemory(context.cameraStagingMemory, 0, VK_WHOLE_SIZE);
    const float aspect = context.currentExtent.width / (float)context.currentExtent.height;
    auto projectionMatrix = glm::perspective(context.settings.FOV, aspect, context.settings.planes.x, context.settings.planes.y);
    projectionMatrix[1][1] *= -1;
    cameraMap->proj = projectionMatrix;
    glm::vec3 lookAt = glm::vec3(glm::rotate(glm::identity<glm::mat4>(), context.settings.rotationAndZoom.x, glm::vec3{ 0, 1, 0 }) * glm::vec4{ 1, 0, 0, 1 });
    lookAt = glm::rotate(glm::identity<glm::mat4>(), context.settings.rotationAndZoom.y, glm::vec3{ 0, 0, 1 }) * glm::vec4(lookAt, 1);
    lookAt = glm::normalize(lookAt) * context.settings.rotationAndZoom.z;
    cameraMap->view = glm::lookAt(context.settings.position + lookAt, context.settings.position, glm::vec3{ 0.0f, 1.0f, 0.0f });
    cameraMap->model = glm::identity<glm::mat4>();
    cameraMap->whole = projectionMatrix * cameraMap->view * cameraMap->model;
    cameraMap->inverse = glm::inverse(projectionMatrix * cameraMap->view);
    cameraMap->colorADepth = context.settings.colorADepth;
    const auto values = ((uint32_t)context.settings.currentLOD);
    cameraMap->lod = context.settings.currentLOD - values;
    cameraMap->type = (context.oldLOD < values ? 1 : 0);
    if (cameraMap->type == 0) {
        const auto oldValues = ((uint32_t)context.oldLOD);
        cameraMap->type = (context.settings.currentLOD < oldValues ? 2 : 0);
    }
    context.oldLOD = context.settings.currentLOD;
    context.changedLOD = cameraMap->type;
#ifndef NDEBUG
    if (context.changedLOD) {
        assert(cameraMap->type == 1 || cameraMap->type == 2);
        if (cameraMap->type == 1) std::cout << "Changed LOD Level up" << std::endl;
        else std::cout << "Changed LOD Level down" << std::endl;
    }
#endif // !NDEBUG

    context.device.unmapMemory(context.cameraStagingMemory);

    const auto [buffer, fence] = context.commandBuffer.get<DataCommandBuffer::DataUpload>();
    vk::CommandBufferBeginInfo beginInfo;
    buffer.begin(beginInfo);
    vk::BufferCopy bufferCopy(0, 0, sizeof(CameraInfo));
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
    vk::BufferCreateInfo bufferCreateInfo({}, sizeof(CameraInfo), vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive, queueFamily);
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
