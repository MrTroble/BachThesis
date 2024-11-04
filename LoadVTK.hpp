#pragma once

#include <vector>
#include <string>
#include <glm/glm.hpp>
#include "Context.hpp"

constexpr uint32_t MAX_WORK_GROUPS = 128;

struct Tetrahedron {
    uint32_t p1, p2, p3, p4;
};
constexpr uint32_t BUFFER_SLAB_AMOUNT = MAX_WORK_GROUPS * sizeof(Tetrahedron);

struct VTKFile {
    size_t amountOfTetrahedrons;
    vk::DeviceMemory memory;
    vk::Buffer vertexBuffer;
    vk::Buffer indexBuffer;
    vk::DescriptorSet descriptor;

    void unload(IContext& context) {
        context.device.freeMemory(memory);
        context.device.destroy(vertexBuffer);
        context.device.destroy(indexBuffer);
    }
};

VTKFile loadVTK(const std::string& vtkFile, IContext& context) {
    std::ifstream valueVTK(vtkFile);
    if(!valueVTK) throw std::runtime_error("Could not find file!");
    std::string value;

    std::vector<glm::vec4> vertices;
    std::vector<Tetrahedron> tetrahedrons;

    while (valueVTK)
    {
        valueVTK >> value;
        if (value == "v") {
            glm::vec4 vertex;
            valueVTK >> vertex.x >> vertex.y >> vertex.z;
            vertex.w = 1.0f;
            vertices.push_back(vertex);
        }
        else if (value == "t") {
            Tetrahedron tetrahedron;
            valueVTK >> tetrahedron.p1 >> tetrahedron.p2 >>
                tetrahedron.p3 >> tetrahedron.p4;
            tetrahedrons.push_back(tetrahedron);
        }
        else {
            assert(false);
        }
    }
    const auto tetrahedronByteSize = tetrahedrons.size() * sizeof(Tetrahedron);
    const auto vertexByteSize = vertices.size() * sizeof(glm::vec4);
    const vk::BufferCreateInfo stagingBufferCreateInfo({},
        vertexByteSize + tetrahedronByteSize, vk::BufferUsageFlagBits::eTransferSrc,
        vk::SharingMode::eExclusive, context.primaryFamilyIndex);
    const auto stagingBuffer = context.device.createBuffer(stagingBufferCreateInfo);
    const ScopeExit cleanStagingBuffer([&]() { context.device.destroy(stagingBuffer); });

    const auto memoryRequirementsStaging = context.device.getBufferMemoryRequirements(stagingBuffer);
    const auto stagingMemory = context.requestMemory(memoryRequirementsStaging.size,
        vk::MemoryPropertyFlagBits::eHostVisible);
    const ScopeExit cleanStagingMemory([&]() { context.device.freeMemory(stagingMemory); });

    void* mapped = context.device.mapMemory(stagingMemory, 0, VK_WHOLE_SIZE);
    std::copy(vertices.begin(), vertices.end(), (glm::vec4*)mapped);
    std::copy(tetrahedrons.begin(), tetrahedrons.end(), (Tetrahedron*)((char*)mapped + vertexByteSize));
    context.device.unmapMemory(stagingMemory);
    context.device.bindBufferMemory(stagingBuffer, stagingMemory, 0);

    vk::BufferCreateInfo localBufferCreateInfo({},
        vertexByteSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer
        | vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer,
        vk::SharingMode::eExclusive, context.primaryFamilyIndex);
    const auto localVertexBuffer = context.device.createBuffer(localBufferCreateInfo);
    const auto localVertexMemory = context.device.getBufferMemoryRequirements(localVertexBuffer);
    localBufferCreateInfo.size = tetrahedronByteSize;
    const auto localIndexBuffer = context.device.createBuffer(localBufferCreateInfo);
    const auto sizeOfMemory = context.device.getBufferMemoryRequirements(localIndexBuffer).size +
        localVertexMemory.size + localVertexMemory.alignment;
    const auto actualeMemory = context.requestMemory(sizeOfMemory,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    context.device.bindBufferMemory(localVertexBuffer, actualeMemory, 0);
    const auto alignedSize = ((localVertexMemory.size + localVertexMemory.alignment) / localVertexMemory.alignment) * localVertexMemory.alignment;
    context.device.bindBufferMemory(localIndexBuffer, actualeMemory, alignedSize);

    auto [commandBuffer, fence] = context.commandBuffer.get<DataCommandBuffer::DataUpload>();
    const vk::CommandBufferBeginInfo beginInfo;
    commandBuffer.begin(beginInfo);
    const vk::BufferCopy copyBuffer(0, 0, vertexByteSize);
    commandBuffer.copyBuffer(stagingBuffer, localVertexBuffer, copyBuffer);
    const vk::BufferCopy copyBufferIndex(vertexByteSize, 0, tetrahedronByteSize);
    commandBuffer.copyBuffer(stagingBuffer, localIndexBuffer, copyBufferIndex);
    commandBuffer.end();
    auto queue = context.device.getQueue(context.primaryFamilyIndex, 0);
    const vk::SubmitInfo submitInfo({}, {}, commandBuffer);
    queue.submit(submitInfo, fence);

    const vk::DescriptorSetAllocateInfo allocateInfo(context.descriptorPool, context.defaultDescriptorSetLayout);
    const auto descriptor = context.device.allocateDescriptorSets(allocateInfo);
    vk::DescriptorBufferInfo descriptorCameraInfo(context.uniformCamera, 0, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo descriptorIndexInfo(localIndexBuffer, 0, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo descriptorVertexInfo(localVertexBuffer, 0, vertexByteSize);
    vk::WriteDescriptorSet writeCameraSets(descriptor[0], 0, 0, vk::DescriptorType::eUniformBuffer, {}, descriptorCameraInfo);
    vk::WriteDescriptorSet writeIndexDescriptorSets(descriptor[0], 1, 0, vk::DescriptorType::eStorageBuffer, {}, descriptorIndexInfo);
    vk::WriteDescriptorSet writeVertexDescriptorSets(descriptor[0], 2, 0, vk::DescriptorType::eStorageBuffer, {}, descriptorVertexInfo);
    std::array writeUpdateInfos = { writeCameraSets, writeIndexDescriptorSets,  writeVertexDescriptorSets };
    context.device.updateDescriptorSets(writeUpdateInfos, {});

    VTKFile file{ tetrahedrons.size(), actualeMemory, localVertexBuffer, localIndexBuffer, descriptor[0]};
    const auto result = context.device.waitForFences(fence, true, std::numeric_limits<uint64_t>().max());
    if (result != vk::Result::eSuccess)
        throw std::runtime_error("Vulkan Error");
    context.device.resetFences(fence);
    return file;
}
