#pragma once

#include <vector>
#include <string>
#include "Context.hpp"

constexpr uint32_t MAX_WORK_GROUPS = 128;

struct Vertex {
    float x, y, z;
};

struct Tetrahedron {
    uint32_t p1, p2, p3, p4;
};
constexpr uint32_t BUFFER_SLAB_AMOUNT = MAX_WORK_GROUPS * sizeof(Tetrahedron);

struct WorkGroup {
    size_t offset;
    size_t groupSize = MAX_WORK_GROUPS;
};

struct VTKFile {
    std::vector<WorkGroup> workGroups;
};

VTKFile loadVTK(const std::string& vtkFile, IContext& context) {
    std::ifstream valueVTK(vtkFile);
    std::string value;

    std::vector<Vertex> vertices;
    std::vector<Tetrahedron> tetrahedrons;

    while (valueVTK)
    {
        valueVTK >> value;
        if (value == "v") {
            Vertex vertex;
            valueVTK >> vertex.x >> vertex.y >> vertex.z;
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
    const auto vertexByteSize = vertices.size() * sizeof(Vertex);
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
    std::copy(vertices.begin(), vertices.end(), mapped);
    std::copy(tetrahedrons.begin(), tetrahedrons.end(), (char*)mapped + vertexByteSize);
    context.device.unmapMemory(stagingMemory);
    context.device.bindBufferMemory(stagingBuffer, stagingMemory, 0);

    vk::BufferCreateInfo localBufferCreateInfo({},
        vertexByteSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
        vk::SharingMode::eExclusive, context.primaryFamilyIndex);
    const auto localVertexBuffer = context.device.createBuffer(localBufferCreateInfo);
    const auto localVertexMemory = context.device.getBufferMemoryRequirements(localVertexBuffer);
    localBufferCreateInfo.size = tetrahedronByteSize;
    const auto localIndexBuffer = context.device.createBuffer(localBufferCreateInfo);
    const auto sizeOfMemory = context.device.getBufferMemoryRequirements(localVertexBuffer).size +
        localVertexMemory.size + localVertexMemory.alignment;
    const auto actualeMemory = context.requestMemory(sizeOfMemory,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    context.device.bindBufferMemory(localVertexBuffer, actualeMemory, 0);
    const auto alignedSize = (localVertexMemory.size + localVertexMemory.alignment) % localVertexMemory.alignment;
    context.device.bindBufferMemory(localIndexBuffer, actualeMemory, alignedSize);

    auto commandBuffer = context.commandBuffer.dataCommandBuffer[(size_t)DataCommandBuffer::DataUpload];
    commandBuffer.begin({});

    vk::BufferCopy copyBuffer(0, 0, vertexByteSize);
    commandBuffer.copyBuffer(stagingBuffer, localVertexBuffer, copyBuffer);

    VTKFile file;
    size_t i = 0;
    for (; i < tetrahedrons.size(); i++)
    {
        if (i == MAX_WORK_GROUPS) {

        }
    }
    return file;
}
