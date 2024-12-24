#pragma once

#include <vector>
#include <array>
#include <string>
#include <glm/glm.hpp>
#include <iostream>

#include "Context.hpp"

constexpr uint32_t MAX_WORK_GROUPS = 256;

using VertIndex = uint32_t;

struct Tetrahedron {
    VertIndex indices[4];
};
constexpr uint32_t BUFFER_SLAB_AMOUNT = MAX_WORK_GROUPS * sizeof(Tetrahedron);
using TetIndex = size_t;

enum class EdgeType : size_t {
    Point = 1, Edge = 2, Face = 3
};

using Connection = std::tuple<TetIndex, EdgeType>;

struct AABB {
    glm::vec3 min{ FLT_MAX };
    glm::vec3 max{ -FLT_MAX };
};

inline AABB extendAABB(const AABB& aabb1, const AABB& aabb2) {
    return { glm::min(aabb1.min, aabb2.min), glm::max(aabb1.max, aabb2.max) };
}

struct VTKFile {
    size_t amountOfTetrahedrons;
    vk::DeviceMemory memory;
    vk::Buffer vertexBuffer;
    vk::Buffer indexBuffer;
    vk::Buffer numberBuffer;
    vk::Buffer cacheBuffer;
    vk::CommandPool sortSecondaryPool;
    vk::CommandBuffer sortSecondary;
    vk::DescriptorSet descriptor;
    AABB aabb;

    void unload(IContext& context) {
        context.device.freeMemory(memory);
        context.device.destroy(vertexBuffer);
        context.device.destroy(indexBuffer);
        context.device.destroy(numberBuffer);
        context.device.destroy(cacheBuffer);
        context.device.destroy(sortSecondaryPool);
    }
};

uint32_t findPowerAbove(uint32_t n) {
    int k = 1;
    while (k > 0 && k < n)
        k <<= 1;
    return k;
}

// Source https://courses.cs.duke.edu//fall08/cps196.1/Pthreads/bitonic.c
void recordBitonicSort(uint32_t n, vk::CommandBuffer buffer, IContext& context, vk::Buffer sortBuffer) {
    const auto N = findPowerAbove(n);
    uint32_t j, k;
    const auto flags = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead;
    vk::BufferMemoryBarrier bufferMemoryBarrier(flags, flags, context.primaryFamilyIndex, context.primaryFamilyIndex, sortBuffer, 0u, VK_WHOLE_SIZE);
    buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eAllGraphics, vk::DependencyFlagBits::eDeviceGroup, {}, { bufferMemoryBarrier }, {});
    for (k = 2; k <= N; k = 2 * k) {
        for (j = k >> 1; j > 0; j = j >> 1) {
            std::array values = { k, j };
            buffer.pushConstants(context.defaultPipelineLayout, vk::ShaderStageFlagBits::eCompute, 0u, 2 * sizeof(uint32_t), values.data());
            buffer.dispatch(n, 1, 1);
            buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, vk::DependencyFlagBits::eDeviceGroup, {}, { bufferMemoryBarrier }, {});
        }
    }
    buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eAllGraphics, vk::DependencyFlagBits::eDeviceGroup, {}, { bufferMemoryBarrier }, {});
}


VTKFile loadVTK(const std::string& vtkFile, IContext& context) {
    std::ifstream valueVTK(vtkFile);
    if (!valueVTK) throw std::runtime_error("Could not find file!");
    std::string value;

    std::vector<glm::vec4> vertices;
    std::vector<Tetrahedron> tetrahedrons;

    AABB aabb;
    while (!valueVTK.eof() && valueVTK)
    {
        valueVTK >> value;
        if (value == "v") {
            glm::vec4 vertex;
            valueVTK >> vertex.x >> vertex.y >> vertex.z;
            vertex.w = 1.0f;
            vertices.push_back(vertex);
            aabb.max = glm::max(aabb.max, glm::vec3(vertex));
            aabb.min = glm::min(aabb.min, glm::vec3(vertex));
        }
        else if (value == "t") {
            Tetrahedron tetrahedron;
            valueVTK >> tetrahedron.indices[0] >> tetrahedron.indices[1] >>
                tetrahedron.indices[2] >> tetrahedron.indices[3];
            tetrahedrons.push_back(tetrahedron);
        }
        else {
            assert(false);
        }
    }

    std::vector<std::vector<TetIndex>> vertexConnection(vertices.size());
    for(auto& vec : vertexConnection) vec.reserve(64);
    for (TetIndex i = 0; i < tetrahedrons.size(); i++)
    {
        const auto& tetrahedron = tetrahedrons[i];
        for (size_t j = 0; j < 4; j++)
            vertexConnection[tetrahedron.indices[j]].push_back(i);
    }

    std::vector<std::vector<Connection>> tetrahedronGraph(tetrahedrons.size());
    for (auto& vec : tetrahedronGraph) vec.reserve(64);
    thread_local std::unordered_map<TetIndex, size_t> currentTetValues;
    size_t currentIndex = 0;
    for (const auto& tet : tetrahedrons)
    {
        currentTetValues.clear();
        for (size_t i = 0; i < 4; i++)
        {
            const auto value = tet.indices[i];
            const auto& connected = vertexConnection[value];
            for (const auto otherTetIndex : connected) {
                if(otherTetIndex == currentIndex) continue;
                currentTetValues[otherTetIndex]++;
            }
        }
        for (const auto& [index, amount] : currentTetValues)
        {
            assert(amount < 4);
            auto& edgeList = tetrahedronGraph[currentIndex];
            edgeList.emplace_back(index, (EdgeType)amount);
        }
        currentIndex++;
    }

    std::vector<char> allowedToTake(tetrahedrons.size(), true);
    // Finding outer tetrahedron and their direct neighbours
    // Do not use as prey -> 3.4 Boundary preservation tests
    for (size_t i = 0; i < tetrahedrons.size(); i++)
    {
        const auto& connected = tetrahedronGraph[i];
        size_t amount = 0;
        for (const auto& [other, type] : connected) {
            if(type == EdgeType::Face) amount++;
            if(amount == 3) break;
        }
        if (amount == 3) continue;
        allowedToTake[i] = false;
        for (const auto& [other, type] : connected) {
            allowedToTake[other] = false;
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
    const auto localIndexSize = context.device.getBufferMemoryRequirements(localIndexBuffer).size;
    localBufferCreateInfo.size = sizeof(uint32_t) * tetrahedrons.size();
    const auto localNumberBuffer = context.device.createBuffer(localBufferCreateInfo);
    const auto localNumberSize = context.device.getBufferMemoryRequirements(localIndexBuffer).size;
    localBufferCreateInfo.size = sizeof(glm::vec4) * tetrahedrons.size();
    const auto localCacheBuffer = context.device.createBuffer(localBufferCreateInfo);
    const auto localCacheSize = context.device.getBufferMemoryRequirements(localIndexBuffer).size;

    const auto sizeOfMemory = localIndexSize + localVertexMemory.size + localNumberSize + localCacheSize;
    const auto actualeMemory = context.requestMemory(sizeOfMemory,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    context.device.bindBufferMemory(localVertexBuffer, actualeMemory, 0);
    context.device.bindBufferMemory(localIndexBuffer, actualeMemory, localVertexMemory.size);
    context.device.bindBufferMemory(localNumberBuffer, actualeMemory, localVertexMemory.size + localIndexSize);
    context.device.bindBufferMemory(localCacheBuffer, actualeMemory,
        localVertexMemory.size + localIndexSize + localNumberSize);

    const vk::DescriptorSetAllocateInfo allocateInfo(context.descriptorPool, context.defaultDescriptorSetLayout);
    const auto descriptor = context.device.allocateDescriptorSets(allocateInfo);
    const vk::DescriptorBufferInfo descriptorCameraInfo(context.uniformCamera, 0, VK_WHOLE_SIZE);
    const vk::DescriptorBufferInfo descriptorIndexInfo(localIndexBuffer, 0, VK_WHOLE_SIZE);
    const vk::DescriptorBufferInfo descriptorVertexInfo(localVertexBuffer, 0, VK_WHOLE_SIZE);
    const vk::DescriptorBufferInfo descriptorNumberInfo(localNumberBuffer, 0, VK_WHOLE_SIZE);
    const vk::DescriptorBufferInfo descriptorCacheInfo(localCacheBuffer, 0, VK_WHOLE_SIZE);
    const vk::WriteDescriptorSet writeCameraSets(descriptor[0], 0, 0, vk::DescriptorType::eUniformBuffer, {}, descriptorCameraInfo);
    const vk::WriteDescriptorSet writeIndexDescriptorSets(descriptor[0], 1, 0, vk::DescriptorType::eStorageBuffer, {}, descriptorIndexInfo);
    const vk::WriteDescriptorSet writeVertexDescriptorSets(descriptor[0], 2, 0, vk::DescriptorType::eStorageBuffer, {}, descriptorVertexInfo);
    const vk::WriteDescriptorSet writeSortIndexDescriptorSets(descriptor[0], 3, 0, vk::DescriptorType::eStorageBuffer, {}, descriptorNumberInfo);
    const vk::WriteDescriptorSet writeCacheDescriptorSets(descriptor[0], 4, 0, vk::DescriptorType::eStorageBuffer, {}, descriptorCacheInfo);
    std::array writeUpdateInfos = { writeCameraSets, writeIndexDescriptorSets,  writeVertexDescriptorSets, writeSortIndexDescriptorSets, writeCacheDescriptorSets };
    context.device.updateDescriptorSets(writeUpdateInfos, {});

    auto [commandBuffer, fence] = context.commandBuffer.get<DataCommandBuffer::DataUpload>();
    vk::CommandBufferBeginInfo beginInfo;
    commandBuffer.begin(beginInfo);
    const vk::BufferCopy copyBuffer(0, 0, vertexByteSize);
    commandBuffer.copyBuffer(stagingBuffer, localVertexBuffer, copyBuffer);
    const vk::BufferCopy copyBufferIndex(vertexByteSize, 0, tetrahedronByteSize);
    commandBuffer.copyBuffer(stagingBuffer, localIndexBuffer, copyBufferIndex);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, context.defaultPipelineLayout, 0, descriptor, {});
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, context.computeInitPipeline);
    commandBuffer.dispatch(1, 1, 1);
    commandBuffer.end();
    auto queue = context.device.getQueue(context.primaryFamilyIndex, 0);
    const vk::SubmitInfo submitInfo({}, {}, commandBuffer);
    queue.submit(submitInfo, fence);

    const vk::CommandPoolCreateInfo commandPoolCreate({}, context.primaryFamilyIndex);
    const auto pool = context.device.createCommandPool(commandPoolCreate);

    const vk::CommandBufferAllocateInfo commandAllocateInfo(pool, vk::CommandBufferLevel::eSecondary, 1);
    const auto buffer = context.device.allocateCommandBuffers(commandAllocateInfo)[0];

    vk::CommandBufferInheritanceInfo inheritanceInfo(context.renderPass, 0);
    beginInfo.setPInheritanceInfo(&inheritanceInfo);
    buffer.begin(beginInfo);
    buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, context.defaultPipelineLayout, 0u, descriptor, {});
    buffer.bindPipeline(vk::PipelineBindPoint::eCompute, context.computeSortPipeline);
    recordBitonicSort(tetrahedrons.size(), buffer, context, localNumberBuffer);
    buffer.end();

    VTKFile file{ tetrahedrons.size(), actualeMemory, localVertexBuffer, localIndexBuffer, localNumberBuffer, localCacheBuffer, pool, buffer, descriptor[0], aabb };
    const auto result = context.device.waitForFences(fence, true, std::numeric_limits<uint64_t>().max());
    std::cout << "Loaded model: " << vtkFile << std::endl;
    if (result != vk::Result::eSuccess)
        throw std::runtime_error("Vulkan Error");
    context.device.resetFences(fence);
    return file;
}
