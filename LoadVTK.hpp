#pragma once

#include <vector>
#include <array>
#include <string>
#include <glm/glm.hpp>
#include <iostream>
#include <bitset>

#include "Context.hpp"

constexpr uint32_t MAX_WORK_GROUPS = 256;

using VertIndex = uint32_t;

struct Tetrahedron {
    VertIndex indices[4];
};
constexpr uint32_t BUFFER_SLAB_AMOUNT = MAX_WORK_GROUPS * sizeof(Tetrahedron);
using TetIndex = uint32_t;

enum class EdgeType : uint32_t {
    Point = 1, Edge = 2, Face = 3
};

using Connection = std::tuple<TetIndex, EdgeType>;
using TetGraph = std::vector<std::vector<Connection>>;

struct AABB {
    glm::vec3 min{ FLT_MAX };
    glm::vec3 max{ -FLT_MAX };
};

inline AABB extendAABB(const AABB& aabb1, const AABB& aabb2) {
    return { glm::min(aabb1.min, aabb2.min), glm::max(aabb1.max, aabb2.max) };
}

struct LODTetrahedron {
    glm::vec4 previous[4];
    glm::vec4 next;
    TetIndex tetrahedron;
};

struct LODLevel {
    std::vector<LODTetrahedron> lodTetrahedrons;
    std::vector<char> usageAfter;
};
constexpr size_t COLAPSING_PER_LEVEL = 10u;

enum class LodLevelFlag {
    None, L1, L2, L3
};
constexpr size_t LOD_COUNT = 4;
enum class Heuristic {
    Random
};
inline std::string stringLODLevel(const LodLevelFlag flag) {
    switch (flag)
    {
    case LodLevelFlag::None: return "None";
    case LodLevelFlag::L1: return "L1";
    case LodLevelFlag::L2: return "L2";
    case LodLevelFlag::L3: return "L3";
    default:
        throw std::runtime_error("Wrong LODLevelFlag!");
    }
}

using VTKBufferArray = std::array<vk::Buffer, 3 + LOD_COUNT * 2>;
using VTKSizeArray = std::array<vk::DeviceSize, 3 + LOD_COUNT * 2>;
using VTKDescriptorArray = std::vector<vk::DescriptorSet>;

struct VTKFile {
    size_t amountOfTetrahedrons;
    vk::DeviceMemory memory;
    VTKBufferArray bufferArray;
    vk::CommandPool sortSecondaryPool;
    vk::CommandBuffer sortSecondary;
    VTKDescriptorArray descriptor;
    AABB aabb;
    std::vector<size_t> lodAmount;

    void unload(IContext& context) {
        context.device.freeMemory(memory);
        for (const auto buffer : bufferArray)
            context.device.destroy(buffer);
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
    buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eAllCommands, vk::DependencyFlagBits::eDeviceGroup, {}, { bufferMemoryBarrier }, {});
    for (k = 2; k <= N; k = 2 * k) {
        for (j = k >> 1; j > 0; j = j >> 1) {
            std::array values = { k, j };
            buffer.pushConstants(context.defaultPipelineLayout, vk::ShaderStageFlagBits::eCompute, 0u, 2 * sizeof(uint32_t), values.data());
            buffer.dispatch(n, 1, 1);
            buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, vk::DependencyFlagBits::eDeviceGroup, {}, { bufferMemoryBarrier }, {});
        }
    }
    buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eAllCommands, vk::DependencyFlagBits::eDeviceGroup, {}, { bufferMemoryBarrier }, {});
}


struct LODGenerateInfo {
    const std::vector<char>& previous;
    const std::vector<char>& outer;
    const TetGraph& graph;
    IContext& context;
    LodLevelFlag level;
    Heuristic heuristic;
    std::string name;
};

inline LODLevel defaultLODLevel(IContext& context, const TetGraph& graph) {
    LODLevel level;
    level.usageAfter.resize(graph.size(), true);
    return level;
}

inline LODLevel loadLODLevel(const LODGenerateInfo& lodGenerateInfo, const std::vector<glm::vec4>& vertices,
    const std::vector<Tetrahedron>& tetrahedrons) {

    LODLevel level;
    level.usageAfter = lodGenerateInfo.previous;
    
    for (size_t i = 0; i < lodGenerateInfo.graph.size(); i++)
    {
        if(level.lodTetrahedrons.size() == COLAPSING_PER_LEVEL)
            break;
        if(!level.usageAfter[i] || lodGenerateInfo.outer[i])
            continue;
        const auto preyIndex = i;
        const auto& prey = tetrahedrons[preyIndex];
        auto& lodInfo = level.lodTetrahedrons.emplace_back();
        lodInfo.tetrahedron = preyIndex;
        glm::vec4 all(0);
        for (size_t i = 0; i < 4; i++)
        {
            const auto currentPoint = vertices[prey.indices[i]];
            lodInfo.previous[i] = currentPoint;
            all += currentPoint;
        }
        // Barycentric middle
        lodInfo.next = all / 4.0f;
        level.usageAfter[preyIndex] = 0;
        for (const auto [connecting, type] : lodGenerateInfo.graph[preyIndex])
        {   
            if(type != EdgeType::Point) continue;
            level.usageAfter[connecting] = 0;
        }
    }
    if (level.lodTetrahedrons.empty()) {
        std::cout << "Warning: No preys found for model " << lodGenerateInfo.name << " at level " << stringLODLevel(lodGenerateInfo.level) << std::endl;
    }
    if (level.lodTetrahedrons.size() < COLAPSING_PER_LEVEL) {
        std::cout << "Warning: Not enough preys found for model " << lodGenerateInfo.name << " at level " << stringLODLevel(lodGenerateInfo.level) << std::endl;
    }
    return level;
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
    for (auto& vec : vertexConnection) vec.reserve(64);
    for (TetIndex i = 0; i < tetrahedrons.size(); i++)
    {
        const auto& tetrahedron = tetrahedrons[i];
        for (size_t j = 0; j < 4; j++)
            vertexConnection[tetrahedron.indices[j]].push_back(i);
    }

    TetGraph tetrahedronGraph(tetrahedrons.size());
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
                if (otherTetIndex == currentIndex) continue;
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
            if (type == EdgeType::Face) amount++;
            if (amount == 3) break;
        }
        if (amount == 3) continue;
        allowedToTake[i] = false;
        for (const auto& [other, type] : connected) {
            allowedToTake[other] = false;
        }
    }

    std::array<LODLevel, LOD_COUNT> levelToGenerate;
    levelToGenerate[0] = defaultLODLevel(context, tetrahedronGraph);
    const size_t stateSize = (tetrahedrons.size() / sizeof(uint32_t) + 1) * sizeof(uint32_t);
    size_t additionalDataSize = LOD_COUNT * stateSize;
    for (size_t i = 1; i < LOD_COUNT; i++)
    {
        LODGenerateInfo generateInfo{
            levelToGenerate[i - 1].usageAfter, allowedToTake, tetrahedronGraph, context,
            (LodLevelFlag)i, Heuristic::Random, vtkFile };
        levelToGenerate[i] = loadLODLevel(generateInfo, vertices, tetrahedrons);
        additionalDataSize += levelToGenerate[i].lodTetrahedrons.size() * sizeof(LODTetrahedron);
    }

    const auto tetrahedronByteSize = tetrahedrons.size() * sizeof(Tetrahedron);
    const auto vertexByteSize = vertices.size() * sizeof(glm::vec4);
    const vk::BufferCreateInfo stagingBufferCreateInfo({},
        vertexByteSize + tetrahedronByteSize + additionalDataSize, vk::BufferUsageFlagBits::eTransferSrc,
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
    char* nextPointer = ((char*)mapped + vertexByteSize + tetrahedronByteSize);
    LODTetrahedron* nextPointerData = (LODTetrahedron*)(nextPointer + LOD_COUNT * stateSize);
    for (const auto& lod : levelToGenerate) {
        std::copy(lod.usageAfter.begin(), lod.usageAfter.end(), nextPointer);
        std::copy(lod.lodTetrahedrons.begin(), lod.lodTetrahedrons.end(), nextPointerData);
        nextPointer += stateSize;
        nextPointerData += lod.lodTetrahedrons.size();
    }
    context.device.unmapMemory(stagingMemory);
    context.device.bindBufferMemory(stagingBuffer, stagingMemory, 0);

    vk::BufferCreateInfo localBufferCreateInfo({},
        0, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer
        | vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer,
        vk::SharingMode::eExclusive, context.primaryFamilyIndex);
    VTKBufferArray localBuffers;
    VTKSizeArray sizesRequested = { vertexByteSize, tetrahedronByteSize,
                    sizeof(uint32_t) * tetrahedrons.size() };
    for (size_t i = 3; i < LOD_COUNT + 3; i++) {
        sizesRequested[i] = stateSize;
        sizesRequested[i + LOD_COUNT] = levelToGenerate[i - 3].lodTetrahedrons.size() * sizeof(LODTetrahedron);
    }

    VTKSizeArray sizesActual;
    size_t totalSizeRequested = 0;
    for (size_t i = 0; i < sizesRequested.size(); i++)
    {
        if (sizesRequested[i] == 0) continue;
        localBufferCreateInfo.size = sizesRequested[i];
        const auto localBuffer = context.device.createBuffer(localBufferCreateInfo);
        const auto requirements = context.device.getBufferMemoryRequirements(localBuffer);
        const auto localSize = (requirements.size / requirements.alignment + 1) * requirements.alignment;
        totalSizeRequested += localSize;
        sizesActual[i] = localSize;
        localBuffers[i] = localBuffer;
    }

    const auto actualeMemory = context.requestMemory(totalSizeRequested,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    size_t currentOffset = 0;
    for (size_t i = 0; i < localBuffers.size(); i++)
    {
        if (sizesRequested[i] == 0) continue;
        context.device.bindBufferMemory(localBuffers[i], actualeMemory, currentOffset);
        currentOffset += sizesActual[i];
    }

    std::array<vk::DescriptorSetLayout, 1 + LOD_COUNT> descriptorsToAllocate = { context.defaultDescriptorSetLayout };
    for (size_t i = 1; i < descriptorsToAllocate.size(); i++)
    {
        descriptorsToAllocate[i] = context.lodDescriptorSetLayout;
    }
    const vk::DescriptorSetAllocateInfo allocateInfo(context.descriptorPool, descriptorsToAllocate);
    const auto descriptor = context.device.allocateDescriptorSets(allocateInfo);
    const vk::DescriptorBufferInfo descriptorCameraInfo(context.uniformCamera, 0, VK_WHOLE_SIZE);
    const vk::DescriptorBufferInfo descriptorVertexInfo(localBuffers[0], 0, VK_WHOLE_SIZE);
    const vk::DescriptorBufferInfo descriptorIndexInfo(localBuffers[1], 0, VK_WHOLE_SIZE);
    const vk::DescriptorBufferInfo descriptorNumberInfo(localBuffers[2], 0, VK_WHOLE_SIZE);
    const vk::WriteDescriptorSet writeCameraSets(descriptor[0], 0, 0, vk::DescriptorType::eUniformBuffer, {}, descriptorCameraInfo);
    const vk::WriteDescriptorSet writeIndexDescriptorSets(descriptor[0], 1, 0, vk::DescriptorType::eStorageBuffer, {}, descriptorIndexInfo);
    const vk::WriteDescriptorSet writeVertexDescriptorSets(descriptor[0], 2, 0, vk::DescriptorType::eStorageBuffer, {}, descriptorVertexInfo);
    const vk::WriteDescriptorSet writeSortIndexDescriptorSets(descriptor[0], 3, 0, vk::DescriptorType::eStorageBuffer, {}, descriptorNumberInfo);
    // LOD Descriptor
    const vk::DescriptorBufferInfo descriptorLOD0(localBuffers[3], 0, VK_WHOLE_SIZE);
    const vk::WriteDescriptorSet writeLOD0DescriptorSets(descriptor[1], 1, 0, vk::DescriptorType::eStorageBuffer, {}, descriptorLOD0);
    std::array<vk::DescriptorBufferInfo, (LOD_COUNT - 1) * 2> lodBufferInfos;
    std::vector writeUpdateInfos = { writeCameraSets, writeIndexDescriptorSets,  writeVertexDescriptorSets, writeSortIndexDescriptorSets, writeLOD0DescriptorSets };
    for (size_t i = 0; i < LOD_COUNT - 1; i++)
    {
        const auto currentDescriptor = descriptor[2 + i];
        const auto visibilityBuffer = localBuffers[4 + i];
        if (visibilityBuffer) {
            auto& visibility = lodBufferInfos[i];
            visibility = vk::DescriptorBufferInfo{ visibilityBuffer, 0, VK_WHOLE_SIZE };
            const vk::WriteDescriptorSet writeVisibility(currentDescriptor, 1, 0, vk::DescriptorType::eStorageBuffer, {}, visibility);
            writeUpdateInfos.push_back(writeVisibility);
        }

        const auto dataBuffer = localBuffers[i + LOD_COUNT + 4];
        if (dataBuffer) {
            auto& tetrahedrons = lodBufferInfos[i * 2];
            tetrahedrons = vk::DescriptorBufferInfo{ dataBuffer , 0, VK_WHOLE_SIZE };
            const vk::WriteDescriptorSet writeData(currentDescriptor, 0, 0, vk::DescriptorType::eStorageBuffer, {}, tetrahedrons);
            writeUpdateInfos.push_back(writeData);
        }
    }
    context.device.updateDescriptorSets(writeUpdateInfos, {});

    const std::array descriptorsWithZeroLOD = { descriptor[0], descriptor[1] };

    auto [commandBuffer, fence] = context.commandBuffer.get<DataCommandBuffer::DataUpload>();
    vk::CommandBufferBeginInfo beginInfo;
    commandBuffer.begin(beginInfo);
    const vk::BufferCopy copyBuffer(0, 0, vertexByteSize);
    commandBuffer.copyBuffer(stagingBuffer, localBuffers[0], copyBuffer);
    const vk::BufferCopy copyBufferIndex(vertexByteSize, 0, tetrahedronByteSize);
    commandBuffer.copyBuffer(stagingBuffer, localBuffers[1], copyBufferIndex);

    vk::BufferCopy copyVisibility(tetrahedronByteSize + vertexByteSize, 0, stateSize);
    vk::BufferCopy copyLODData(tetrahedronByteSize + vertexByteSize + LOD_COUNT * stateSize);
    size_t visible = 0;
    for (const auto& lod : levelToGenerate) {
        commandBuffer.copyBuffer(stagingBuffer, localBuffers[3 + visible], copyVisibility);
        const auto sizeOfData = lod.lodTetrahedrons.size() * sizeof(LODTetrahedron);
        if (sizeOfData != 0) {
            copyLODData.size = sizeOfData;
            commandBuffer.copyBuffer(stagingBuffer, localBuffers[3 + LOD_COUNT + visible], copyLODData);
        }
        visible++;
        copyVisibility.srcOffset += stateSize;
        copyLODData.srcOffset += sizeOfData;
    }
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, context.defaultPipelineLayout, 0, descriptorsWithZeroLOD, {});
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
    buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, context.defaultPipelineLayout, 0u, descriptorsWithZeroLOD, {});
    buffer.bindPipeline(vk::PipelineBindPoint::eCompute, context.computeSortPipeline);
    recordBitonicSort(tetrahedrons.size(), buffer, context, localBuffers[2]);
    buffer.end();

    VTKFile file{ tetrahedrons.size(), actualeMemory, localBuffers, pool, buffer, descriptor, aabb };
    for (const auto& level : levelToGenerate)
    {
        file.lodAmount.push_back(level.lodTetrahedrons.size());
    }
    const auto result = context.device.waitForFences(fence, true, std::numeric_limits<uint64_t>().max());
    std::cout << "Loaded model: " << vtkFile << std::endl;
    if (result != vk::Result::eSuccess)
        throw std::runtime_error("Vulkan Error");
    context.device.resetFences(fence);
    return file;
}
