
#version 460

#extension GL_EXT_mesh_shader : require

layout(location=0) in vec4 depthsMinMax;

layout(location=1) perprimitiveEXT in flat uint workGroupID;

layout(location=0) out vec4 colorOut;

layout (binding=0) uniform Camera {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 colorDepth;
} camera;

void main() {
    colorOut = vec4(workGroupID, 0.0f, 0.0f, 1.0f);
}
