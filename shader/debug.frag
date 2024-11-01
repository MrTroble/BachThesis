#version 460

#extension GL_EXT_mesh_shader : require

layout(location=0) in float depthsMin;
layout(location=1) in float depthsMax;

layout(location=2) perprimitiveEXT in vec3 lambdas;

layout(location=0) out vec4 colorOut;

void main() {
    colorOut = vec4(vec3(1.0f, 1.0f, 1.0f) * (depthsMax - depthsMin), 1.0f);
}