#version 460

#extension GL_EXT_mesh_shader : require

layout(location=0) in float depthsMin;
layout(location=1) in float depthsMax;

layout(location=2) perprimitiveEXT in vec3 lambdas;

layout(location=0) out vec4 colorOut;

void main() {
    float depth = (depthsMax - depthsMin);
    if(depth < 0.0f) {
        colorOut = vec4(1.0f, 0.0f, 0.0f, abs(depth));
        return;
    }
    colorOut = vec4(vec3(1.0f, 1.0f, 1.0f) * depth * 500.0f, 1.0f);
}