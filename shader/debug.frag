#version 460

#extension GL_EXT_mesh_shader : require

layout(location=0) in float depthsMin;
layout(location=1) in float depthsMax;

layout(location=2) perprimitiveEXT in vec3 lambdas;

layout(location=0) out vec4 colorOut;

layout (binding=0) uniform Camera {
    mat4 model;
    mat4 view;
    mat4 proj;
    float depth;
} camera;

void main() {
    float depth = (depthsMax - depthsMin);
    if(depth < 0.0f) {
        colorOut = vec4(1.0f, 0.0f, 0.0f, abs(depth));
        return;
    }
    colorOut = vec4(vec3(1.0f, 1.0f, 1.0f) * depth * camera.depth, 1.0f);
}