#version 460

#extension GL_EXT_mesh_shader : require

layout(location=0) in vec4 depthsMinMax;

layout(location=2) perprimitiveEXT in vec3 lambdas;

layout(location=0) out vec4 colorOut;

layout (binding=0) uniform Camera {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 colorDepth;
} camera;

void main() {
    mat4 matInverse = inverse(camera.proj * camera.view);

    vec4 minValue = matInverse * vec4(depthsMinMax.xy, depthsMinMax.z, 1.0f);    
    vec4 maxValue = matInverse * vec4(depthsMinMax.xy, depthsMinMax.w, 1.0f);
    minValue /= minValue.w;
    maxValue /= maxValue.w;

    float depth = length(maxValue - minValue);
    colorOut = vec4(camera.colorDepth.xyz * depth * camera.colorDepth.w, 1.0f);
}