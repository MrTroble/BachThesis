#version 460

#extension GL_EXT_mesh_shader : require

layout(location=0) in vec4 depthsMinMax;

layout(location=2) perprimitiveEXT in vec3 colorIn;

layout(location=0) out vec4 colorOut;

layout (binding=0) uniform Camera {
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 whole;
    mat4 inverseM;
    vec4 colorDepth;
} camera;

void main() {
    vec4 minValue = camera.inverseM * vec4(depthsMinMax.xy, depthsMinMax.z, 1.0f);    
    vec4 maxValue = camera.inverseM * vec4(depthsMinMax.xy, depthsMinMax.w, 1.0f);
    minValue /= minValue.w;
    maxValue /= maxValue.w;

    float depth = length(maxValue - minValue);
    colorOut = vec4(colorIn * depth * camera.colorDepth.w, 1.0f);
}