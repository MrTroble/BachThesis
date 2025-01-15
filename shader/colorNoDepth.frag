#version 460

#extension GL_EXT_mesh_shader : require

layout(location=0) in vec4 depthsMinMax;

layout(location=2) perprimitiveEXT in vec3 colorIn;

layout(location=0) out vec4 colorOut;

void main() {
    colorOut = vec4(colorIn, 1.0f);
}