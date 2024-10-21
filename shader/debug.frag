#version 460

#extension GL_EXT_mesh_shader : require

layout(location=0) perprimitiveEXT in vec3 lambdas;
layout(location=0) out vec4 colorOut;

void main() {
    colorOut = vec4(lambdas, 1.0f);
}