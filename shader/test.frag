#version 460

#extension all : disable
#extension GL_EXT_fragment_shading_rate : disable

layout(location=0) out vec4 colorOut;

void main() {
    colorOut = vec4(1.0f, 0.0f, 0.0f, 1.0f);
}