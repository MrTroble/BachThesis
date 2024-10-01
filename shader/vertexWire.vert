#version 460

layout (binding=0) uniform Camera {
    mat4 mvp;
} camera;
layout (binding=1) buffer Index {
    readonly uvec4 data[];
} index;
layout (binding=2) buffer Vertex {
    readonly vec4 vertexData[];
} vertex;

layout(push_constant) uniform PushConsts {
    uint offset;
} push;

void main() {
    const uint tetraID = gl_VertexIndex / 12;
    const uvec4 tetrahedron = index.data[tetraID];
    const uint vertexID = gl_VertexIndex % 12;
    const uint nextID = vertexID - (vertexID / 2);
    gl_Position = vertex.vertexData[tetrahedron[nextID]] * camera.mvp;
}
