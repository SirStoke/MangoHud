#version 450
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 vColor;

layout(set = 0, binding = 0) uniform UBO {
    mat4 mvp;
} ubo;

void main() {
    vColor = inColor;
    gl_Position = ubo.mvp * vec4(inPos, 1.0);
}
