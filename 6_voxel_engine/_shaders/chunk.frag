#version 330 core

layout (location=0) out vec4 fragColor;

in vec3 voxel_color;

void main() {
    fragColor = vec4(voxel_color, 1);
}