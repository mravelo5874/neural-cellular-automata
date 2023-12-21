#version 330

layout (location=0) in vec2 in_pos;
layout (location=1) in vec2 in_uvs;

out vec2 v_uvs;

void main() {
    v_uvs = in_uvs;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}