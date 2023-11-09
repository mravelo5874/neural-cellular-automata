#version 330 core

layout (location=0) in vec3 in_pos;
layout (location=1) in int voxel_id;
layout (location=2) in int face_id;

uniform mat4 m_proj;
uniform mat4 m_view;
uniform mat4 m_model;

out vec3 voxel_color;

float rand(vec2 co){
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    voxel_color = vec3(rand(vec2(voxel_id, voxel_id+1)), rand(vec2(voxel_id+1, voxel_id+2)), rand(vec2(voxel_id+2, voxel_id+3)));
    gl_Position = m_proj*m_view*m_model*vec4(in_pos, 1.0);
}