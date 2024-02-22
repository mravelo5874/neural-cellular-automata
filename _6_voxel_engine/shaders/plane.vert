#version 330 core

layout (location=0) in vec3 in_pos;
layout (location=1) in vec3 in_color;

uniform mat4 u_proj;
uniform mat4 u_view;
uniform mat4 u_model;

uniform vec3 u_plane_pos;
uniform vec3 u_plane_rot;

out vec3 v_color;

void main() {
    gl_Position = u_model * u_proj * u_view * vec4(in_pos, 1.0) + vec4(u_plane_pos, 0.0) + vec4(u_plane_rot, 0.0);
    v_color = in_color;
    
}