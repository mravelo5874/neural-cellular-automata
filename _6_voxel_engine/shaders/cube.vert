#version 330 core

layout (location=0) in vec3 in_pos;

uniform mat4 u_proj;
uniform mat4 u_view;
uniform mat4 u_model;
uniform vec3 u_eye;

out vec3 v_eye;
out vec3 v_ray;

void main() {
    gl_Position = u_model * u_proj * u_view * vec4(in_pos, 1.0);
    v_eye = u_eye;
    v_ray = in_pos - u_eye;
}