#version 330 core

layout (location=0) out vec4 fragColor;

in vec3 v_color;

void main() {   
    fragColor = vec4(1.0, 1.0, 1.0, 0.0);
}