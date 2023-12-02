#version 330 core

layout (location=0) out vec4 fragColor;

in vec3 v_color;

void main() {   
    fragColor = vec4(v_color, 0.0);
}