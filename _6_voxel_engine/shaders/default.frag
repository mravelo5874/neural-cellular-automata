#version 330 core

layout (location=0) out vec4 fragColor;

uniform sampler3D u_vol;

void main() {
    float r = texture(u_vol, vec3(0.0)).r;
    fragColor = vec4(1.0, 0.0, 0.0, 1.0);
    
}

