#version 330

layout (location=0) out vec4 fragColor;

uniform sampler2D u_texture;

in vec2 v_uvs;

void main() {
    vec4 color = texture(u_texture, v_uvs);
    if (color.a < 0.01) {
        discard;
    }
    fragColor = color;
}