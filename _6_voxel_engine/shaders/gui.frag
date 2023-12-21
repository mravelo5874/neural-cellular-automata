#version 330

out vec4 fragColor;
uniform sampler2D u_texture;
in vec2 v_uv;

void main() 
{
    fragColor = texture(u_texture, v_uv);
}