#version 330

in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;

void main()
{
    v_uv = in_uv;
    gl_Position = vec4(in_position, 0.0, 1.0);
}