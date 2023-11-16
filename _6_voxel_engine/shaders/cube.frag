#version 330 core

layout (location=0) out vec4 fragColor;

uniform sampler3D u_volume;

in vec3 v_eye;
in vec3 v_ray;

vec2 intersect_box(vec3 orig, vec3 dir) {
	const vec3 box_min = vec3(-1, -1, -1);
	const vec3 box_max = vec3(1, 1, 1);
	vec3 inv_dir = 1.0 / dir;
	vec3 tmin_tmp = (box_min - orig) * inv_dir;
	vec3 tmax_tmp = (box_max - orig) * inv_dir;
	vec3 tmin = min(tmin_tmp, tmax_tmp);
	vec3 tmax = max(tmin_tmp, tmax_tmp);
	float t0 = max(tmin.x, max(tmin.y, tmin.z));
	float t1 = min(tmax.x, min(tmax.y, tmax.z));
	return vec2(t0, t1);
}

void main() {   
    vec4 my_color = vec4(0.0);

    // step 1: normalize ray
    vec3 ray = normalize(v_ray);

    // fragColor = vec4(ray, 1.0);

    // step 2: intersect ray with volume, find interval along ray inside volume
    vec2 t_hit = intersect_box(v_eye, ray);
    if (t_hit.x > t_hit.y) {
        discard;
    }

    // avoid sampling behind eye
    t_hit.x = max(t_hit.x, 0.0);

    // step 3: set step size to march through volume
    float dt = 0.001;

    // step 4: march ray through volume and sample
    vec3 p = v_eye + t_hit.x * ray;
    for (float t = t_hit.x; t < t_hit.y; t += dt) {
        // step 5: sample volume
        vec3 pos = (p/2.0)+0.5;
        vec4 rgba = texture(u_volume, pos);

        // Step 6: Accumulate the color and opacity using 
		//  the front-to-back compositing equation
        rgba.a *= 0.2;
        if (rgba.a > 0.1) {
            my_color.rgb += (1.0 - my_color.a) * rgba.a * rgba.rgb;
            my_color.a += (1.0 - my_color.a) * rgba.a;
        }
        
        if (my_color.a >= 0.95) {
            break;
        }
        p += ray * dt;
    }

    // set color to white if no voxels hit
    if (my_color == vec4(0.0, 0.0, 0.0, 0.0)) {
        my_color = vec4(1.0);
    }

    fragColor = my_color;
}