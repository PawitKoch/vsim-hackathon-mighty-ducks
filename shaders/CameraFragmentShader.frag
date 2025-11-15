#version 130

out vec4 outputF;

uniform sampler3D gSampler;

in vec4 TexCoord;

void main()
{
	outputF = texture(gSampler, TexCoord.stp);
	//outputF = vec4(TexCoord.xyz, 1.0);
} 
