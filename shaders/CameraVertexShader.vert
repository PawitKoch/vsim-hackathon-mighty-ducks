#version 130

out vec4 TexCoord;

void main()
{
	vec4 pos = gl_ModelViewProjectionMatrix * gl_Vertex;
	gl_Position = pos;
	TexCoord = gl_TexCoord[0];
}
