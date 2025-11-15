#version 130

uniform sampler3D transTex;
uniform int nbTexLayers;
uniform vec3 eyeDir;

out vec3 vertexWorldPos;
out vec3 vertexColor;
out vec3 vertexNormal;


void main()
{
	vertexColor = vec3(gl_Color);

	vertexNormal = gl_Normal;
	vertexWorldPos = vec3(gl_Vertex);
	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;

}
