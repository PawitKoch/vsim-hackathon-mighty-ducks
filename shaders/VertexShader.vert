#version 130

uniform sampler3D transTex;
uniform int nbTexLayers;
uniform vec3 eyeDir;

out vec3 vertexWorldPos;
out vec3 vertexColor;
out vec3 vertexNormal;


void main()
{
	vec4 pos = gl_Vertex;
	


	int ti = int(pos.w);
	pos.w = 1.0;

	const int kDim = 512;

	int nbPerRow = kDim/16;

	int xCoord = ti&(nbPerRow-1);
	int yCoord = (ti/nbPerRow)&(kDim-1);
	int zCoord = ti/(nbPerRow*kDim);

	float xStep = 1.0/float(kDim/4);

	float x = (4*xCoord + 0.5)*xStep;
	float y = (yCoord+0.5)/kDim;
	float z = (zCoord+0.5)/nbTexLayers;

	vec4 col0 = texture(transTex, vec3(x, y, z));
	vec4 col1 = texture(transTex, vec3(x + xStep, y, z));
	vec4 col2 = texture(transTex, vec3(x + 2.0*xStep, y, z));
	vec4 col3 = texture(transTex, vec3(x + 3.0*xStep, y, z));

	vec3 color = vec3(col0.w, col1.w, col2.w);
	col0.w = 0.0; col1.w = 0.0; col2.w = 0.0; col3.w = 1.0;

	mat4 matrix = mat4(col0,col1,col2,col3);

	vertexNormal = mat3(matrix) * gl_Normal;

	vec4 worldPos = matrix * pos;

	vertexWorldPos = vec3(worldPos);

	vec4 eyeSpacePos = gl_ModelViewMatrix * worldPos;

	gl_Position = gl_ProjectionMatrix * eyeSpacePos;
	vertexColor = color;
}
