#version 330 core

uniform vec3 lightPos;
uniform vec3 lightPos2;
uniform vec3 eyeDir;
uniform vec3 eyePos;
uniform float specularScaleFactor;
uniform float lightIntensity;
uniform float lightIntensity2;

out vec4 outputF;
in vec3 vertexColor;
in vec3 vertexWorldPos;
in vec3 vertexNormal;

void main()
{
	vec3 norm = normalize(vertexNormal);
	vec3 lightDir = normalize(lightPos - vertexWorldPos);
	
	float diff = max(dot(norm, lightDir), 0.0) * lightIntensity;

	vec3 lightDir2 = normalize(lightPos2 - vertexWorldPos);
	float diff2 = max(dot(norm, lightDir2), 0.0) * lightIntensity2;

	float light = (diff + diff2);

	vec3 viewDirection = normalize(eyeDir - vec3(vertexWorldPos));

	float specularScale = 0.0;
	float specular = 0.0;
	if(dot(lightDir, norm) > 0.0)
	{
		specular = pow(max(0.0, dot(reflect(-lightDir, norm), -viewDirection)), 100);
		specularScale = specularScaleFactor;
	}

	if(dot(lightDir2, norm) > 0.0)
	{
		specular += pow(max(0.0, dot(reflect(-lightDir2, norm), -viewDirection)), 100);
		specularScale += specularScaleFactor;
	}

	if(specularScale > 0.0)
	{
		specular /= specularScale;
		light += specular;
	}

	outputF = vec4(min(light + 0.2, 1.0) * vertexColor, 1.0);
} 
