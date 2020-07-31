#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable

#include "wavefront.glsl"

// clang-format off
layout(binding = 2, set = 0, scalar) buffer ScnDesc { sceneDesc i[]; } scnDesc;
// clang-format on

layout(binding = 0) uniform UniformBufferObject
{
  mat4 view;
  mat4 proj;
  mat4 viewI;
}
ubo;

layout(push_constant) uniform shaderInformation
{
  vec3  lightPosition;
  uint  instanceId;
  float lightIntensity;
  int   lightType;
}
pushC;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 texCoord;


layout(location = 0) out vec3 fragColor;
out gl_PerVertex
{
  vec4 gl_Position;
};

void main()
{

  fragColor = inColor;
  gl_Position = ubo.proj * ubo.view * vec4(inPosition, 1.0);
}
