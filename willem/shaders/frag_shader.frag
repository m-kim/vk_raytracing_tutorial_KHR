#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout : enable

#include "wavefront.glsl"


layout(push_constant) uniform shaderInformation
{
  vec3  lightPosition;
  uint  instanceId;
  float lightIntensity;
  int   lightType;
}
pushC;

// clang-format off
// Incoming 
layout(location = 0) in vec3 fragColor;
// Outgoing
layout(location = 0) out vec4 outColor;
// Buffers

// clang-format on


void main()
{
  // Result
  outColor = vec4(0,0,0, 1.0);//vec4(lightIntensity * (diffuse + specular), 1);
}
