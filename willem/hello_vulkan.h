/* Copyright (c) 2014-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once

#define NVVK_ALLOC_DEDICATED
#include "nvvk/allocator_vk.hpp"
#include "nvvk/appbase_vkpp.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"

#include <vulkan/vulkan_core.h>
#include <nvvk/commands_vk.hpp>
#include <nvvk/images_vk.hpp>


//--------------------------------------------------------------------------------------------------
// Simple rasterizer of OBJ objects
// - Each OBJ loaded are stored in an `ObjModel` and referenced by a `ObjInstance`
// - It is possible to have many `ObjInstance` referencing the same `ObjModel`
// - Rendering is done in an offscreen framebuffer
// - The image of the framebuffer is displayed in post-process in a full-screen quad
//
class HelloVulkan : public nvvk::AppBase
{
public:
  void setup(const vk::Instance&       instance,
             const vk::Device&         device,
             const vk::PhysicalDevice& physicalDevice,
             uint32_t                  queueFamily) override;
  void createDescriptorSetLayout();
  void createGraphicsPipeline();
  void loadModel(const std::string& filename, nvmath::mat4f transform = nvmath::mat4f(1));
  void updateDescriptorSet();
  void createUniformBuffer();
  void createSceneDescriptionBuffer();
  void createTextureImages(const vk::CommandBuffer&        cmdBuf,
                           const std::vector<std::string>& textures);
  void updateUniformBuffer();
  void onResize(int /*w*/, int /*h*/) override;
  void destroyResources();
  void rasterize(const vk::CommandBuffer& cmdBuff);

  auto addShader(const std::string&      code,
                                            vk::ShaderStageFlagBits stage,
                                            const char*             entryPoint = "main");

  // The OBJ model
  struct ObjModel
  {
    uint32_t   nbIndices{0};
    uint32_t   nbVertices{0};
    nvvk::Buffer vertexBuffer;    // Device buffer of all 'Vertex'
    nvvk::Buffer indexBuffer;     // Device buffer of the indices forming triangles
    nvvk::Buffer matColorBuffer;  // Device buffer of array of 'Wavefront material'
    nvvk::Buffer matIndexBuffer;  // Device buffer of array of 'Wavefront material'
  };

  // Instance of the OBJ
  struct ObjInstance
  {
    uint32_t      objIndex{0};     // Reference to the `m_objModel`
    uint32_t      txtOffset{0};    // Offset in `m_textures`
    nvmath::mat4f transform{1};    // Position of the instance
    nvmath::mat4f transformIT{1};  // Inverse transpose
  };

  // Information pushed at each draw call
  struct ObjPushConstant
  {
    nvmath::vec3f lightPosition{10.f, 15.f, 8.f};
    int           instanceId{0};  // To retrieve the transformation matrix
    float         lightIntensity{100.f};
    int           lightType{0};  // 0: point, 1: infinite
  };
  ObjPushConstant m_pushConstant;

  struct Vertex
  {
    float position[3];
    float color[3];
  };
  // Array of objects and instances in the scene
  std::vector<ObjModel>    m_objModel;
  std::vector<ObjInstance> m_objInstance;

  // Graphic pipeline
  vk::PipelineLayout          m_pipelineLayout;
  vk::Pipeline                m_graphicsPipeline;
  nvvk::DescriptorSetBindings m_descSetLayoutBind;
  vk::DescriptorPool          m_descPool;
  vk::DescriptorSetLayout     m_descSetLayout;
  vk::DescriptorSet           m_descSet;

  nvvk::Buffer               m_cameraMat;  // Device-Host of the camera matrices
  nvvk::Buffer               m_sceneDesc;  // Device buffer of the OBJ instances
  std::vector<nvvk::Texture> m_textures;   // vector of all textures of the scene


  nvvk::AllocatorDedicated m_alloc;  // Allocator for buffer, images, acceleration structures
  nvvk::DebugUtil          m_debug;  // Utility to name objects




  // find memory type with desired properties.
  //this is the same as getMemoryType *shrug*
  uint32_t findMemoryType(uint32_t memoryTypeBits, VkMemoryPropertyFlags properties)
  {
    VkPhysicalDeviceMemoryProperties memoryProperties;

    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memoryProperties);

    /*
        How does this search work?
        See the documentation of VkPhysicalDeviceMemoryProperties for a detailed description. 
        */
    for(uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i)
    {
      if((memoryTypeBits & (1 << i))
         && ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties))
        return i;
    }
    throw "Can't find compatible mappable memory for image";
    return -1;
  }

  void saveRenderedImage();
};
