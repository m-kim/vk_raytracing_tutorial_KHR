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

#include <sstream>
#include <vulkan/vulkan.hpp>



#define STB_IMAGE_IMPLEMENTATION
#include "fileformats/stb_image.h"
#include "obj_loader.h"

#include "hello_vulkan.h"
#include "nvh//cameramanipulator.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/pipeline_vk.hpp"

#include "nvh/fileoperations.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"

#define EMBED_SHADERS

namespace {
#ifdef EMBED_SHADERS
//cat vert_shader.vert.spv | xxd -i > vert_shader.vert.array
uint8_t s_computeShader[] = {

#include "shaders/shader.comp.array"
};
#else
uint8_t s_vertexShader[] = {0};
#endif
}

// Holding the camera matrices
struct CameraMatrices
{
  nvmath::mat4f view;
  nvmath::mat4f proj;
  nvmath::mat4f viewInverse;
  // #VKRay
  nvmath::mat4f projInverse;
};

//--------------------------------------------------------------------------------------------------
// Keep the handle on the device
// Initialize the tool to do all our allocations: buffers, images
//
void HelloVulkan::setup(const vk::Instance&       instance,
                        const vk::Device&         device,
                        const vk::PhysicalDevice& physicalDevice,
                        uint32_t                  queueFamily)
{
  AppBase::setup(instance, device, physicalDevice, queueFamily);
  m_alloc.init(device, physicalDevice);
  m_debug.setup(m_device);

  m_bufferSize = WIDTH * HEIGHT * 4 * 4;
}

//--------------------------------------------------------------------------------------------------
// Called at each frame to update the camera matrix
//
void HelloVulkan::updateUniformBuffer()
{
  const float aspectRatio = m_size.width / static_cast<float>(m_size.height);

  CameraMatrices ubo = {};
  ubo.view           = CameraManip.getMatrix();
  ubo.proj           = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, 0.1f, 1000.0f);
  // ubo.proj[1][1] *= -1;  // Inverting Y for Vulkan
  ubo.viewInverse = nvmath::invert(ubo.view);
  // #VKRay
  ubo.projInverse = nvmath::invert(ubo.proj);

  void* data = m_device.mapMemory(m_cameraMat.allocation, 0, sizeof(ubo));
  memcpy(data, &ubo, sizeof(ubo));
  m_device.unmapMemory(m_cameraMat.allocation);
}

//--------------------------------------------------------------------------------------------------
// Describing the layout pushed when rendering
//
void HelloVulkan::createDescriptorSetLayout()
{
  using vkDS     = vk::DescriptorSetLayoutBinding;
  using vkDT     = vk::DescriptorType;
  using vkSS     = vk::ShaderStageFlagBits;
  uint32_t nbTxt = static_cast<uint32_t>(m_textures.size());
  uint32_t nbObj = static_cast<uint32_t>(m_objModel.size());

  //Compute shader
  m_descSetLayoutBind.addBinding(vkDS(0, vkDT::eStorageBuffer, 1, vkSS::eCompute));
 
  // Camera matrices (binding = 1)
  m_descSetLayoutBind.addBinding(
      vkDS(1, vkDT::eUniformBuffer, 1, vkSS::eCompute));
  //// Materials (binding = 1)
  //m_descSetLayoutBind.addBinding(
  //    vkDS(1, vkDT::eStorageBuffer, nbObj, vkSS::eVertex | vkSS::eFragment | vkSS::eClosestHitKHR));
  //// Scene description (binding = 2)
  //m_descSetLayoutBind.addBinding(  //
  //    vkDS(2, vkDT::eStorageBuffer, 1, vkSS::eVertex | vkSS::eFragment | vkSS::eClosestHitKHR));
  //// Textures (binding = 3)
  //m_descSetLayoutBind.addBinding(
  //    vkDS(3, vkDT::eCombinedImageSampler, nbTxt, vkSS::eFragment | vkSS::eClosestHitKHR));
  //// Materials (binding = 4)
  //m_descSetLayoutBind.addBinding(
  //    vkDS(4, vkDT::eStorageBuffer, nbObj, vkSS::eFragment | vkSS::eClosestHitKHR));
  //// Storing vertices (binding = 5)
  //m_descSetLayoutBind.addBinding(  //
  //    vkDS(5, vkDT::eStorageBuffer, nbObj, vkSS::eClosestHitKHR));
  //// Storing indices (binding = 6)
  //m_descSetLayoutBind.addBinding(  //
  //    vkDS(6, vkDT::eStorageBuffer, nbObj, vkSS::eClosestHitKHR));
  // The top level acceleration structure
  //TODO
  //m_descSetLayoutBind.addBinding(  //
  //    vkDS(7, vkDT::eAccelerationStructureKHR, 1, vkSS::eFragment));

  m_descSetLayout = m_descSetLayoutBind.createLayout(m_device);
  m_descPool      = m_descSetLayoutBind.createPool(m_device, 1);
  m_descSet       = nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout);
}

//--------------------------------------------------------------------------------------------------
// Setting up the buffers in the descriptor set
//
void HelloVulkan::updateDescriptorSet()
{
  std::vector<vk::WriteDescriptorSet> writes;

  //compute shader
  vk::DescriptorBufferInfo dbiComShdr = {buffer, 0, m_bufferSize};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, 0, &dbiComShdr));
  // Camera matrices and scene description
  vk::DescriptorBufferInfo dbiUnif{m_cameraMat.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, 1, &dbiUnif));
  //vk::DescriptorBufferInfo dbiSceneDesc{  .buffer, 0, VK_WHOLE_SIZE};
  //writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, 2, &dbiSceneDesc));

  //// All material buffers, 1 buffer per OBJ
  //std::vector<vk::DescriptorBufferInfo> dbiMat;
  //std::vector<vk::DescriptorBufferInfo> dbiMatIdx;
  //std::vector<vk::DescriptorBufferInfo> dbiVert;
  //std::vector<vk::DescriptorBufferInfo> dbiIdx;
  //for(auto& obj : m_objModel)
  //{
  //  dbiMat.emplace_back(obj.matColorBuffer.buffer, 0, VK_WHOLE_SIZE);
  //  dbiMatIdx.emplace_back(obj.matIndexBuffer.buffer, 0, VK_WHOLE_SIZE);
  //  dbiVert.emplace_back(obj.vertexBuffer.buffer, 0, VK_WHOLE_SIZE);
  //  dbiIdx.emplace_back(obj.indexBuffer.buffer, 0, VK_WHOLE_SIZE);
  //}
  //writes.emplace_back(m_descSetLayoutBind.makeWriteArray(m_descSet, 1, dbiMat.data()));
  //writes.emplace_back(m_descSetLayoutBind.makeWriteArray(m_descSet, 4, dbiMatIdx.data()));
  //writes.emplace_back(m_descSetLayoutBind.makeWriteArray(m_descSet, 5, dbiVert.data()));
  //writes.emplace_back(m_descSetLayoutBind.makeWriteArray(m_descSet, 6, dbiIdx.data()));

  //// All texture samplers
  //std::vector<vk::DescriptorImageInfo> diit;
  //for(auto& texture : m_textures)
  //{
  //  diit.push_back(texture.descriptor);
  //}
  //writes.emplace_back(m_descSetLayoutBind.makeWriteArray(m_descSet, 3, diit.data()));


  //TODO: Ray tracing
  //vk::AccelerationStructureKHR                   tlas = m_rtBuilder.getAccelerationStructure();
  //vk::WriteDescriptorSetAccelerationStructureKHR descASInfo;
  //descASInfo.setAccelerationStructureCount(1);
  //descASInfo.setPAccelerationStructures(&tlas);
  //writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, 7, &descASInfo));


  // Writing the information
  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}



void HelloVulkan::createComputePipeline()
{
  VkDescriptorSetLayout descriptorSetLayout = m_descSetLayout;
  VkPipeline            pipeline            = m_graphicsPipeline;
  VkPipelineLayout      pipelineLayout      = m_pipelineLayout;
  /*
        We create a compute pipeline here. 
        */

  /*
        Create a shader module. A shader module basically just encapsulates some shader code.
        */
  uint32_t filelength;
  // the code in comp.spv was created by running the command:
  // glslangValidator.exe -V shader.comp
  std::vector<std::string> paths = defaultSearchPaths;
  auto                     code  = nvh::loadFile("shaders/shader.comp.spv", true, paths);
  std::vector<char>        v;
  std::copy(code.begin(), code.end(), std::back_inserter(v));
#ifdef EMBED_SHADERS
  VkShaderModuleCreateInfo createInfo = {};
  createInfo.sType                    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize                 = sizeof(s_computeShader);
  createInfo.pCode                    = reinterpret_cast<const uint32_t*>(s_computeShader);
#else
  createInfo.pCode                    = reinterpret_cast<const uint32_t*>(v.data());
  createInfo.codeSize                 = v.size();
#endif

  NVVK_CHECK(vkCreateShaderModule(m_device, &createInfo, NULL, &computeShaderModule));

  /*
        Now let us actually create the compute pipeline.
        A compute pipeline is very simple compared to a graphics pipeline.
        It only consists of a single stage with a compute shader. 
        So first we specify the compute shader stage, and it's entry point(main).
        */
  VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
  shaderStageCreateInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shaderStageCreateInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
  shaderStageCreateInfo.module = computeShaderModule;
  shaderStageCreateInfo.pName  = "main";

  /*
        The pipeline layout allows the pipeline to access descriptor sets. 
        So we just specify the descriptor set layout we created earlier.
        */
  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
  pipelineLayoutCreateInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutCreateInfo.setLayoutCount = 1;
  pipelineLayoutCreateInfo.pSetLayouts    = &descriptorSetLayout;
  NVVK_CHECK(vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, NULL, &pipelineLayout));

  VkComputePipelineCreateInfo pipelineCreateInfo = {};
  pipelineCreateInfo.sType                       = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineCreateInfo.stage                       = shaderStageCreateInfo;
  pipelineCreateInfo.layout                      = pipelineLayout;

  /*
        Now, we finally create the compute pipeline. 
        */
  NVVK_CHECK(
      vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, NULL, &pipeline));

  m_graphicsPipeline = pipeline;
  m_pipelineLayout   = pipelineLayout;
  m_descSetLayout    = descriptorSetLayout;
}

//--------------------------------------------------------------------------------------------------
// Creating the uniform buffer holding the camera matrices
// - Buffer is host visible
//
void HelloVulkan::createUniformBuffer()
{
  using vkBU = vk::BufferUsageFlagBits;
  using vkMP = vk::MemoryPropertyFlagBits;

  m_cameraMat = m_alloc.createBuffer(sizeof(CameraMatrices), vkBU::eUniformBuffer,
                                     vkMP::eHostVisible | vkMP::eHostCoherent);
  m_debug.setObjectName(m_cameraMat.buffer, "cameraMat");
}

//--------------------------------------------------------------------------------------------------
// Create a storage buffer containing the description of the scene elements
// - Which geometry is used by which instance
// - Transformation
// - Offset for texture
//
void HelloVulkan::createSceneDescriptionBuffer()
{
  using vkBU = vk::BufferUsageFlagBits;
  nvvk::CommandPool cmdGen(m_device, m_graphicsQueueIndex);

  auto cmdBuf = cmdGen.createCommandBuffer();
  m_sceneDesc = m_alloc.createBuffer(cmdBuf, m_objInstance, vkBU::eStorageBuffer);
  cmdGen.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();
  m_debug.setObjectName(m_sceneDesc.buffer, "sceneDesc");
}

//--------------------------------------------------------------------------------------------------
// Creating all textures and samplers
//
void HelloVulkan::createTextureImages(const vk::CommandBuffer&        cmdBuf,
                                      const std::vector<std::string>& textures)
{
  using vkIU = vk::ImageUsageFlagBits;

  vk::SamplerCreateInfo samplerCreateInfo{
      {}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear};
  samplerCreateInfo.setMaxLod(FLT_MAX);
  vk::Format format = vk::Format::eR8G8B8A8Srgb;

  // If no textures are present, create a dummy one to accommodate the pipeline layout
  if(textures.empty() && m_textures.empty())
  {
    nvvk::Texture texture;

    std::array<uint8_t, 4> color{255u, 255u, 255u, 255u};
    vk::DeviceSize         bufferSize      = sizeof(color);
    auto                   imgSize         = vk::Extent2D(1, 1);
    auto                   imageCreateInfo = nvvk::makeImage2DCreateInfo(imgSize, format);

    // Creating the VKImage
    nvvk::Image image = m_alloc.createImage(cmdBuf, bufferSize, color.data(), imageCreateInfo);
    vk::ImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
    texture                        = m_alloc.createTexture(image, ivInfo, samplerCreateInfo);

    // The image format must be in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    nvvk::cmdBarrierImageLayout(cmdBuf, texture.image, vk::ImageLayout::eUndefined,
                                vk::ImageLayout::eShaderReadOnlyOptimal);
    m_textures.push_back(texture);
  }
  else
  {
    // Uploading all images
    for(const auto& texture : textures)
    {
      std::stringstream o;
      int               texWidth, texHeight, texChannels;
      o << "media/textures/" << texture;
      std::string txtFile = nvh::findFile(o.str(), defaultSearchPaths);

      stbi_uc* stbi_pixels = stbi_load(txtFile.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

      std::array<stbi_uc, 4> color{255u, 0u, 255u, 255u};

      stbi_uc* pixels = stbi_pixels;
      // Handle failure
      if(!stbi_pixels)
      {
        texWidth = texHeight = 1;
        texChannels          = 4;
        pixels               = reinterpret_cast<stbi_uc*>(color.data());
      }

      vk::DeviceSize bufferSize = static_cast<uint64_t>(texWidth) * texHeight * sizeof(uint8_t) * 4;
      auto           imgSize    = vk::Extent2D(texWidth, texHeight);
      auto imageCreateInfo = nvvk::makeImage2DCreateInfo(imgSize, format, vkIU::eSampled, true);

      {
        nvvk::Image image = m_alloc.createImage(cmdBuf, bufferSize, pixels, imageCreateInfo);
        nvvk::cmdGenerateMipmaps(cmdBuf, image.image, format, imgSize, imageCreateInfo.mipLevels);
        vk::ImageViewCreateInfo ivInfo =
            nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
        nvvk::Texture texture = m_alloc.createTexture(image, ivInfo, samplerCreateInfo);

        m_textures.push_back(texture);
      }

      stbi_image_free(stbi_pixels);
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Destroying all allocations
//
void HelloVulkan::destroyResources()
{
  vkFreeMemory(m_device, bufferMemory, NULL);
  vkDestroyBuffer(m_device, buffer, NULL);
  vkDestroyShaderModule(m_device, computeShaderModule, NULL);

  m_device.destroy(m_graphicsPipeline);
  m_device.destroy(m_pipelineLayout);
  m_device.destroy(m_descPool);
  m_device.destroy(m_descSetLayout);
  m_alloc.destroy(m_cameraMat);


  for(auto& t : m_textures)
  {
    m_alloc.destroy(t);
  }

}


//--------------------------------------------------------------------------------------------------
// Handling resize of the window
//
void HelloVulkan::onResize(int /*w*/, int /*h*/)
{
}

