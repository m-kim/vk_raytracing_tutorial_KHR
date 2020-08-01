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

extern std::vector<std::string> defaultSearchPaths;

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

#include <lodepng.h>

// Holding the camera matrices
struct CameraMatrices
{
  nvmath::mat4f view;
  nvmath::mat4f proj;
  nvmath::mat4f viewInverse;
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
  //ubo.proj[1][1] *= -1;  // Inverting Y for Vulkan
  ubo.viewInverse = nvmath::invert(ubo.view);
  void* data      = m_device.mapMemory(m_cameraMat.allocation, 0, sizeof(ubo));
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

  // Camera matrices (binding = 0)
  m_descSetLayoutBind.addBinding(vkDS(0, vkDT::eUniformBuffer, 1, vkSS::eVertex));


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

  // Camera matrices and scene description
  vk::DescriptorBufferInfo dbiUnif{m_cameraMat.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, 0, &dbiUnif));


  // Writing the information
  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}


auto HelloVulkan::addShader(const std::string&      code,
                                                       vk::ShaderStageFlagBits stage,
                                                       const char*             entryPoint)
{
  std::vector<char> v;
  std::copy(code.begin(), code.end(), std::back_inserter(v));

  VkShaderModuleCreateInfo createInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  createInfo.codeSize = sizeof(char) * v.size();
  createInfo.pCode    = reinterpret_cast<const uint32_t*>(v.data());
  VkShaderModule shaderModule;
  vkCreateShaderModule(m_device, &createInfo, nullptr, &shaderModule);
  //temporaryModules.push_back(shaderModule);
  VkPipelineShaderStageCreateInfo shaderStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  shaderStage.stage  = (VkShaderStageFlagBits)stage;
  shaderStage.module = shaderModule;
  shaderStage.pName  = entryPoint;
\
  return std::make_tuple(shaderStage, shaderModule);
}

//--------------------------------------------------------------------------------------------------
// Creating the pipeline layout
//
void HelloVulkan::createGraphicsPipeline()
{
  using vkSS = vk::ShaderStageFlagBits;

  vk::PushConstantRange pushConstantRanges = {vkSS::eVertex | vkSS::eFragment, 0,
                                              sizeof(ObjPushConstant)};

  // Creating the Pipeline Layout
  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
  vk::DescriptorSetLayout      descSetLayout(m_descSetLayout);
  pipelineLayoutCreateInfo.setSetLayoutCount(1);
  pipelineLayoutCreateInfo.setPSetLayouts(&descSetLayout);
  pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
  pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstantRanges);
  m_pipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);

  //// Creating the Pipeline
  //std::vector<std::string>                paths = defaultSearchPaths;
  //nvvk::GraphicsPipelineGeneratorCombined gpb(m_device, m_pipelineLayout, m_renderPass);
  //gpb.depthStencilState.depthTestEnable = true;
  //gpb.addShader(nvh::loadFile("shaders/vert_shader.vert.spv", true, paths), vkSS::eVertex);
  //gpb.addShader(nvh::loadFile("shaders/frag_shader.frag.spv", true, paths), vkSS::eFragment);
  //gpb.addBindingDescription({0, sizeof(VertexObj)});
  //gpb.addAttributeDescriptions(std::vector<vk::VertexInputAttributeDescription>{
  //    {0, 0, vk::Format::eR32G32B32Sfloat, offsetof(VertexObj, pos)},
  //    {1, 0, vk::Format::eR32G32B32Sfloat, offsetof(VertexObj, nrm)},
  //    {2, 0, vk::Format::eR32G32B32Sfloat, offsetof(VertexObj, color)},
  //    {3, 0, vk::Format::eR32G32Sfloat, offsetof(VertexObj, texCoord)}});

  //m_graphicsPipeline = gpb.createPipeline();
  //m_debug.setObjectName(m_graphicsPipeline, "Graphics");


  VkPipelineInputAssemblyStateCreateInfo pipelineInputAssemblyStateCreateInfo{};
  pipelineInputAssemblyStateCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  pipelineInputAssemblyStateCreateInfo.topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  pipelineInputAssemblyStateCreateInfo.flags                  = 0;
  pipelineInputAssemblyStateCreateInfo.primitiveRestartEnable = VK_FALSE;

  VkPipelineRasterizationStateCreateInfo pipelineRasterizationStateCreateInfo{};
  pipelineRasterizationStateCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  pipelineRasterizationStateCreateInfo.polygonMode      = VK_POLYGON_MODE_FILL;
  pipelineRasterizationStateCreateInfo.cullMode         = VK_CULL_MODE_BACK_BIT;
  pipelineRasterizationStateCreateInfo.frontFace        = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  pipelineRasterizationStateCreateInfo.flags            = 0;
  pipelineRasterizationStateCreateInfo.depthClampEnable = VK_FALSE;
  pipelineRasterizationStateCreateInfo.lineWidth        = 1.0f;

  VkPipelineColorBlendAttachmentState pipelineColorBlendAttachmentState{};
  pipelineColorBlendAttachmentState.colorWriteMask = 0xf;
  pipelineColorBlendAttachmentState.blendEnable    = VK_FALSE;

  VkPipelineColorBlendStateCreateInfo pipelineColorBlendStateCreateInfo{};
  pipelineColorBlendStateCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  pipelineColorBlendStateCreateInfo.attachmentCount = 1;
  pipelineColorBlendStateCreateInfo.pAttachments    = &pipelineColorBlendAttachmentState;

  VkPipelineDepthStencilStateCreateInfo pipelineDepthStencilStateCreateInfo{};
  pipelineDepthStencilStateCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  pipelineDepthStencilStateCreateInfo.depthTestEnable  = VK_TRUE;
  pipelineDepthStencilStateCreateInfo.depthWriteEnable = VK_TRUE;
  pipelineDepthStencilStateCreateInfo.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;
  pipelineDepthStencilStateCreateInfo.back.compareOp   = VK_COMPARE_OP_ALWAYS;


  VkPipelineViewportStateCreateInfo pipelineViewportStateCreateInfo{};
  pipelineViewportStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  pipelineViewportStateCreateInfo.viewportCount = 1;
  pipelineViewportStateCreateInfo.scissorCount  = 1;
  pipelineViewportStateCreateInfo.flags         = 0;

  VkPipelineMultisampleStateCreateInfo pipelineMultisampleStateCreateInfo{};
  pipelineMultisampleStateCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  pipelineMultisampleStateCreateInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  pipelineMultisampleStateCreateInfo.flags                = 0;

  std::vector<VkDynamicState>      dynamicStateEnables = {VK_DYNAMIC_STATE_VIEWPORT,
                                                     VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo pipelineDynamicStateCreateInfo{};
  pipelineDynamicStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  pipelineDynamicStateCreateInfo.pDynamicStates = dynamicStateEnables.data();
  pipelineDynamicStateCreateInfo.dynamicStateCount =
      static_cast<uint32_t>(dynamicStateEnables.size());
  pipelineDynamicStateCreateInfo.flags = 0;

  // Vertex bindings and attributes
  // Binding description
  VkVertexInputBindingDescription vInputBindDescription{};
  vInputBindDescription.binding   = 0;
  vInputBindDescription.stride    = sizeof(VertexObj);
  vInputBindDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  // Attribute descriptions
  std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
      vk::VertexInputAttributeDescription{0, 0, vk::Format::eR32G32B32Sfloat,
                                          offsetof(VertexObj, pos)},
      vk::VertexInputAttributeDescription{1, 0, vk::Format::eR32G32B32Sfloat,
                                          offsetof(VertexObj, nrm)},
      vk::VertexInputAttributeDescription{2, 0, vk::Format::eR32G32B32Sfloat,
                                          offsetof(VertexObj, color)},
      vk::VertexInputAttributeDescription{3, 0, vk::Format::eR32G32Sfloat,
                                          offsetof(VertexObj, texCoord)}};

  VkPipelineVertexInputStateCreateInfo pipelineVertexInputStateCreateInfo{};
  pipelineVertexInputStateCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

  VkPipelineVertexInputStateCreateInfo vertexInputState{};
  vertexInputState.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInputState.vertexBindingDescriptionCount = static_cast<uint32_t>(1);
  vertexInputState.pVertexBindingDescriptions    = &vInputBindDescription;
  vertexInputState.vertexAttributeDescriptionCount =
      static_cast<uint32_t>(vertexInputAttributes.size());
  vertexInputState.pVertexAttributeDescriptions = vertexInputAttributes.data();

  //std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = {
  //    loadShader(getShadersPath() + "screenshot/mesh.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
  //    loadShader(getShadersPath() + "screenshot/mesh.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT),
  //};
  std::vector<VkShaderModule> shaderModules;
  std::vector<VkPipelineShaderStageCreateInfo> shaderStages;

  std::vector<std::string>    paths = defaultSearchPaths;
  auto vertShad= addShader(nvh::loadFile("shaders/vert_shader.vert.spv", true, paths), vkSS::eVertex);
  shaderStages.push_back(std::get<0>(vertShad));
  shaderModules.push_back(std::get<1>(vertShad));
  auto fragShad =
      addShader(nvh::loadFile("shaders/frag_shader.frag.spv", true, paths), vkSS::eFragment);
  shaderStages.push_back(std::get<0>(fragShad));
  shaderModules.push_back(std::get<1>(fragShad));

  //VkGraphicsPipelineCreateInfo pipelineCreateInfo =
  //vks::initializers::pipelineCreateInfo(pipelineLayout, renderPass, 0);
  VkGraphicsPipelineCreateInfo pipelineCreateInfo{};
  pipelineCreateInfo.sType              = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineCreateInfo.layout             = m_pipelineLayout;
  pipelineCreateInfo.renderPass         = m_renderPass;
  pipelineCreateInfo.flags              = 0;
  pipelineCreateInfo.basePipelineIndex  = -1;
  pipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;

  pipelineCreateInfo.pInputAssemblyState = &pipelineInputAssemblyStateCreateInfo;
  pipelineCreateInfo.pRasterizationState = &pipelineRasterizationStateCreateInfo;
  pipelineCreateInfo.pColorBlendState    = &pipelineColorBlendStateCreateInfo;
  pipelineCreateInfo.pMultisampleState   = &pipelineMultisampleStateCreateInfo;
  pipelineCreateInfo.pViewportState      = &pipelineViewportStateCreateInfo;
  pipelineCreateInfo.pDepthStencilState  = &pipelineDepthStencilStateCreateInfo;
  pipelineCreateInfo.pDynamicState       = &pipelineDynamicStateCreateInfo;
  pipelineCreateInfo.stageCount          = shaderStages.size();
  pipelineCreateInfo.pStages             = shaderStages.data();
  pipelineCreateInfo.pVertexInputState   = &vertexInputState;

  VkPipeline pipeline = m_graphicsPipeline;
  NVVK_CHECK(vkCreateGraphicsPipelines(m_device, m_pipelineCache, 1, &pipelineCreateInfo, nullptr,
                                       &pipeline));
  m_graphicsPipeline = pipeline;

  for(auto mod : shaderModules)
  {
    vkDestroyShaderModule(m_device, mod, nullptr);
  }

}

//--------------------------------------------------------------------------------------------------
// Loading the OBJ file and setting up all buffers
//
void HelloVulkan::loadModel(const std::string& filename, nvmath::mat4f transform)
{
  std::vector<VertexObj> vertices(3);
  vertices[0] = {{-.5, 0.0, 0.0}, {0, 0, 0}, {1.0, 0.0, 0.0}, {0, 0}};
  vertices[1] = {{.5, 0, 0.0}, {0, 0, 0}, {0.0, 1.0, 0.0}, {0, 0}};
  vertices[2] = {{0, .5, 0.0}, {0, 0, 0}, {0.0, 0.0, 1.0}, {0, 0}};

  std::vector<uint32_t> indices{0, 1, 2};
  uint32_t              indicesSize = indices.size() * sizeof(uint32_t);

  std::vector<MaterialObj> materials(3);

 using vkBU = vk::BufferUsageFlagBits;

  ObjLoader loader;
  
  loader.m_indices = indices;
  loader.m_materials = materials;
  loader.m_vertices  = vertices;
  loader.m_matIndx   = indices;

  // Converting from Srgb to linear
  for(auto& m : loader.m_materials)
  {
    m.ambient  = nvmath::pow(m.ambient, 2.2f);
    m.diffuse  = nvmath::pow(m.diffuse, 2.2f);
    m.specular = nvmath::pow(m.specular, 2.2f);
  }

  ObjInstance instance;
  instance.objIndex    = static_cast<uint32_t>(m_objModel.size());
  instance.transform   = transform;
  instance.transformIT = nvmath::transpose(nvmath::invert(transform));
  instance.txtOffset   = static_cast<uint32_t>(m_textures.size());

  ObjModel model;
  model.nbIndices  = static_cast<uint32_t>(loader.m_indices.size());
  model.nbVertices = static_cast<uint32_t>(loader.m_vertices.size());

  // Create the buffers on Device and copy vertices, indices and materials
  nvvk::CommandPool cmdBufGet(m_device, m_graphicsQueueIndex);
  vk::CommandBuffer cmdBuf = cmdBufGet.createCommandBuffer();
  model.vertexBuffer       = m_alloc.createBuffer(cmdBuf, loader.m_vertices, vkBU::eVertexBuffer);
  model.indexBuffer        = m_alloc.createBuffer(cmdBuf, loader.m_indices, vkBU::eIndexBuffer);
  model.matColorBuffer     = m_alloc.createBuffer(cmdBuf, loader.m_materials, vkBU::eStorageBuffer);
  model.matIndexBuffer     = m_alloc.createBuffer(cmdBuf, loader.m_matIndx, vkBU::eStorageBuffer);
  // Creates all textures found
  createTextureImages(cmdBuf, loader.m_textures);
  cmdBufGet.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();

  std::string objNb = std::to_string(instance.objIndex);
  m_debug.setObjectName(model.vertexBuffer.buffer, (std::string("vertex_" + objNb).c_str()));
  m_debug.setObjectName(model.indexBuffer.buffer, (std::string("index_" + objNb).c_str()));
  m_debug.setObjectName(model.matColorBuffer.buffer, (std::string("mat_" + objNb).c_str()));
  m_debug.setObjectName(model.matIndexBuffer.buffer, (std::string("matIdx_" + objNb).c_str()));

  m_objModel.emplace_back(model);
  m_objInstance.emplace_back(instance);
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
    imageCreateInfo.setUsage(vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled);
    // Creating the dummy texure
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
        nvvk::ImageDedicated image =
            m_alloc.createImage(cmdBuf, bufferSize, pixels, imageCreateInfo);
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
  m_device.destroy(m_graphicsPipeline);
  m_device.destroy(m_pipelineLayout);
  m_device.destroy(m_descPool);
  m_device.destroy(m_descSetLayout);
  m_alloc.destroy(m_cameraMat);
  m_alloc.destroy(m_sceneDesc);

  for(auto& m : m_objModel)
  {
    m_alloc.destroy(m.vertexBuffer);
    m_alloc.destroy(m.indexBuffer);
    m_alloc.destroy(m.matColorBuffer);
    m_alloc.destroy(m.matIndexBuffer);
  }

  for(auto& t : m_textures)
  {
    m_alloc.destroy(t);
  }


}

//--------------------------------------------------------------------------------------------------
// Drawing the scene in raster mode
//
void HelloVulkan::rasterize(const vk::CommandBuffer& cmdBuf)
{
  using vkPBP = vk::PipelineBindPoint;
  using vkSS  = vk::ShaderStageFlagBits;
  vk::DeviceSize offset{0};

  m_debug.beginLabel(cmdBuf, "Rasterize");

  // Dynamic Viewport
  cmdBuf.setViewport(0, {vk::Viewport(0, 0, (float)m_size.width, (float)m_size.height, 0, 1)});
  cmdBuf.setScissor(0, {{{0, 0}, {m_size.width, m_size.height}}});


  // Drawing all triangles
  cmdBuf.bindPipeline(vkPBP::eGraphics, m_graphicsPipeline);
  cmdBuf.bindDescriptorSets(vkPBP::eGraphics, m_pipelineLayout, 0, {m_descSet}, {});
  for(int i = 0; i < m_objInstance.size(); ++i)
  {
    auto& inst                = m_objInstance[i];
    auto& model               = m_objModel[inst.objIndex];
    m_pushConstant.instanceId = i;  // Telling which instance is drawn
    cmdBuf.pushConstants<ObjPushConstant>(m_pipelineLayout, vkSS::eVertex | vkSS::eFragment, 0,
                                          m_pushConstant);

    cmdBuf.bindVertexBuffers(0, {model.vertexBuffer.buffer}, {offset});
    cmdBuf.bindIndexBuffer(model.indexBuffer.buffer, 0, vk::IndexType::eUint32);
    cmdBuf.drawIndexed(model.nbIndices, 1, 0, 0, 0);
  }
  m_debug.endLabel(cmdBuf);
}

//--------------------------------------------------------------------------------------------------
// Handling resize of the window
//
void HelloVulkan::onResize(int /*w*/, int /*h*/)
{
}
void HelloVulkan::createSurface(const vk::SurfaceKHR& surface,
                           uint32_t              width,
                           uint32_t              height,
                           vk::Format            colorFormat,
                           vk::Format            depthFormat,
                           bool                  vsync)
{
  m_size        = vk::Extent2D(width, height);
  m_depthFormat = depthFormat;
  m_colorFormat = colorFormat;
  m_vsync       = vsync;

  m_swapChain.init(m_device, m_physicalDevice, m_queue, m_graphicsQueueIndex, surface,
                   static_cast<VkFormat>(colorFormat));
  updateSwapchain(m_size.width, m_size.height, vsync);
  m_colorFormat = static_cast<vk::Format>(m_swapChain.getFormat());

  // Create Synchronization Primitives
  m_waitFences.resize(m_swapChain.getImageCount());
  for(auto& fence : m_waitFences)
  {
    fence = m_device.createFence({vk::FenceCreateFlagBits::eSignaled});
  }

  // Command buffers store a reference to the frame buffer inside their render pass info
  // so for static usage without having to rebuild them each frame, we use one per frame buffer
  m_commandBuffers = m_device.allocateCommandBuffers(
      {m_cmdPool, vk::CommandBufferLevel::ePrimary, m_swapChain.getImageCount()});

#ifdef _DEBUG
  for(size_t i = 0; i < m_commandBuffers.size(); i++)
  {
    std::string name = std::string("AppBase") + std::to_string(i);
    m_device.setDebugUtilsObjectNameEXT({vk::ObjectType::eCommandBuffer,
                                         reinterpret_cast<const uint64_t&>(m_commandBuffers[i]),
                                         name.c_str()});
  }
#endif  // _DEBUG

  // Setup camera
  CameraManip.setWindowSize(m_size.width, m_size.height);
}


VkExtent2D HelloVulkan::updateSwapchain(int width, int height, bool vsync) {
  m_swapChain.m_changeID++;

  VkResult       err;
  VkSwapchainKHR oldSwapchain = m_swapChain.getSwapchain();//m_swapchain;

  err = vkDeviceWaitIdle(m_device);
  if(nvvk::checkResult(err, __FILE__, __LINE__))
  {
    exit(-1);
  }
  // Check the surface capabilities and formats
  VkSurfaceCapabilitiesKHR surfCapabilities;
  err = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physicalDevice, m_surface, &surfCapabilities);
  assert(!err);

  uint32_t presentModeCount;
  err = vkGetPhysicalDeviceSurfacePresentModesKHR(m_physicalDevice, m_surface, &presentModeCount,
                                                  nullptr);
  assert(!err);
  std::vector<VkPresentModeKHR> presentModes(presentModeCount);
  err = vkGetPhysicalDeviceSurfacePresentModesKHR(m_physicalDevice, m_surface, &presentModeCount,
                                                  presentModes.data());
  assert(!err);

  VkExtent2D swapchainExtent;
  // width and height are either both -1, or both not -1.
  if(surfCapabilities.currentExtent.width == (uint32_t)-1)
  {
    // If the surface size is undefined, the size is set to
    // the size of the images requested.
    swapchainExtent.width  = width;
    swapchainExtent.height = height;
  }
  else
  {
    // If the surface size is defined, the swap chain size must match
    swapchainExtent = surfCapabilities.currentExtent;
  }

  // test against valid size, typically hit when windows are minimized, the app must
  // prevent triggering this code accordingly
  assert(swapchainExtent.width && swapchainExtent.height);

  // everyone must support FIFO mode
  VkPresentModeKHR swapchainPresentMode = VK_PRESENT_MODE_FIFO_KHR;
  // no vsync try to find a faster alternative to FIFO
  if(!vsync)
  {
    for(uint32_t i = 0; i < presentModeCount; i++)
    {
      if(presentModes[i] == VK_PRESENT_MODE_MAILBOX_KHR)
      {
        // prefer mailbox due to no tearing
        swapchainPresentMode = VK_PRESENT_MODE_MAILBOX_KHR;
        break;
      }
      if(presentModes[i] == VK_PRESENT_MODE_IMMEDIATE_KHR)
      {
        swapchainPresentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
      }
    }
  }

  // Determine the number of VkImage's to use in the swap chain (we desire to
  // own only 1 image at a time, besides the images being displayed and
  // queued for display):
  uint32_t desiredNumberOfSwapchainImages = surfCapabilities.minImageCount + 1;
  if((surfCapabilities.maxImageCount > 0)
     && (desiredNumberOfSwapchainImages > surfCapabilities.maxImageCount))
  {
    // Application must settle for fewer images than desired:
    desiredNumberOfSwapchainImages = surfCapabilities.maxImageCount;
  }

  VkSurfaceTransformFlagBitsKHR preTransform;
  if(surfCapabilities.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
  {
    preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
  }
  else
  {
    preTransform = surfCapabilities.currentTransform;
  }

  VkSwapchainCreateInfoKHR swapchain = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  swapchain.surface                  = m_swapChain.m_surface;
  swapchain.minImageCount            = desiredNumberOfSwapchainImages;
  swapchain.imageFormat              = m_swapChain.m_surfaceFormat;
  swapchain.imageColorSpace          = m_swapChain.m_surfaceColor;
  swapchain.imageExtent              = swapchainExtent;

  //This is why this is recreated here from swapchain_vk
  //VK_IMAGE_USAGE_TRANSFER_SRC_BIT is not included in swapchain_VK
  //but it's needed if we want to map to cpu memory.
  swapchain.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT
                         | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  swapchain.preTransform          = preTransform;
  swapchain.compositeAlpha        = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  swapchain.imageArrayLayers      = 1;
  swapchain.imageSharingMode      = VK_SHARING_MODE_EXCLUSIVE;
  swapchain.queueFamilyIndexCount = 1;
  swapchain.pQueueFamilyIndices   = &m_swapChain.m_queueFamilyIndex;
  swapchain.presentMode           = swapchainPresentMode;
  swapchain.oldSwapchain          = oldSwapchain;
  swapchain.clipped               = true;

  err = vkCreateSwapchainKHR(m_device, &swapchain, nullptr, &m_swapChain.m_swapchain);
  assert(!err);

  nvvk::DebugUtil debugUtil(m_device);

  debugUtil.setObjectName(m_swapChain.m_swapchain, "SwapChain::m_swapchain");

  // If we just re-created an existing swapchain, we should destroy the old
  // swapchain at this point.
  // Note: destroying the swapchain also cleans up all its associated
  // presentable images once the platform is done with them.
  if(oldSwapchain != VK_NULL_HANDLE)
  {
    for(auto it : m_swapChain.m_entries)
    {
      vkDestroyImageView(m_device, it.imageView, nullptr);
      vkDestroySemaphore(m_device, it.readSemaphore, nullptr);
      vkDestroySemaphore(m_device, it.writtenSemaphore, nullptr);
    }
    vkDestroySwapchainKHR(m_device, oldSwapchain, nullptr);
  }

  err =
      vkGetSwapchainImagesKHR(m_device, m_swapChain.m_swapchain, &m_swapChain.m_imageCount, nullptr);
  assert(!err);

  m_swapChain.m_entries.resize(m_swapChain.m_imageCount);
  m_swapChain.m_barriers.resize(m_swapChain.m_imageCount);

  std::vector<VkImage> images(m_swapChain.m_imageCount);

  err = vkGetSwapchainImagesKHR(m_device, m_swapChain.m_swapchain, &m_swapChain.m_imageCount,
                                images.data());
  assert(!err);
  //
  // Image views
  //
  for(uint32_t i = 0; i < m_swapChain.m_imageCount; i++)
  {
    nvvk::SwapChain::Entry & entry = m_swapChain.m_entries[i];

    // image
    entry.image = images[i];

    // imageview
    VkImageViewCreateInfo viewCreateInfo = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                            NULL,
                                            0,
                                            entry.image,
                                            VK_IMAGE_VIEW_TYPE_2D,
                                            m_swapChain.m_surfaceFormat,
                                            {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G,
                                             VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A},
                                            {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};

    err = vkCreateImageView(m_device, &viewCreateInfo, nullptr, &entry.imageView);
    assert(!err);

    // semaphore
    VkSemaphoreCreateInfo semCreateInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};

    err = vkCreateSemaphore(m_device, &semCreateInfo, nullptr, &entry.readSemaphore);
    assert(!err);
    err = vkCreateSemaphore(m_device, &semCreateInfo, nullptr, &entry.writtenSemaphore);
    assert(!err);

    // initial barriers
    VkImageSubresourceRange range = {0};
    range.aspectMask              = VK_IMAGE_ASPECT_COLOR_BIT;
    range.baseMipLevel            = 0;
    range.levelCount              = VK_REMAINING_MIP_LEVELS;
    range.baseArrayLayer          = 0;
    range.layerCount              = VK_REMAINING_ARRAY_LAYERS;

    VkImageMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    memBarrier.sType                = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    memBarrier.dstAccessMask        = 0;
    memBarrier.srcAccessMask        = 0;
    memBarrier.oldLayout            = VK_IMAGE_LAYOUT_UNDEFINED;
    memBarrier.newLayout            = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    memBarrier.image                = entry.image;
    memBarrier.subresourceRange     = range;

    m_swapChain.m_barriers[i] = memBarrier;

    debugUtil.setObjectName(entry.image, "swapchainImage:" + std::to_string(i));
    debugUtil.setObjectName(entry.imageView, "swapchainImageView:" + std::to_string(i));
    debugUtil.setObjectName(entry.readSemaphore, "swapchainReadSemaphore:" + std::to_string(i));
    debugUtil.setObjectName(entry.writtenSemaphore,
                            "swapchainWrittenSemaphore:" + std::to_string(i));
  }
  m_swapChain.m_updateWidth  = width;
  m_swapChain.m_updateHeight = height;
  m_swapChain.m_vsync        = vsync;
  m_swapChain.m_extent       = swapchainExtent;

  m_swapChain.m_currentSemaphore = 0;
  m_swapChain.m_currentImage     = 0;

  return swapchainExtent;

  }
  void HelloVulkan::saveRenderedImage() {
  auto data = readPixels();
  lodepng::encode("output.png", (unsigned char *)data.data(), m_size.width, m_size.height);

  std::cout << "Screenshot saved to disk" << std::endl;
}

  