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

#include<iostream>
#include <fstream>

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

  void saveRenderedImage()
  {

    using vkBU = vk::BufferUsageFlagBits;
    using vkMP = vk::MemoryPropertyFlagBits;

    uint32_t imageIndex = m_swapChain.getActiveImageIndex();
    VkImage  srcImage   = m_swapChain.getImage(imageIndex);
    //VkImage           srcImage   = m_swapChain.getActiveImage();
    VkImageCreateInfo imageCreateCI = {};

    imageCreateCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;

    imageCreateCI.imageType = VK_IMAGE_TYPE_2D;
    // Note that vkCmdBlitImage (if supported) will also do format conversions if the swapchain color format would differ
    imageCreateCI.format = VK_FORMAT_R8G8B8A8_UNORM;//VK_FORMAT_R32G32B32A32_SFLOAT;
    imageCreateCI.extent.width  = m_size.width;
    imageCreateCI.extent.height = m_size.height;
    imageCreateCI.extent.depth  = 1;
    imageCreateCI.arrayLayers   = 1;
    imageCreateCI.mipLevels     = 1;
    imageCreateCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageCreateCI.samples       = VK_SAMPLE_COUNT_1_BIT;
    imageCreateCI.tiling        = VK_IMAGE_TILING_LINEAR;
    imageCreateCI.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageCreateCI.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    //imageCreateCI = nvvk::makeImage2DCreateInfo(m_size, m_offscreenColorFormat,
    //                                            vk::ImageUsageFlagBits::eTransferDst);
    //imageCreateCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;


    //// Create the image
    VkImage dstImage;
    NVVK_CHECK(vkCreateImage(m_device, &imageCreateCI, nullptr, &dstImage));

    //// Create memory to back up the image
    VkMemoryRequirements memRequirements = {};
    VkMemoryAllocateInfo memAllocInfo{};
    memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    VkDeviceMemory dstImageMemory;
    vkGetImageMemoryRequirements(m_device, dstImage, &memRequirements);

    memAllocInfo.allocationSize = memRequirements.size;
    // Memory must be host visible to copy from
    memAllocInfo.memoryTypeIndex =
        findMemoryType(memRequirements.memoryTypeBits,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    NVVK_CHECK(vkAllocateMemory(m_device, &memAllocInfo, nullptr, &dstImageMemory));
    NVVK_CHECK(vkBindImageMemory(m_device, dstImage, dstImageMemory, 0));

    //// Do the actual blit from the swapchain image to our host visible destination image
    //nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
    //auto              cmdBuf = genCmdBuf.createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
    VkCommandBufferAllocateInfo cmdBufAllocateInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmdBufAllocateInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdBufAllocateInfo.commandPool        = m_cmdPool;
    cmdBufAllocateInfo.commandBufferCount = 1;
    VkCommandBuffer cmdBuffer;
    NVVK_CHECK(vkAllocateCommandBuffers(m_device, &cmdBufAllocateInfo, &cmdBuffer));
    VkCommandBufferBeginInfo cmdBufInfo{};
    cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    NVVK_CHECK(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));


    //VkImageMemoryBarrier imageMemoryBarrier;
    //imageMemoryBarrier.sType                = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    //imageMemoryBarrier.srcQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
    //imageMemoryBarrier.dstQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
    //imageMemoryBarrier.srcAccessMask        = 0;
    //imageMemoryBarrier.dstAccessMask        = VK_ACCESS_TRANSFER_WRITE_BIT;
    //imageMemoryBarrier.oldLayout            = VK_IMAGE_LAYOUT_UNDEFINED;
    //imageMemoryBarrier.newLayout            = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    //imageMemoryBarrier.subresourceRange =
    //    VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    //vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
    //                     0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
    nvvk::cmdBarrierImageLayout(cmdBuffer, dstImage, VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    nvvk::cmdBarrierImageLayout(cmdBuffer, srcImage, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    bool supportsBlit = 0;
    if(supportsBlit)
    {
      // Define the region to blit (we will blit the whole swapchain image)
      VkOffset3D blitSize;
      blitSize.x = m_size.width;
      blitSize.y = m_size.height;
      blitSize.z = 1;
      VkImageBlit imageBlitRegion{};
      imageBlitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      imageBlitRegion.srcSubresource.layerCount = 1;
      imageBlitRegion.srcOffsets[1]             = blitSize;
      imageBlitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      imageBlitRegion.dstSubresource.layerCount = 1;
      imageBlitRegion.dstOffsets[1]             = blitSize;

      // Issue the blit command
      vkCmdBlitImage(cmdBuffer, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstImage,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageBlitRegion, VK_FILTER_NEAREST);
    }
    else
    {

      // Otherwise use image copy (requires us to manually flip components)
      VkImageCopy imageCopyRegion{};
      imageCopyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      imageCopyRegion.srcSubresource.layerCount = 1;
      imageCopyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      imageCopyRegion.dstSubresource.layerCount = 1;
      imageCopyRegion.extent.width              = m_size.width;
      imageCopyRegion.extent.height             = m_size.height;
      imageCopyRegion.extent.depth              = 1;

      // Issue the copy command
      vkCmdCopyImage(cmdBuffer, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstImage,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageCopyRegion);
    }

    nvvk::cmdBarrierImageLayout(cmdBuffer, dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                VK_IMAGE_LAYOUT_GENERAL);
    nvvk::cmdBarrierImageLayout(cmdBuffer, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

  vkEndCommandBuffer(cmdBuffer);
    // Get layout of the image (including row pitch)
    VkImageSubresource  subResource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
    VkSubresourceLayout subResourceLayout;
    vkGetImageSubresourceLayout(m_device, dstImage, &subResource, &subResourceLayout);

    // Map image memory so we can start copying from it
    const char* data;
    vkMapMemory(m_device, dstImageMemory, 0, VK_WHOLE_SIZE, 0, (void**)&data);
    data += subResourceLayout.offset;
    std::string   filename("wtf.ppm");
    std::ofstream file(filename, std::ios::out | std::ios::binary);

    // ppm header
    file << "P6\n" << m_size.width << "\n" << m_size.height << "\n" << 255 << "\n";

    // If source is BGR (destination is always RGB) and we can't use blit (which does automatic conversion), we'll have to manually swizzle color components
    bool colorSwizzle = false;   

    		// ppm binary pixel data
    for(uint32_t y = 0; y < m_size.height; y++)
    {
      unsigned int* row = (unsigned int*)data;
      for(uint32_t x = 0; x < m_size.width; x++)
      {
        if(colorSwizzle)
        {
          file.write((char*)row + 2, 1);
          file.write((char*)row + 1, 1);
          file.write((char*)row, 1);
        }
        else
        {
          file.write((char*)row, 3);
        }
        row++;
      }
      data += subResourceLayout.rowPitch;
    }
    file.close();

    std::cout << "Screenshot saved to disk" << std::endl;

    // Clean up resources
    vkUnmapMemory(m_device, dstImageMemory);
    vkFreeMemory(m_device, dstImageMemory, nullptr);
    vkDestroyImage(m_device, dstImage, nullptr);

  }
};
