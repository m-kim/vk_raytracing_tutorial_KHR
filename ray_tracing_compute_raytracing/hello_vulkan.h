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
#include "nvh/fileoperations.hpp"
// #VKRay
#include "nvvk/raytraceKHR_vk.hpp"
#include "lodepng.h"

#define WORKGROUP_SIZE 32
//--------------------------------------------------------------------------------------------------
// Simple rasterizer of OBJ objects
// - Each OBJ loaded are stored in an `ObjModel` and referenced by a `ObjInstance`
// - It is possible to have many `ObjInstance` referencing the same `ObjModel`
// - Rendering is done in an offscreen framebuffer
// - The image of the framebuffer is displayed in post-process in a full-screen quad
//

extern std::vector<std::string> defaultSearchPaths;
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


  // The OBJ model
  struct ObjModel
  {
    uint32_t     nbIndices{0};
    uint32_t     nbVertices{0};
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


  struct Pixel
  {
    float r, g, b, a;
  };
  int WIDTH = 1280;
  int HEIGHT = 720;




  VkShaderModule   computeShaderModule;

  /*
    The mandelbrot set will be rendered to this buffer.
    The memory that backs the buffer is bufferMemory. 
    */
  VkBuffer       buffer;
  VkDeviceMemory bufferMemory;

  uint32_t bufferSize;  // size of `buffer` in bytes.

public:

  void saveRenderedImage()
  {
    void* mappedMemory = NULL;
    // Map the buffer memory, so that we can read from it on the CPU.
    vkMapMemory(m_device, bufferMemory, 0, bufferSize, 0, &mappedMemory);
    Pixel* pmappedMemory = (Pixel*)mappedMemory;

    // Get the color data from the buffer, and cast it to bytes.
    // We save the data to a vector.
    std::vector<unsigned char> image;
    image.reserve(WIDTH * HEIGHT * 4);
    for(int i = 0; i < WIDTH * HEIGHT; i += 1)
    {
      image.push_back((unsigned char)(255.0f * (pmappedMemory[i].r)));
      image.push_back((unsigned char)(255.0f * (pmappedMemory[i].g)));
      image.push_back((unsigned char)(255.0f * (pmappedMemory[i].b)));
      image.push_back((unsigned char)(255.0f * (pmappedMemory[i].a)));
    }
    // Done reading, so unmap.
    vkUnmapMemory(m_device, bufferMemory);

    // Now we save the acquired color data to a .png.
    unsigned error = lodepng::encode("compute.png", image, WIDTH, HEIGHT);
    if(error)
      printf("encoder error %d: %s", error, lodepng_error_text(error));
  }

  
  // find memory type with desired properties.
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
    return -1;
  }

  void createComputeBuffer()
  {
    /*
        We will now create a buffer. We will render the mandelbrot set into this buffer
        in a computer shade later. 
        */

    VkBufferCreateInfo bufferCreateInfo = {};
    bufferCreateInfo.sType              = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size               = bufferSize;  // buffer size in bytes.
    bufferCreateInfo.usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;  // buffer is used as a storage buffer.
    bufferCreateInfo.sharingMode =
        VK_SHARING_MODE_EXCLUSIVE;  // buffer is exclusive to a single queue family at a time.
    VkDevice device = m_device;
    NVVK_CHECK(vkCreateBuffer(device, &bufferCreateInfo, NULL, &buffer));  // create buffer.
    m_device = device;
    /*
        But the buffer doesn't allocate memory for itself, so we must do that manually.
        */

    /*
        First, we find the memory requirements for the buffer.
        */
    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(m_device, buffer, &memoryRequirements);

    /*
        Now use obtained memory requirements info to allocate the memory for the buffer.
        */
    VkMemoryAllocateInfo allocateInfo = {};
    allocateInfo.sType                = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize       = memoryRequirements.size;  // specify required memory.
    /*
        There are several types of memory that can be allocated, and we must choose a memory type that:
        1) Satisfies the memory requirements(memoryRequirements.memoryTypeBits). 
        2) Satifies our own usage requirements. We want to be able to read the buffer memory from the GPU to the CPU
           with vkMapMemory, so we set VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT. 
        Also, by setting VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory written by the device(GPU) will be easily 
        visible to the host(CPU), without having to call any extra flushing commands. So mainly for convenience, we set
        this flag.
        */
    allocateInfo.memoryTypeIndex =
        findMemoryType(memoryRequirements.memoryTypeBits,
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    NVVK_CHECK(vkAllocateMemory(m_device, &allocateInfo, NULL,
                                     &bufferMemory));  // allocate memory on device.

    // Now associate that allocated memory with the buffer. With that, the buffer is backed by actual memory.
    NVVK_CHECK(vkBindBufferMemory(m_device, buffer, bufferMemory, 0));
  }

  void createComputeDescriptorSetLayout()
  {
    VkDescriptorSetLayout descriptorSetLayout = m_descSetLayout;
    /*
        Here we specify a descriptor set layout. This allows us to bind our descriptors to 
        resources in the shader. 
        */

    /*
        Here we specify a binding of type VK_DESCRIPTOR_TYPE_STORAGE_BUFFER to the binding point
        0. This binds to 
          layout(std140, binding = 0) buffer buf
        in the compute shader.
        */
    VkDescriptorSetLayoutBinding descriptorSetLayoutBinding = {};
    descriptorSetLayoutBinding.binding                      = 0;  // binding = 0
    descriptorSetLayoutBinding.descriptorType               = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorSetLayoutBinding.descriptorCount              = 1;
    descriptorSetLayoutBinding.stageFlags                   = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
    descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.bindingCount =
        1;  // only a single binding in this descriptor set layout.
    descriptorSetLayoutCreateInfo.pBindings = &descriptorSetLayoutBinding;

    // Create the descriptor set layout.
    NVVK_CHECK(vkCreateDescriptorSetLayout(m_device, &descriptorSetLayoutCreateInfo, NULL,
                                                &descriptorSetLayout));

    m_descSetLayout = descriptorSetLayout;
  }

  void createDescriptorSet()
  {
    VkDescriptorPool descriptorPool = m_descPool;
    VkDescriptorSetLayout descriptorSetLayout = m_descSetLayout;
    VkDescriptorSet       descriptorSet       = m_descSet;

    /*
        So we will allocate a descriptor set here.
        But we need to first create a descriptor pool to do that. 
        */

    /*
        Our descriptor pool can only allocate a single storage buffer.
        */
    VkDescriptorPoolSize descriptorPoolSize = {};
    descriptorPoolSize.type                 = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorPoolSize.descriptorCount      = 1;

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
    descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.maxSets =
        1;  // we only need to allocate one descriptor set from the pool.
    descriptorPoolCreateInfo.poolSizeCount = 1;
    descriptorPoolCreateInfo.pPoolSizes    = &descriptorPoolSize;

    // create descriptor pool.
    NVVK_CHECK(
        vkCreateDescriptorPool(m_device, &descriptorPoolCreateInfo, NULL, &descriptorPool));

    /*
        With the pool allocated, we can now allocate the descriptor set. 
        */
    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
    descriptorSetAllocateInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.descriptorPool     = descriptorPool;  // pool to allocate from.
    descriptorSetAllocateInfo.descriptorSetCount = 1;  // allocate a single descriptor set.
    descriptorSetAllocateInfo.pSetLayouts        = &descriptorSetLayout;

    // allocate descriptor set.
    NVVK_CHECK(vkAllocateDescriptorSets(m_device, &descriptorSetAllocateInfo, &descriptorSet));

    /*
        Next, we need to connect our actual storage buffer with the descrptor. 
        We use vkUpdateDescriptorSets() to update the descriptor set.
        */

    // Specify the buffer to bind to the descriptor.
    VkDescriptorBufferInfo descriptorBufferInfo = {};
    descriptorBufferInfo.buffer                 = buffer;
    descriptorBufferInfo.offset                 = 0;
    descriptorBufferInfo.range                  = bufferSize;

    VkWriteDescriptorSet writeDescriptorSet = {};
    writeDescriptorSet.sType                = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet.dstSet               = descriptorSet;  // write to this descriptor set.
    writeDescriptorSet.dstBinding           = 0;  // write to the first, and only binding.
    writeDescriptorSet.descriptorCount      = 1;  // update a single descriptor.
    writeDescriptorSet.descriptorType       = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;  // storage buffer.
    writeDescriptorSet.pBufferInfo          = &descriptorBufferInfo;

    // perform the update of the descriptor set.
    vkUpdateDescriptorSets(m_device, 1, &writeDescriptorSet, 0, NULL);

    m_descPool  = descriptorPool;
    m_descSetLayout = descriptorSetLayout;
    m_descSet       = descriptorSet;
  }

  // Read file into array of bytes, and cast to uint32_t*, then return.
  // The data has been padded, so that it fits into an array uint32_t.
  uint32_t* readFile(uint32_t& length, const char* filename)
  {

    FILE* fp = fopen(filename, "rb");
    if(fp == NULL)
    {
      printf("Could not find or open file: %s\n", filename);
    }

    // get file size.
    fseek(fp, 0, SEEK_END);
    long filesize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    long filesizepadded = long(ceil(filesize / 4.0)) * 4;

    // read file contents.
    char* str = new char[filesizepadded];
    fread(str, filesize, sizeof(char), fp);
    fclose(fp);

    // data padding.
    for(int i = filesize; i < filesizepadded; i++)
    {
      str[i] = 0;
    }

    length = filesizepadded;
    return (uint32_t*)str;
  }

  void createComputePipeline();


  void createCommandBuffer(uint32_t& _queueFamilyIndex)
  {
    VkDescriptorSet    descriptorSet  = m_descSet;
    VkPipeline pipeline = m_graphicsPipeline;
    VkPipelineLayout pipelineLayout = m_pipelineLayout;
    VkCommandPool    commandPool    = m_cmdPool;
    VkCommandBuffer  commandBuffer;

    /*
        Now allocate a command buffer from the command pool. 
        */
    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool =
        commandPool;  // specify the command pool to allocate from.
    // if the command buffer is primary, it can be directly submitted to queues.
    // A secondary buffer has to be called from some primary command buffer, and cannot be directly
    // submitted to a queue. To keep things simple, we use a primary command buffer.
    commandBufferAllocateInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;  // allocate a single command buffer.
    NVVK_CHECK(vkAllocateCommandBuffers(m_device, &commandBufferAllocateInfo,
                                             &commandBuffer));  // allocate command buffer.

    /*
        Now we shall start recording commands into the newly allocated command buffer. 
        */
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags =
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;  // the buffer is only submitted and used once in this application.
    NVVK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));  // start recording commands.

    /*
        We need to bind a pipeline, AND a descriptor set before we dispatch.
        The validation layer will NOT give warnings if you forget these, so be very careful not to forget them.
        */
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
                            &descriptorSet, 0, NULL);

    /*
        Calling vkCmdDispatch basically starts the compute pipeline, and executes the compute shader.
        The number of workgroups is specified in the arguments.
        If you are already familiar with compute shaders from OpenGL, this should be nothing new to you.
        */
    vkCmdDispatch(commandBuffer, (uint32_t)ceil(WIDTH / float(WORKGROUP_SIZE)),
                  (uint32_t)ceil(HEIGHT / float(WORKGROUP_SIZE)), 1);

    NVVK_CHECK(vkEndCommandBuffer(commandBuffer));  // end recording commands.

    m_graphicsPipeline = pipeline;
    m_pipelineLayout   = pipelineLayout;
    m_cmdPool          = commandPool;
    m_commandBuffers.push_back(commandBuffer);
    m_descSet = descriptorSet;
  }

  void runCommandBuffer()
  {
    VkCommandBuffer commandBuffer = m_commandBuffers[0];
    /*
        Now we shall finally submit the recorded command buffer to a queue.
        */

    VkSubmitInfo submitInfo       = {};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;               // submit a single command buffer
    submitInfo.pCommandBuffers    = &commandBuffer;  // the command buffer to submit.

    /*
          We create a fence.
        */
    VkFence           fence;
    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags             = 0;
    NVVK_CHECK(vkCreateFence(m_device, &fenceCreateInfo, NULL, &fence));

    /*
        We submit the command buffer on the queue, at the same time giving a fence.
        */
    NVVK_CHECK(vkQueueSubmit(m_queue, 1, &submitInfo, fence));
    /*
        The command will not have finished executing until the fence is signalled.
        So we wait here.
        We will directly after this read our buffer from the GPU,
        and we will not be sure that the command has finished executing unless we wait for the fence.
        Hence, we use a fence here.
        */
    NVVK_CHECK(vkWaitForFences(m_device, 1, &fence, VK_TRUE, 100000000000));

    vkDestroyFence(m_device, fence, NULL);

    m_commandBuffers[0] = commandBuffer;
  }
};
