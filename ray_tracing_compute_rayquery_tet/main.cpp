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

// ImGui - standalone example application for Glfw + Vulkan, using programmable
// pipeline If you are new to ImGui, see examples/README.txt and documentation
// at the top of imgui.cpp.

#include <array>
#include <iostream>

#include <vulkan/vulkan.hpp>
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#include "imgui.h"
#include "imgui_impl_glfw.h"

#include "hello_vulkan.h"
#include "nvh/cameramanipulator.hpp"
#include "nvh/fileoperations.hpp"
#include "nvpsystem.hpp"
#include "nvvk/appbase_vkpp.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/context_vk.hpp"


//////////////////////////////////////////////////////////////////////////
#define UNUSED(x) (void)(x)
//////////////////////////////////////////////////////////////////////////

// Default search path for shaders
std::vector<std::string> defaultSearchPaths;

// GLFW Callback functions
static void onErrorCallback(int error, const char* description)
{
  fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

// Extra UI
void renderUI(HelloVulkan& helloVk)
{
  static int item = 1;
  if(ImGui::Combo("Up Vector", &item, "X\0Y\0Z\0\0"))
  {
    nvmath::vec3f pos, eye, up;
    CameraManip.getLookat(pos, eye, up);
    up = nvmath::vec3f(item == 0, item == 1, item == 2);
    CameraManip.setLookat(pos, eye, up);
  }
  ImGui::SliderFloat3("Light Position", &helloVk.m_pushConstant.lightPosition.x, -20.f, 20.f);
  ImGui::SliderFloat("Light Intensity", &helloVk.m_pushConstant.lightIntensity, 0.f, 100.f);
  ImGui::RadioButton("Point", &helloVk.m_pushConstant.lightType, 0);
  ImGui::SameLine();
  ImGui::RadioButton("Infinite", &helloVk.m_pushConstant.lightType, 1);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//static int const SAMPLE_WIDTH  = 3200;
//static int const SAMPLE_HEIGHT = 2400;

//--------------------------------------------------------------------------------------------------
// Application Entry
//
int main(int argc, char** argv)
{
  UNUSED(argc);

  // Setup GLFW window
  glfwSetErrorCallback(onErrorCallback);
  if(!glfwInit())
  {
    return 1;
  }
  //glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  //GLFWwindow* window = glfwCreateWindow(SAMPLE_WIDTH, SAMPLE_HEIGHT,
  //                                      "NVIDIA Vulkan Raytracing Tutorial", nullptr, nullptr);

  //// Setup camera
  //CameraManip.setWindowSize(SAMPLE_WIDTH, SAMPLE_HEIGHT);
  //CameraManip.setLookat(nvmath::vec3f(5, 4, -4), nvmath::vec3f(0, 1, 0), nvmath::vec3f(0, 1, 0));

  //// Setup Vulkan
  //if(!glfwVulkanSupported())
  //{
  //  printf("GLFW: Vulkan Not Supported\n");
  //  return 1;
  //}

  //// setup some basic things for the sample, logging file for example
  //NVPSystem system(argv[0], PROJECT_NAME);

  // Search path for shaders and other media
  defaultSearchPaths = {
      PROJECT_ABSDIRECTORY,
      PROJECT_ABSDIRECTORY "../",
      NVPSystem::exePath() + std::string(PROJECT_RELDIRECTORY),
      NVPSystem::exePath() + std::string(PROJECT_RELDIRECTORY) + std::string("../"),
  };

  // Enabling the extension feature
  vk::PhysicalDeviceRayTracingFeaturesKHR raytracingFeature;

  // Requesting Vulkan extensions and layers
  nvvk::ContextCreateInfo contextInfo(true);
  contextInfo.setVersion(1, 2);
  contextInfo.addInstanceLayer("VK_LAYER_LUNARG_monitor", true);
  contextInfo.addInstanceExtension(VK_KHR_SURFACE_EXTENSION_NAME);
#ifdef WIN32
  contextInfo.addInstanceExtension(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#else
  contextInfo.addInstanceExtension(VK_KHR_XLIB_SURFACE_EXTENSION_NAME);
  contextInfo.addInstanceExtension(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
#endif
  contextInfo.addInstanceExtension(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
  // #VKRay: Activate the ray tracing extension
  contextInfo.addDeviceExtension(VK_KHR_RAY_TRACING_EXTENSION_NAME, false, &raytracingFeature);
  contextInfo.addDeviceExtension(VK_KHR_MAINTENANCE3_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);


  // Creating Vulkan base application
  nvvk::Context vkctx{};
  vkctx.initInstance(contextInfo);
  // Find all compatible devices
  auto compatibleDevices = vkctx.getCompatibleDevices(contextInfo);
  assert(!compatibleDevices.empty());
  // Use a compatible device
  vkctx.initDevice(compatibleDevices[0], contextInfo);


  // Create example
  HelloVulkan helloVk;

  //// Window need to be opened to get the surface on which to draw
  //const vk::SurfaceKHR surface = helloVk.getVkSurface(vkctx.m_instance, window);
  //vkctx.setGCTQueueWithPresent(surface);

  helloVk.setup(vkctx.m_instance, vkctx.m_device, vkctx.m_physicalDevice,
                vkctx.m_queueGCT.familyIndex);
  //helloVk.createSurface(surface, SAMPLE_WIDTH, SAMPLE_HEIGHT);

  //helloVk.m_bufferSize = SAMPLE_WIDTH * SAMPLE_HEIGHT * 4 * 4;
  helloVk.createComputeBuffer();
  helloVk.createUniformBuffer();

    // Creation of the example
  helloVk.loadModel(nvh::findFile("media/scenes/cube_multi.obj", defaultSearchPaths));
  helloVk.createDescriptorSetLayout();

  //raytracing
  helloVk.initRayTracing();
  helloVk.createBottomLevelAS();
  helloVk.createTopLevelAS();
  
  helloVk.updateDescriptorSet();

  helloVk.createComputePipeline();
  helloVk.createCommandBuffer(vkctx.m_queueGCT.familyIndex);

  helloVk.runCommandBuffer();
  helloVk.saveRenderedImage();


  nvmath::vec4f clearColor = nvmath::vec4f(1, 1, 1, 1.00f);

  // Cleanup
  helloVk.getDevice().waitIdle();
  helloVk.destroyResources();
  helloVk.destroy();

  vkctx.deinit();

  glfwTerminate();

  return 0;
}
