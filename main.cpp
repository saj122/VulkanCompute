#include "vulkan/vulkan.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <optional>
#include <set>

const std::vector<const char*> validationLayers = 
{
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> extensions = 
{
    VK_EXT_DEBUG_UTILS_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

VkResult createDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) 
{
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) 
    {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } 
    else 
    {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void destroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) 
{
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) 
    {
        func(instance, debugMessenger, pAllocator);
    }
}

struct QueueFamilyIndices 
{
    std::optional<uint32_t> computeFamily;

    bool isComplete() 
    {
        return computeFamily.has_value();
    }
};

class VulkanCompute 
{
    public:
        void run()
        {
            initVulkan();
            cleanUp();
        }

    private:
        void initVulkan()
        {
            createInstance();
            setupDebugMessenger();
            pickPhysicalDevice();
            createLogicalDevice();
            createCommandPoolAndFence();
            loadAsset();
            prepareTextureTarget();
            createComputePipeline();
            buildComputeCommandBuffer();
        }

        void cleanUp()
        {

            vkDestroyPipeline(_device, _computePipeline, nullptr);

            vkDestroyFence(_device, _fence, nullptr);
            vkFreeCommandBuffers(_device, _computeCommandPool, 1, &_computeCommandBuffer);

            vkDestroyDescriptorSetLayout(_device, _computeDescriptorSetLayout, nullptr);

            vkDestroyCommandPool(_device, _computeCommandPool, nullptr);
            
            vkDestroyPipelineLayout(_device, _pipelineLayout, nullptr);

            vkDestroyPipelineCache(_device, _pipelineCache, nullptr);

            vkDestroyDevice(_device, nullptr);

            if(enableValidationLayers)
            {
                destroyDebugUtilsMessengerEXT(_instance, _debugMessenger, nullptr);
            }
            vkDestroyInstance(_instance, nullptr);
        }

        void createInstance()
        {
            if (enableValidationLayers && !checkValidationLayerSupport()) 
            {
                throw std::runtime_error("Validation layers requested, but not available.");
            }

            VkApplicationInfo app_info;
            app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
            app_info.pApplicationName = "VulkanCompute";
            app_info.applicationVersion = VK_MAKE_VERSION(1,0,0);
            app_info.pEngineName = "No Engine";
            app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
            app_info.apiVersion = VK_API_VERSION_1_0;
            app_info.pNext = nullptr;

            VkInstanceCreateInfo create_info{};
            create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
            create_info.pApplicationInfo = &app_info;
            if (enableValidationLayers) 
            {
                create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
                create_info.ppEnabledExtensionNames = extensions.data();
            }
            else
            {
                create_info.enabledExtensionCount = 0;
            }
            
            VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
            if (enableValidationLayers) 
            {
                create_info.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
                create_info.ppEnabledLayerNames = validationLayers.data();

                populateDebugMessengerCreateInfo(debugCreateInfo);
                create_info.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
            } 
            else 
            {
                create_info.enabledLayerCount = 0;

                create_info.pNext = nullptr;
            }

            if (vkCreateInstance(&create_info, nullptr, &_instance) != VK_SUCCESS) 
            {
                throw std::runtime_error("Failed to create instance.");
            }   
        }

        void buildComputeCommandBuffer()
        {
            vkQueueWaitIdle(_computeQueue);

            VkCommandBufferBeginInfo cmdBufferBeginInfo{};
            cmdBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

            if (vkBeginCommandBuffer(_computeCommandBuffer, &cmdBufferBeginInfo) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to record command buffer.");
            }

            vkCmdBindPipeline(_computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _computePipeline);
            vkCmdBindDescriptorSets(_computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipelineLayout, 0, 1, &_computeDescriptorSet, 0, 0);

            vkCmdDispatch(_computeCommandBuffer, 800 / 16, 771 / 16, 1);

            if (vkEndCommandBuffer(_computeCommandBuffer) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to record command buffer.");
            }
        }

        void createCommandPoolAndFence()
        {
            QueueFamilyIndices indices = findQueueFamilies(_physicalDevice);

            VkCommandPoolCreateInfo cmdPoolInfo = {};
            cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            cmdPoolInfo.queueFamilyIndex = indices.computeFamily.value();
            cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            if (vkCreateCommandPool(_device, &cmdPoolInfo, nullptr, &_computeCommandPool) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create compute command pool.");
            }

            VkCommandBufferAllocateInfo cmdBufferAllocateInfo{};
            cmdBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cmdBufferAllocateInfo.commandPool = _computeCommandPool;
            cmdBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmdBufferAllocateInfo.commandBufferCount = 1;

            if (vkAllocateCommandBuffers(_device, &cmdBufferAllocateInfo, &_computeCommandBuffer) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to allocate command buffer.");
            }

            VkFenceCreateInfo fenceCreateInfo{};
            fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

            if (vkCreateFence(_device, &fenceCreateInfo, nullptr, &_fence) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create fence.");
            }
        }

        uint32_t getMemoryType(VkPhysicalDeviceMemoryProperties memoryProperties, uint32_t typeBits, VkMemoryPropertyFlags properties, VkBool32* memTypeFound) const
        {
            for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++)
            {
                if ((typeBits & 1) == 1)
                {
                    if ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
                    {
                        if (memTypeFound)
                        {
                            *memTypeFound = true;
                        }
                        return i;
                    }
                }
                typeBits >>= 1;
            }

            if (memTypeFound)
            {
                *memTypeFound = false;
                return 0;
            }
            else
            {
                throw std::runtime_error("Could not find a matching memory type");
            }
        }

        void loadAsset()
        {
            
            int width, height, channels;
            _image = stbi_load("../data/vulkan.png", &width, &height, &channels, 4);
            if (_image == NULL)
            {
                throw std::runtime_error("Error loading image.");
            }

            VkFormatProperties formatProperties;
            vkGetPhysicalDeviceFormatProperties(_physicalDevice, VK_FORMAT_R8G8B8A8_UNORM, &formatProperties);

            // Only use linear tiling if requested (and supported by the device)
            // Support for linear tiling is mostly limited, so prefer to use
            // optimal tiling instead
            // On most implementations linear tiling will only support a very
            // limited amount of formats and features (mip maps, cubemaps, arrays, etc.)
            VkBool32 useStaging = VK_FALSE;

            VkMemoryAllocateInfo memAllocInfo{};
            memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            VkMemoryRequirements memReqs;

            vkGetImageMemoryRequirements(_device, _view_image, &memReqs);

            // Use a separate command buffer for texture loading
            VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
            commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            commandBufferAllocateInfo.commandPool = _computeCommandPool;
            commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            commandBufferAllocateInfo.commandBufferCount = 1;
            if (vkAllocateCommandBuffers(_device, &commandBufferAllocateInfo, &_computeCommandBuffer) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to allocate command buffer.");
            }

            VkCommandBufferBeginInfo cmdBufferBeginInfo{};
            cmdBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            if (vkBeginCommandBuffer(_computeCommandBuffer, &cmdBufferBeginInfo) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to record buffer.");
            }
            
            // Create a host-visible staging buffer that contains the raw image data
            VkBuffer stagingBuffer;
            VkDeviceMemory stagingMemory;

            VkBufferCreateInfo bufCreateInfo{};
            bufCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufCreateInfo.size = 800 * 771 * 3;
            bufCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            bufCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            if (vkCreateBuffer(_device, &bufCreateInfo, nullptr, &stagingBuffer) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create staging buffer.");
            }

            vkGetBufferMemoryRequirements(_device, stagingBuffer, &memReqs);

            memAllocInfo.allocationSize = memReqs.size;

            memAllocInfo.memoryTypeIndex = getMemoryType(pdmp,);
            
            if (vkAllocateMemory(_device, &memAllocInfo, nullptr, &_deviceMemory) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to allocate memory.");
            }

            if (vkBindImageMemory(_device, _view_image, _deviceMemory, 0) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to bind to image memory.");
            }

            uint8_t* data;
            if (vkMapMemory(_device, stagingMemory, 0, memReqs.size, 0, (void**)&data) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to map memory.");
            }
            memcpy(data, _image, width*height*4);
            vkUnmapMemory(_device, stagingMemory);

            std::vector<VkBufferImageCopy> bufferCopyRegions;

            VkBufferImageCopy bufferCopyRegion = {};
            bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            bufferCopyRegion.imageSubresource.mipLevel = 1;
            bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
            bufferCopyRegion.imageSubresource.layerCount = 1;
            bufferCopyRegion.imageExtent.width = std::max(1u, width >> i);
            bufferCopyRegion.imageExtent.height = std::max(1u, height >> i);
            bufferCopyRegion.imageExtent.depth = 1;
            bufferCopyRegion.bufferOffset = 4;

            bufferCopyRegions.push_back(bufferCopyRegion);

            VkImageCreateInfo imageCreateInfo{};
            imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
            imageCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
            imageCreateInfo.mipLevels = 1;
            imageCreateInfo.arrayLayers = 1;
            imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            imageCreateInfo.extent = { width, height, 1 };
            imageCreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
            // Ensure that the TRANSFER_DST bit is set for staging
            if (!(imageCreateInfo.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT))
            {
                imageCreateInfo.usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            }
            if (vkCreateImage(_device, &imageCreateInfo, nullptr, &_view_image) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create image.");
            }

            vkGetImageMemoryRequirements(_device, _view_image, &memReqs);

            memAllocInfo.allocationSize = memReqs.size;

            VkPhysicalDeviceMemoryProperties pdmp;
            vkGetPhysicalDeviceMemoryProperties(_physicalDevice, &pdmp);

            uint32_t typeBits = memReqs.memoryTypeBits;
            VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            VkBool32* memTypeFound = nullptr;

            memAllocInfo.memoryTypeIndex = getMemoryType(pdmp, typeBits, properties, memTypeFound);

            if (vkAllocateMemory(_device, &memAllocInfo, nullptr, &_deviceMemory) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to allocate memory.");
            }

            if (vkBindImageMemory(_device, _view_image, _deviceMemory, 0) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to bind image memory.");
            }

            VkImageSubresourceRange subresourceRange = {};
            subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            subresourceRange.baseMipLevel = 0; 
            subresourceRange.levelCount = 1;
            subresourceRange.layerCount = 1;

            VkImageMemoryBarrier imageMemoryBarrier{};
            imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            imageMemoryBarrier.image = _view_image;
            imageMemoryBarrier.subresourceRange = subresourceRange;
            imageMemoryBarrier.srcAccessMask = 0;
            imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
              
            vkCmdPipelineBarrier(
                _computeCommandBuffer,
                65536U,
                65536U,
                0,
                0, nullptr,
                0, nullptr,
                1, &imageMemoryBarrier);

            vkCmdCopyBufferToImage(_computeCommandBuffer, stagingBuffer, _view_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, static_cast<uint32_t>(bufferCopyRegions.size()), bufferCopyRegions.data());

            _imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkImageMemoryBarrier imageMemoryBarrier{};
            imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            imageMemoryBarrier.image = _view_image;
            imageMemoryBarrier.subresourceRange = subresourceRange;
            imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            if (_computeCommandBuffer == VK_NULL_HANDLE)
            {
                return;
            }

            if (vkEndCommandBuffer(_computeCommandBuffer) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to record command buffer.");
            }

            VkSubmitInfo submitInfo = vks::initializers::submitInfo();
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffer;
            // Create fence to ensure that the command buffer has finished executing
            VkFenceCreateInfo fenceInfo = vks::initializers::fenceCreateInfo(VK_FLAGS_NONE);
            VkFence fence;
            VK_CHECK_RESULT(vkCreateFence(logicalDevice, &fenceInfo, nullptr, &fence));
            // Submit to the queue
            VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence));
            // Wait for the fence to signal that command buffer has finished executing
            VK_CHECK_RESULT(vkWaitForFences(logicalDevice, 1, &fence, VK_TRUE, DEFAULT_FENCE_TIMEOUT));
            vkDestroyFence(logicalDevice, fence, nullptr);
            if (free)
            {
                vkFreeCommandBuffers(logicalDevice, pool, 1, &commandBuffer);
            }
        }

        void prepareTextureTarget()
        {
            VkFormatProperties formatProperties;

            vkGetPhysicalDeviceFormatProperties(_physicalDevice, VK_FORMAT_R8G8B8A8_UNORM, &formatProperties);

            VkImageCreateInfo imageCreateInfo{};
            imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
            imageCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
            imageCreateInfo.extent = { 800, 771, 1 };
            imageCreateInfo.mipLevels = 1;
            imageCreateInfo.arrayLayers = 1;
            imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageCreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
            imageCreateInfo.flags = 0;

            QueueFamilyIndices indices = findQueueFamilies(_physicalDevice);

            std::vector<uint32_t> queueFamilyIndices;
            queueFamilyIndices.push_back(indices.computeFamily.value());

            imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            imageCreateInfo.queueFamilyIndexCount = 1;
            imageCreateInfo.pQueueFamilyIndices = queueFamilyIndices.data();

            VkMemoryAllocateInfo memAllocInfo{};
            memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            VkMemoryRequirements memReqs;

            if (vkCreateImage(_device, &imageCreateInfo, nullptr, &_target_image) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create image.");
            }

            vkGetImageMemoryRequirements(_device, _target_image, &memReqs);
            memAllocInfo.allocationSize = memReqs.size;

            VkPhysicalDeviceMemoryProperties pdmp;
            vkGetPhysicalDeviceMemoryProperties(_physicalDevice, &pdmp);

            uint32_t typeBits = memReqs.memoryTypeBits;
            VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            VkBool32* memTypeFound = nullptr;

            memAllocInfo.memoryTypeIndex = getMemoryType(pdmp, typeBits, properties, memTypeFound);

            if (vkAllocateMemory(_device, &memAllocInfo, nullptr, &_deviceMemory) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to allocate device memory.");
            }

            if (vkBindImageMemory(_device, _target_image, _deviceMemory, 0) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to bind image to memory.");
            }

            VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
            commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            commandBufferAllocateInfo.commandPool = _computeCommandPool;
            commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            commandBufferAllocateInfo.commandBufferCount = 1;

            if (vkAllocateCommandBuffers(_device, &commandBufferAllocateInfo, &_computeCommandBuffer) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to allocate command buffers.");
            }

            VkCommandBufferBeginInfo cmdBufferBeginInfo{};
            cmdBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            if (vkBeginCommandBuffer(_computeCommandBuffer, &cmdBufferBeginInfo) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to record command.");
            }

            _imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkImageSubresourceRange subresourceRange = {};
            subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            subresourceRange.baseMipLevel = 0;
            subresourceRange.levelCount = 1;
            subresourceRange.layerCount = 1;

            VkImageMemoryBarrier imageMemoryBarrier{};
            imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            imageMemoryBarrier.newLayout = _imageLayout;
            imageMemoryBarrier.image = _target_image;
            imageMemoryBarrier.subresourceRange = subresourceRange;
            imageMemoryBarrier.srcAccessMask = 0;

            vkCmdPipelineBarrier(
                _computeCommandBuffer,
                65536U,
                65536U,
                0,
                0, nullptr,
                0, nullptr,
                1, &imageMemoryBarrier);

            if (_computeCommandBuffer == VK_NULL_HANDLE)
            {
                return;
            }

            if (vkEndCommandBuffer(_computeCommandBuffer) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to record command buffer.");
            }

            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &_computeCommandBuffer;

            VkFenceCreateInfo fenceCreateInfo{};
            fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceCreateInfo.flags = 0;
            VkFence fence;
            if (vkCreateFence(_device, &fenceCreateInfo, nullptr, &fence) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create fence.");
            }
            if (vkQueueSubmit(_computeQueue, 1, &submitInfo, fence) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed queue submission.");
            }
            if (vkWaitForFences(_device, 1, &fence, VK_TRUE, 100000000000) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed waiting for fence.");
            }
            vkDestroyFence(_device, fence, nullptr);

            vkFreeCommandBuffers(_device, _computeCommandPool, 1, &_computeCommandBuffer);

            VkSamplerCreateInfo sampler{};
            sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sampler.maxAnisotropy = 1.0f;
            sampler.magFilter = VK_FILTER_LINEAR;
            sampler.minFilter = VK_FILTER_LINEAR;
            sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            sampler.addressModeV = sampler.addressModeU;
            sampler.addressModeW = sampler.addressModeU;
            sampler.mipLodBias = 0.0f;
            sampler.maxAnisotropy = 1.0f;
            sampler.compareOp = VK_COMPARE_OP_NEVER;
            sampler.minLod = 0.0f;
            sampler.maxLod = 0;
            sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
            if (vkCreateSampler(_device, &sampler, nullptr, &_sampler) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create sampler.");
            }

            VkImageViewCreateInfo view{};
            view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            view.image = VK_NULL_HANDLE;
            view.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view.format = VK_FORMAT_R8G8B8A8_UNORM;
            view.components = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A };
            view.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            view.image = _target_image;
            if (vkCreateImageView(_device, &view, nullptr, &_targetView) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create image view.");
            }

            _targetDescriptor.imageLayout = _imageLayout;
            _targetDescriptor.imageView = _targetView;
            _targetDescriptor.sampler = _sampler;
        }

        void createComputePipeline()
        {
            auto computeShaderCode = readFile("../shaders/edgedetect.comp.spv");

            VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

            VkPipelineShaderStageCreateInfo compShaderStageInfo{};
            compShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            compShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            compShaderStageInfo.module = computeShaderModule;
            compShaderStageInfo.pName = "main";

            VkPipelineShaderStageCreateInfo shaderStages[] = { compShaderStageInfo };

            VkDescriptorSetLayoutBinding setLayoutBinding {};
            setLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            setLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            setLayoutBinding.binding = 0;
            setLayoutBinding.descriptorCount = 1;
            
            VkDescriptorSetLayoutBinding setLayoutBinding2 {};
            setLayoutBinding2.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            setLayoutBinding2.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            setLayoutBinding2.binding = 1;
            setLayoutBinding2.descriptorCount = 1;

			std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings;
            setLayoutBindings.reserve(2);

            setLayoutBindings.push_back(setLayoutBinding);
            setLayoutBindings.push_back(setLayoutBinding2);

            VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo{};
			descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
			descriptorSetLayoutCreateInfo.pBindings = setLayoutBindings.data();
			descriptorSetLayoutCreateInfo.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
		    if(vkCreateDescriptorSetLayout(_device,	&descriptorSetLayoutCreateInfo, nullptr, &_computeDescriptorSetLayout) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create descriptor set layout.");
            }

			VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
            pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutInfo.setLayoutCount = 0;
            pipelineLayoutInfo.pSetLayouts = 0;
            pipelineLayoutInfo.pushConstantRangeCount = 0;
            
            if (vkCreatePipelineLayout(_device, &pipelineLayoutInfo, nullptr, &_pipelineLayout) != VK_SUCCESS) 
            {
                throw std::runtime_error("Failed to create pipeline layout.");
            }

            VkDescriptorPoolSize descriptorPoolSize{};
            descriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorPoolSize.descriptorCount = 2;

            VkDescriptorPoolCreateInfo descriptorPoolInfo{};
            descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            descriptorPoolInfo.poolSizeCount = 1;
            descriptorPoolInfo.pPoolSizes = &descriptorPoolSize;
            descriptorPoolInfo.maxSets = 3;

            if (vkCreateDescriptorPool(_device, &descriptorPoolInfo, nullptr, &_computeDescriptorPool) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create descriptor pool.");
            }

            VkDescriptorSetAllocateInfo descriptorSetAllocateInfo {};
			descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			descriptorSetAllocateInfo.descriptorPool = _computeDescriptorPool;
			descriptorSetAllocateInfo.pSetLayouts = &_computeDescriptorSetLayout;
			descriptorSetAllocateInfo.descriptorSetCount = 1;

            if(vkAllocateDescriptorSets(_device, &descriptorSetAllocateInfo, &_computeDescriptorSet) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to allocate descriptor set.");
            }

            VkWriteDescriptorSet writeDescriptorSet {};
			writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			writeDescriptorSet.dstSet = _computeDescriptorSet;
			writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			writeDescriptorSet.dstBinding = 0;
			writeDescriptorSet.pImageInfo = &_imageDescriptor;
			writeDescriptorSet.descriptorCount = 1;

            VkWriteDescriptorSet writeDescriptorSet2 {};
			writeDescriptorSet2.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			writeDescriptorSet2.dstSet = _computeDescriptorSet;
			writeDescriptorSet2.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			writeDescriptorSet2.dstBinding = 1;
			writeDescriptorSet2.pImageInfo = &_targetDescriptor;
			writeDescriptorSet2.descriptorCount = 1;

            std::vector <VkWriteDescriptorSet> computeWriteDescriptorSets;
            computeWriteDescriptorSets.reserve(2);

            computeWriteDescriptorSets.push_back(writeDescriptorSet);
            computeWriteDescriptorSets.push_back(writeDescriptorSet2);

            vkUpdateDescriptorSets(_device, static_cast<uint32_t>(computeWriteDescriptorSets.size()), computeWriteDescriptorSets.data(), 0, NULL);

            VkComputePipelineCreateInfo pipelineInfo {};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipelineInfo.layout = _pipelineLayout;
            pipelineInfo.flags = 0;

            VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
            pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
            if(vkCreatePipelineCache(_device, &pipelineCacheCreateInfo, nullptr, &_pipelineCache) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create pipeline cache.");
            }

            pipelineInfo.stage = compShaderStageInfo;
            if(vkCreateComputePipelines(_device, _pipelineCache, 1, &pipelineInfo, nullptr, &_computePipeline) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create compute pipeline.");
            }

            vkDestroyShaderModule(_device, computeShaderModule, nullptr);
        }

        VkShaderModule createShaderModule(const std::vector<char>& code) 
        {
            VkShaderModuleCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            createInfo.codeSize = code.size();
            createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

            VkShaderModule shaderModule;
            if (vkCreateShaderModule(_device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) 
            {
                throw std::runtime_error("Failed to create shader module.");
            }

            return shaderModule;
        }
        
        void setupDebugMessenger() 
        {
            if (!enableValidationLayers) return;

            VkDebugUtilsMessengerCreateInfoEXT createInfo;
            populateDebugMessengerCreateInfo(createInfo);

            if (createDebugUtilsMessengerEXT(_instance, &createInfo, nullptr, &_debugMessenger) != VK_SUCCESS) 
            {
                throw std::runtime_error("Failed to set up debug messenger.");
            }
        }

        void pickPhysicalDevice() 
        {
            uint32_t deviceCount = 0;
            vkEnumeratePhysicalDevices(_instance, &deviceCount, nullptr);

            if (deviceCount == 0) 
            {
                throw std::runtime_error("Failed to find GPUs with Vulkan support.");
            }

            std::vector<VkPhysicalDevice> devices(deviceCount);
            vkEnumeratePhysicalDevices(_instance, &deviceCount, devices.data());

            for (const auto& device : devices) 
            {
                if (isDeviceSuitable(device)) 
                {
                    _physicalDevice = device;
                    break;
                }
            }

            if (_physicalDevice == VK_NULL_HANDLE) 
            {
                throw std::runtime_error("Failed to find a suitable GPU.");
            }
        }

        void createLogicalDevice() 
        {
            QueueFamilyIndices indices = findQueueFamilies(_physicalDevice);

            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = indices.computeFamily.value();
            queueCreateInfo.queueCount = 1;

            float queuePriority = 1.0f;
            queueCreateInfo.pQueuePriorities = &queuePriority;

            VkPhysicalDeviceFeatures deviceFeatures{};

            VkDeviceCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

            createInfo.pQueueCreateInfos = &queueCreateInfo;
            createInfo.queueCreateInfoCount = 1;

            createInfo.pEnabledFeatures = &deviceFeatures;

            createInfo.enabledExtensionCount = 0;

            if (enableValidationLayers) 
            {
                createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
                createInfo.ppEnabledLayerNames = validationLayers.data();
            } 
            else 
            {
                createInfo.enabledLayerCount = 0;
            }

            if (vkCreateDevice(_physicalDevice, &createInfo, nullptr, &_device) != VK_SUCCESS) 
            {
                throw std::runtime_error("Failed to create logical device.");
            }

            vkGetDeviceQueue(_device, indices.computeFamily.value(), 0, &_computeQueue);
        }

        QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) 
        {
            QueueFamilyIndices indices;

            uint32_t queueFamilyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

            std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

            int i = 0;
            for (const auto& queueFamily : queueFamilies) 
            {
                if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) 
                {
                    indices.computeFamily = i;
                }

                if (indices.isComplete()) 
                {
                    break;
                }

                i++;
            }

            return indices;
        }

        bool isDeviceSuitable(VkPhysicalDevice device) 
        {
            QueueFamilyIndices indices = findQueueFamilies(device);

            return indices.isComplete();
        }

        void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) 
        {
            createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            createInfo.pfnUserCallback = debugCallback;
        }

        bool checkValidationLayerSupport() 
        {
            uint32_t layerCount;
            vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

            std::vector<VkLayerProperties> availableLayers(layerCount);
            vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

            for (const char* layerName : validationLayers) 
            {
                bool layerFound = false;

                for (const auto& layerProperties : availableLayers) 
                {
                    if (strcmp(layerName, layerProperties.layerName) == 0) 
                    {
                        layerFound = true;
                        break;
                    }
                }

                if (!layerFound) 
                {
                    return false;
                }
            }

            return true;
        }

        static std::vector<char> readFile(const std::string& filename) 
        {
            std::ifstream file(filename, std::ios::ate | std::ios::binary);

            if (!file.is_open()) 
            {
                throw std::runtime_error("Failed to open file.");
            }

            size_t fileSize = (size_t) file.tellg();
            std::vector<char> buffer(fileSize);

            file.seekg(0);
            file.read(buffer.data(), fileSize);

            file.close();

            return buffer;
        }

        static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) 
        {
            std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

            return VK_FALSE;
        }

    private:
        VkInstance _instance;    
        VkDebugUtilsMessengerEXT _debugMessenger;
        VkPhysicalDevice _physicalDevice = VK_NULL_HANDLE;
        VkDevice _device;
        VkQueue _computeQueue;
        VkPipelineLayout _pipelineLayout;
        VkDescriptorSetLayout _computeDescriptorSetLayout;
        VkPipelineCache _pipelineCache;
        VkCommandPool _computeCommandPool;
        VkDescriptorPool _computeDescriptorPool;
        VkDescriptorSet _computeDescriptorSet;
        VkDescriptorImageInfo _imageDescriptor;
        VkDescriptorImageInfo _targetDescriptor;
        VkPipeline _computePipeline;
        VkCommandBuffer _computeCommandBuffer;
        VkFence _fence;
        VkImage _view_image;
        VkImage _target_image;
        VkDeviceMemory _deviceMemory;
        VkImageLayout _imageLayout;
        VkSampler _sampler;
        VkImageView _targetView;
        VkImageView _imageView;

        std::vector<std::string> _shaderNames;

        unsigned char* _image;
};

int main(int argc, char** argv)
{
    VulkanCompute vc;

    try
    {
        vc.run();
        system("pause");
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}