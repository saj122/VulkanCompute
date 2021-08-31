#include "vulkan/vulkan.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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
            vkDeviceWaitIdle(_device);
            cleanUp();
        }

    private:
        void initVulkan()
        {
            createInstance();
            setupDebugMessenger();
            pickPhysicalDevice();
            createLogicalDevice();
            createCommandPool();
            loadAsset();
            prepareTextureTarget();
            setupDescriptorPool();
            prepareCompute();
            computeShader();
        }

        void cleanUp()
        {
            vkDestroyShaderModule(_device, compute.shaderModule, nullptr);

            vkDestroyPipelineLayout(_device, compute.pipelineLayout, nullptr);
            vkDestroyPipeline(_device, compute.pipeline, nullptr);

            vkDestroyDescriptorSetLayout(_device, compute.descriptorSetLayout, nullptr);
            vkDestroyDescriptorPool(_device, compute.descriptorPool, nullptr);
            
            vkDestroyImageView(_device, compute.targetImageView, nullptr);
            vkDestroyImageView(_device, compute.texImageView, nullptr);

            vkDestroySampler(_device, compute.texSampler, nullptr);
            vkDestroySampler(_device, compute.targetSampler, nullptr);

            vkFreeMemory(_device, compute.stagingMemory, nullptr);

            vkDestroyImage(_device, compute.targetImage, nullptr);
            vkFreeMemory(_device, compute.targetDeviceMemory, nullptr);

            vkDestroyImage(_device, compute.texImage, nullptr);
            vkFreeMemory(_device, compute.texDeviceMemory, nullptr);

            vkFreeCommandBuffers(_device,compute.commandPool, 1, &compute.commandBuffer);
            vkDestroyCommandPool(_device, compute.commandPool, nullptr);

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

            uint32_t extCount = 0;
            std::vector<std::string> supportedInstanceExtensions;
            std::vector<const char*> enabledInstanceExtensions;
            std::vector<const char*> instanceExtensions = { VK_KHR_SURFACE_EXTENSION_NAME };
            vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
            if (extCount > 0)
            {
                std::vector<VkExtensionProperties> extensions(extCount);
                if (vkEnumerateInstanceExtensionProperties(nullptr, &extCount, &extensions.front()) == VK_SUCCESS)
                {
                    for (VkExtensionProperties extension : extensions)
                    {
                        supportedInstanceExtensions.push_back(extension.extensionName);
                    }
                }
            }

            if (enabledInstanceExtensions.size() > 0) 
            {
                for (const char * enabledExtension : enabledInstanceExtensions) 
                {
                    if (std::find(supportedInstanceExtensions.begin(), supportedInstanceExtensions.end(), enabledExtension) == supportedInstanceExtensions.end())
                    {
                        std::cerr << "Enabled instance extension \"" << enabledExtension << "\" is not present at instance level.\n";
                    }
                    instanceExtensions.push_back(enabledExtension);
                }
            }

            VkInstanceCreateInfo create_info{};
            create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
            create_info.pApplicationInfo = &app_info;
            if (enableValidationLayers) 
            {
                instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            }
            create_info.enabledExtensionCount = (uint32_t)instanceExtensions.size();
		    create_info.ppEnabledExtensionNames = instanceExtensions.data();
            
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

            vkGetDeviceQueue(_device, indices.computeFamily.value(), 0, &compute.queue);
        }

        void createCommandPool()
        {
            QueueFamilyIndices indices = findQueueFamilies(_physicalDevice);

            VkCommandPoolCreateInfo cmdPoolInfo = {};
            cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            cmdPoolInfo.queueFamilyIndex = indices.computeFamily.value();
            cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            if(vkCreateCommandPool(_device, &cmdPoolInfo, nullptr, &compute.commandPool) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create command pool.");
            }
        }

        void loadAsset()
        {
            int texChannels;
            stbi_uc* pixels = stbi_load("/home/stephen/git/VulkanCompute/data/vulkan.png", &_texWidth, &_texHeight, &texChannels, STBI_rgb_alpha);
            VkDeviceSize imageSize = _texWidth * _texHeight * 4;

            if (!pixels) 
            {
                throw std::runtime_error("Failed to load texture image.");
            }

            VkFormatProperties formatProperties;
    		vkGetPhysicalDeviceFormatProperties(_physicalDevice, VK_FORMAT_R8G8B8A8_UNORM, &formatProperties);

            VkMemoryAllocateInfo memAllocInfo {};
			memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            VkMemoryRequirements memReqs;

            VkCommandBufferAllocateInfo commandBufferAllocateInfo {};
			commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			commandBufferAllocateInfo.commandPool = compute.commandPool;
			commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			commandBufferAllocateInfo.commandBufferCount = 1;
            VkCommandBuffer cmdBuffer;
            if(vkAllocateCommandBuffers(_device, &commandBufferAllocateInfo, &cmdBuffer) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to allocate command buffers.");
            }

            VkCommandBufferBeginInfo cmdBufferBeginInfo {};
			cmdBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            if(vkBeginCommandBuffer(cmdBuffer, &cmdBufferBeginInfo) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to begin command buffer.");
            }

            VkBuffer stagingBuffer;

            VkBufferCreateInfo bufCreateInfo {};
			bufCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufCreateInfo.size = imageSize;
            bufCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            bufCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            if(vkCreateBuffer(_device, &bufCreateInfo, nullptr, &stagingBuffer) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create buffer.");
            }

            vkGetBufferMemoryRequirements(_device, stagingBuffer, &memReqs);

            memAllocInfo.allocationSize = memReqs.size;
            memAllocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);

            if(vkAllocateMemory(_device, &memAllocInfo, nullptr, &compute.stagingMemory) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to allocate memory.");
            }

            if(vkBindBufferMemory(_device, stagingBuffer, compute.stagingMemory, 0) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to bind buffer memory.");
            }

            uint8_t* data;
            if(vkMapMemory(_device, compute.stagingMemory, 0, memReqs.size, 0, (void **)&data) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to map memory.");
            }

            memcpy(data, pixels, static_cast<size_t>(imageSize));

            vkUnmapMemory(_device, compute.stagingMemory);

            VkImageCreateInfo imageCreateInfo {};
			imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
			imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
			imageCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
			imageCreateInfo.mipLevels = 1;
			imageCreateInfo.arrayLayers = 1;
			imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
			imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
			imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			imageCreateInfo.extent.width = static_cast<uint32_t>(_texWidth);
            imageCreateInfo.extent.height = static_cast<uint32_t>(_texHeight);
            imageCreateInfo.extent.depth = 1;
			imageCreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
			if (!(imageCreateInfo.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT))
			{
				imageCreateInfo.usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
			}
			if(vkCreateImage(_device, &imageCreateInfo, nullptr, &compute.texImage) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create texture image.");
            }

			vkGetImageMemoryRequirements(_device, compute.texImage, &memReqs);

			memAllocInfo.allocationSize = memReqs.size;
			memAllocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			if(vkAllocateMemory(_device, &memAllocInfo, nullptr, &compute.texDeviceMemory) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to allocate memory.");
            }

			if(vkBindImageMemory(_device, compute.texImage, compute.texDeviceMemory, 0) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to bind image memory.");
            }

			VkImageSubresourceRange subresourceRange = {};
			subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			subresourceRange.baseMipLevel = 0;
			subresourceRange.levelCount = 1;
			subresourceRange.layerCount = 1;

			VkBufferImageCopy bufferCopyRegion = {};
            bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            bufferCopyRegion.imageSubresource.mipLevel = 0;
            bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
            bufferCopyRegion.imageSubresource.layerCount = 1;
            bufferCopyRegion.imageExtent.width = static_cast<uint32_t>(_texWidth);
            bufferCopyRegion.imageExtent.height = static_cast<uint32_t>(_texHeight);
            bufferCopyRegion.imageExtent.depth = 1;
            bufferCopyRegion.bufferOffset = 0;

            VkImageMemoryBarrier imageMemoryBarrier {};
			imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			imageMemoryBarrier.image = compute.texImage;
			imageMemoryBarrier.subresourceRange = subresourceRange;
            imageMemoryBarrier.srcAccessMask = 0;
            imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            vkCmdPipelineBarrier(
				cmdBuffer,
				65536U,
				65536U,
				0,
				0, nullptr,
				0, nullptr,
				1, &imageMemoryBarrier);

			vkCmdCopyBufferToImage(
				cmdBuffer,
				stagingBuffer,
				compute.texImage,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1,
				&bufferCopyRegion
			);

            VkImageMemoryBarrier imageMemoryReadBarrier {};
			imageMemoryReadBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			imageMemoryReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageMemoryReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageMemoryReadBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			imageMemoryReadBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
			imageMemoryReadBarrier.image = compute.texImage;
			imageMemoryReadBarrier.subresourceRange = subresourceRange;
            imageMemoryReadBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            if (imageMemoryReadBarrier.srcAccessMask == 0)
            {
                imageMemoryReadBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            }
            imageMemoryReadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(
				cmdBuffer,
				65536U,
				65536U,
				0,
				0, nullptr,
				0, nullptr,
				1, &imageMemoryReadBarrier);

			if (cmdBuffer == VK_NULL_HANDLE)
            {
                return;
            }

            if(vkEndCommandBuffer(cmdBuffer) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to end command buffer.");
            }

            VkSubmitInfo submitInfo {};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmdBuffer;

            VkFenceCreateInfo fenceCreateInfo {};
			fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
			fenceCreateInfo.flags = 0;
            VkFence fence;
            if(vkCreateFence(_device, &fenceCreateInfo, nullptr, &fence) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create fence.");
            }
            if(vkQueueSubmit(compute.queue, 1, &submitInfo, fence) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to submit to queue.");
            }
            if(vkWaitForFences(_device, 1, &fence, VK_TRUE, 100000000000) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to wait for fences.");
            }
            vkDestroyFence(_device, fence, nullptr);

			vkFreeCommandBuffers(_device, compute.commandPool, 1, &cmdBuffer);

			vkDestroyBuffer(_device, stagingBuffer, nullptr);

            VkSamplerCreateInfo samplerCreateInfo = {};
            samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
            samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
            samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            samplerCreateInfo.mipLodBias = 0.0f;
            samplerCreateInfo.compareOp = VK_COMPARE_OP_NEVER;
            samplerCreateInfo.minLod = 0.0f;
            samplerCreateInfo.maxLod = 1.0f;
            samplerCreateInfo.maxAnisotropy = 1.0f;
            samplerCreateInfo.anisotropyEnable = VK_FALSE;
            samplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
            if(vkCreateSampler(_device, &samplerCreateInfo, nullptr, &compute.texSampler) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create texture sampler.");
            }

            VkImageViewCreateInfo viewCreateInfo = {};
            viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
            viewCreateInfo.components = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A };
            viewCreateInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            viewCreateInfo.subresourceRange.levelCount = 1;
            viewCreateInfo.image = compute.texImage;
            if(vkCreateImageView(_device, &viewCreateInfo, nullptr, &compute.texImageView) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create texture image view.");
            }

            compute.texDescriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            compute.texDescriptor.imageView = compute.texImageView;
            compute.texDescriptor.sampler = compute.texSampler;
        }

        void prepareTextureTarget()
        {
            VkFormatProperties formatProperties;

            vkGetPhysicalDeviceFormatProperties(_physicalDevice, VK_FORMAT_R8G8B8A8_UNORM, &formatProperties);

            assert(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT);

            QueueFamilyIndices indices = findQueueFamilies(_physicalDevice);

            VkImageCreateInfo imageCreateInfo {};
			imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
            imageCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
            imageCreateInfo.extent.width = static_cast<uint32_t>(_texWidth);
            imageCreateInfo.extent.height = static_cast<uint32_t>(_texHeight);
            imageCreateInfo.extent.depth = 1;
            imageCreateInfo.mipLevels = 1;
            imageCreateInfo.arrayLayers = 1;
            imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageCreateInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		    imageCreateInfo.flags = 0;
            imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            imageCreateInfo.queueFamilyIndexCount = 1;
            imageCreateInfo.pQueueFamilyIndices = &indices.computeFamily.value();

            VkMemoryAllocateInfo memAllocInfo {};
			memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            VkMemoryRequirements memReqs;

            if(vkCreateImage(_device, &imageCreateInfo, nullptr, &compute.targetImage) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create target image.");
            }

            vkGetImageMemoryRequirements(_device, compute.targetImage, &memReqs);
            memAllocInfo.allocationSize = memReqs.size;
            memAllocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if(vkAllocateMemory(_device, &memAllocInfo, nullptr, &compute.targetDeviceMemory) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to allocate memory for target device memory.");
            }

            if(vkBindImageMemory(_device, compute.targetImage, compute.targetDeviceMemory, 0) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to bind target image memory.");
            }

            VkCommandBufferAllocateInfo commandBufferAllocateInfo {};
			commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			commandBufferAllocateInfo.commandPool = compute.commandPool;
			commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			commandBufferAllocateInfo.commandBufferCount = 1;
		    VkCommandBuffer cmdBuffer;
		    if(vkAllocateCommandBuffers(_device, &commandBufferAllocateInfo, &cmdBuffer) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to allocate command buffer.");
            }

			VkCommandBufferBeginInfo cmdBufferBeginInfo {};
			cmdBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			if(vkBeginCommandBuffer(cmdBuffer, &cmdBufferBeginInfo) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to begin command buffer.");
            }

            VkImageSubresourceRange subresourceRange = {};
			subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			subresourceRange.baseMipLevel = 0;
			subresourceRange.levelCount = 1;
			subresourceRange.layerCount = 1;

            VkImageMemoryBarrier imageMemoryBarrier {};
			imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
			imageMemoryBarrier.image = compute.targetImage;
			imageMemoryBarrier.subresourceRange = subresourceRange;
            imageMemoryBarrier.srcAccessMask = 0;

            vkCmdPipelineBarrier(
				cmdBuffer,
				65536U,
				65536U,
				0,
				0, nullptr,
				0, nullptr,
				1, &imageMemoryBarrier);

            if (cmdBuffer == VK_NULL_HANDLE)
            {
                return;
            }

            if(vkEndCommandBuffer(cmdBuffer) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to end command buffer.");
            }

            VkSubmitInfo submitInfo {};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmdBuffer;
            VkFenceCreateInfo fenceCreateInfo {};
			fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
			fenceCreateInfo.flags = 0;
            VkFence fence;
            if(vkCreateFence(_device, &fenceCreateInfo, nullptr, &fence) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create fence.");
            }
            if(vkQueueSubmit(compute.queue, 1, &submitInfo, fence) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to submit to queue.");
            }
            if(vkWaitForFences(_device, 1, &fence, VK_TRUE, 100000000000) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to wait for fences.");
            }
            vkDestroyFence(_device, fence, nullptr);
  
            vkFreeCommandBuffers(_device, compute.commandPool, 1, &cmdBuffer);
  
            VkSamplerCreateInfo sampler {};
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
            sampler.maxLod = 1.0f;
            sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
            if(vkCreateSampler(_device, &sampler, nullptr, &compute.targetSampler) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create sampler.");
            }

            VkImageViewCreateInfo view {};
			view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            view.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view.format = VK_FORMAT_R8G8B8A8_UNORM;
            view.components = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A };
            view.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            view.image = compute.targetImage;
            if(vkCreateImageView(_device, &view, nullptr, &compute.targetImageView) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create image view.");
            }

            compute.targetDescriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            compute.targetDescriptor.imageView = compute.targetImageView;
            compute.targetDescriptor.sampler = compute.targetSampler;
        }

        void setupDescriptorPool()
        {
            VkDescriptorPoolSize descriptorPoolSize {};
			descriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			descriptorPoolSize.descriptorCount = 2;

            std::vector<VkDescriptorPoolSize> poolSizes = 
            {
                descriptorPoolSize
            };

            VkDescriptorPoolCreateInfo descriptorPoolInfo{};
			descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
			descriptorPoolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
			descriptorPoolInfo.pPoolSizes = poolSizes.data();
			descriptorPoolInfo.maxSets = 1;
            if(vkCreateDescriptorPool(_device, &descriptorPoolInfo, nullptr, &compute.descriptorPool) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create descriptor pool.");
            }
        }

        void buildComputeCommandBuffer()
        {
            vkQueueWaitIdle(compute.queue);

            VkCommandBufferBeginInfo cmdBufferBeginInfo {};
			cmdBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

            if(vkBeginCommandBuffer(compute.commandBuffer, &cmdBufferBeginInfo) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to begin command buffer.");
            }

            vkCmdBindPipeline(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipeline);
            vkCmdBindDescriptorSets(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayout, 0, 1, &compute.descriptorSet, 0, 0);

            vkCmdDispatch(compute.commandBuffer, _texWidth / 16, _texHeight / 16, 1);

            vkEndCommandBuffer(compute.commandBuffer);
        }

        VkShaderModule loadShaderModule(const char *fileName, VkDevice device)
		{
			std::ifstream is(fileName, std::ios::binary | std::ios::in | std::ios::ate);

			if (is.is_open())
			{
				size_t size = is.tellg();
				is.seekg(0, std::ios::beg);
				char* shaderCode = new char[size];
				is.read(shaderCode, size);
				is.close();

				assert(size > 0);

				VkShaderModule shaderModule;
				VkShaderModuleCreateInfo moduleCreateInfo{};
				moduleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				moduleCreateInfo.codeSize = size;
				moduleCreateInfo.pCode = (uint32_t*)shaderCode;

				if(vkCreateShaderModule(device, &moduleCreateInfo, NULL, &shaderModule) != VK_SUCCESS)
                {
                    throw std::runtime_error("Failed to create shader module.");
                }

				delete[] shaderCode;

				return shaderModule;
			}
			else
			{
				std::cerr << "Error: Could not open shader file \"" << fileName << "\"" << "\n";
				return VK_NULL_HANDLE;
			}
		}

        VkPipelineShaderStageCreateInfo loadShader(std::string fileName, VkShaderStageFlagBits stage)
        {
            compute.shaderModule = loadShaderModule(fileName.c_str(), _device);
            VkPipelineShaderStageCreateInfo shaderStage = {};
            shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            shaderStage.stage = stage;
            shaderStage.module = compute.shaderModule;
            shaderStage.pName = "main";
            assert(shaderStage.module != VK_NULL_HANDLE);
            return shaderStage;
        }

        void prepareCompute()
        {
            QueueFamilyIndices indices = findQueueFamilies(_physicalDevice);
            vkGetDeviceQueue(_device, indices.computeFamily.value(), 0, &compute.queue);

            VkDescriptorSetLayoutBinding setTexLayoutBinding {};
			setTexLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			setTexLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
			setTexLayoutBinding.binding = 0;
			setTexLayoutBinding.descriptorCount = 1;

            VkDescriptorSetLayoutBinding setTargetLayoutBinding {};
			setTargetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			setTargetLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
			setTargetLayoutBinding.binding = 1;
			setTargetLayoutBinding.descriptorCount = 1;

            std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings =
            {
                setTexLayoutBinding,
                setTargetLayoutBinding
            };

            VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo{};
			descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
			descriptorSetLayoutCreateInfo.pBindings = setLayoutBindings.data();
			descriptorSetLayoutCreateInfo.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());

      		if(vkCreateDescriptorSetLayout(_device, &descriptorSetLayoutCreateInfo, nullptr, &compute.descriptorSetLayout) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create descriptor set layout.");
            }

            VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo {};
			pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
			pipelineLayoutCreateInfo.setLayoutCount = 1;
			pipelineLayoutCreateInfo.pSetLayouts = &compute.descriptorSetLayout;

            if(vkCreatePipelineLayout(_device, &pipelineLayoutCreateInfo, nullptr, &compute.pipelineLayout) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create pipeline layout.");
            }

            VkDescriptorSetAllocateInfo descriptorSetAllocateInfo {};
			descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			descriptorSetAllocateInfo.descriptorPool = compute.descriptorPool;
			descriptorSetAllocateInfo.pSetLayouts = &compute.descriptorSetLayout;
			descriptorSetAllocateInfo.descriptorSetCount = 1;

            if(vkAllocateDescriptorSets(_device, &descriptorSetAllocateInfo, &compute.descriptorSet) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to allocate descriptor set.");
            }

            VkWriteDescriptorSet writeTexDescriptorSet {};
			writeTexDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			writeTexDescriptorSet.dstSet = compute.descriptorSet;
			writeTexDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			writeTexDescriptorSet.dstBinding = 0;
			writeTexDescriptorSet.pImageInfo = &compute.texDescriptor;
			writeTexDescriptorSet.descriptorCount = 1U;

            VkWriteDescriptorSet writeTargetDescriptorSet {};
			writeTargetDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			writeTargetDescriptorSet.dstSet = compute.descriptorSet;
			writeTargetDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			writeTargetDescriptorSet.dstBinding = 1;
			writeTargetDescriptorSet.pImageInfo = &compute.targetDescriptor;
			writeTargetDescriptorSet.descriptorCount = 1U;

            std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets =
            {
                writeTexDescriptorSet,
                writeTargetDescriptorSet
            };

       		vkUpdateDescriptorSets(_device, computeWriteDescriptorSets.size(), computeWriteDescriptorSets.data(), 0, NULL);

            VkComputePipelineCreateInfo computePipelineCreateInfo {};
			computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
			computePipelineCreateInfo.layout = compute.pipelineLayout;
			computePipelineCreateInfo.flags = 0;

            std::string fileName = "../shaders/greyscale.comp.spv";
			computePipelineCreateInfo.stage = loadShader(fileName, VK_SHADER_STAGE_COMPUTE_BIT);
			VkPipeline pipeline;
			if(vkCreateComputePipelines(_device, nullptr, 1, &computePipelineCreateInfo, nullptr, &compute.pipeline) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create compute pipeline.");
            }

            VkCommandBufferAllocateInfo commandBufferAllocateInfo {};
			commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			commandBufferAllocateInfo.commandPool = compute.commandPool;
			commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			commandBufferAllocateInfo.commandBufferCount = 1;

            if(vkAllocateCommandBuffers(_device, &commandBufferAllocateInfo, &compute.commandBuffer) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to allocate command buffer.");
            }

            buildComputeCommandBuffer();
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

        uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) 
        {
            VkPhysicalDeviceMemoryProperties memProperties;
            vkGetPhysicalDeviceMemoryProperties(_physicalDevice, &memProperties);

            for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) 
            {
                if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) 
                {
                    return i;
                }
            }

            throw std::runtime_error("Failed to find suitable memory type.");
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

        static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) 
        {
            std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

            return VK_FALSE;
        }

        void computeShader()
        {
            VkDeviceSize imageSize = _texWidth * _texHeight * 4;

            vkDeviceWaitIdle( _device );
            VkSubmitInfo submit{ VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &compute.commandBuffer, 0, nullptr };
            vkQueueSubmit( compute.queue, 1, &submit, VK_NULL_HANDLE ); 
            vkQueueWaitIdle( compute.queue );

			VkBufferImageCopy bufferCopyRegion = {};
            bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            bufferCopyRegion.imageSubresource.mipLevel = 0;
            bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
            bufferCopyRegion.imageSubresource.layerCount = 1;
            bufferCopyRegion.imageExtent.width = static_cast<uint32_t>(_texWidth);
            bufferCopyRegion.imageExtent.height = static_cast<uint32_t>(_texHeight);
            bufferCopyRegion.imageExtent.depth = 1;
            bufferCopyRegion.bufferOffset = 0;

            VkBuffer stagingBuffer;
            VkDeviceMemory stagingMemory;
            VkBufferCreateInfo bufCreateInfo {};
			bufCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufCreateInfo.size = imageSize;
            bufCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            bufCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VkMemoryAllocateInfo memAllocInfo {};
			memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            VkMemoryRequirements memReqs;

            if(vkCreateBuffer(_device, &bufCreateInfo, nullptr, &stagingBuffer) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create buffer.");
            }

            vkGetBufferMemoryRequirements(_device, stagingBuffer, &memReqs);

            memAllocInfo.allocationSize = memReqs.size;
            memAllocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);

            if(vkAllocateMemory(_device, &memAllocInfo, nullptr, &stagingMemory) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to allocate memory.");
            }

            if(vkBindBufferMemory(_device, stagingBuffer, stagingMemory, 0) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to bind buffer memory.");
            }

            VkCommandBufferAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandPool = compute.commandPool;
            allocInfo.commandBufferCount = 1;

            VkCommandBuffer commandBuffer;
            vkAllocateCommandBuffers(_device, &allocInfo, &commandBuffer);

            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            vkBeginCommandBuffer(commandBuffer, &beginInfo);

            vkCmdCopyImageToBuffer(
				commandBuffer,
                compute.targetImage,
                VK_IMAGE_LAYOUT_GENERAL,
                stagingBuffer,
                1,
                &bufferCopyRegion
			);

            vkEndCommandBuffer(commandBuffer);

            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffer;

            vkQueueSubmit(compute.queue, 1, &submitInfo, VK_NULL_HANDLE);
            vkQueueWaitIdle(compute.queue);

            vkFreeCommandBuffers(_device, compute.commandPool, 1, &commandBuffer);

            void* data;
            vkMapMemory( _device, stagingMemory, 0, VK_WHOLE_SIZE, 0, &data ); 
            stbi_write_png("image.png", _texWidth, _texHeight, 4, data, _texWidth*4);
            vkUnmapMemory( _device, stagingMemory );

            vkDestroyBuffer(_device, stagingBuffer, nullptr);
            vkFreeMemory(_device, stagingMemory, nullptr);
        }

    private:
        VkPhysicalDevice _physicalDevice;
        VkDevice _device;
        VkInstance _instance;
        VkDebugUtilsMessengerEXT _debugMessenger;
        struct Compute {
            VkQueue queue;								
            VkCommandPool commandPool;					
            VkCommandBuffer commandBuffer;			
            VkDescriptorSetLayout descriptorSetLayout;	
            VkDescriptorPool descriptorPool;
            VkDescriptorSet descriptorSet;
            VkDescriptorImageInfo texDescriptor;	
            VkDescriptorImageInfo targetDescriptor;			
            VkPipelineLayout pipelineLayout;	
            VkSampler texSampler;
            VkSampler targetSampler;		
            VkPipeline pipeline;			
            VkImage texImage;
            VkImage targetImage;	
            VkImageView texImageView;
            VkImageView targetImageView;	
            VkDeviceMemory texDeviceMemory;
            VkDeviceMemory targetDeviceMemory;	
            VkDeviceMemory stagingMemory;
            VkShaderModule shaderModule;
        } compute;

        int _texWidth, _texHeight;
};

int main(int argc, char** argv)
{
    VulkanCompute vc;

    try
    {
        vc.run();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}