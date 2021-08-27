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
            createPipelineCache();
            createDescriptorSetLayout();
            createComputePipeline();
            createCommandPool();
            createViewImage();
            createTextureImage();
            setupDescriptorPool();
            createDescriptorSets();
            createCommandBuffers();
            createSyncObjects();
            compute();
        }

        void cleanUp()
        {
            vkDestroyImage(_device, _viewImage, nullptr);
            vkFreeMemory(_device, _viewImageDeviceMemory, nullptr);
            vkDestroyImage(_device, _targetImage, nullptr);
            vkFreeMemory(_device, _targetImageDeviceMemory, nullptr);
            for(int i = 0;i < _inFlightFences.size();++i)
                vkDestroyFence(_device, _inFlightFences[i], nullptr);
            vkDestroyImageView(_device, _targetImageView, nullptr);
            vkDestroyImageView(_device, _viewImageView, nullptr);
            vkDestroySampler(_device, _viewSampler, nullptr);
            vkDestroySampler(_device, _targetSampler, nullptr);
            vkDestroyDescriptorPool(_device, _descriptorPool, nullptr);
            vkDestroyDescriptorSetLayout(_device, _descriptorSetLayout, nullptr);
            vkDestroyPipelineCache(_device, _pipelineCache, nullptr);
            vkDestroyPipeline(_device, _computePipeline, nullptr);
            vkDestroyPipelineLayout(_device, _pipelineLayout, nullptr);
            vkDestroyCommandPool(_device, _commandPool, nullptr);
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

        void createDescriptorSetLayout() 
        {
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

            std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = 
            {
                setLayoutBinding,
                setLayoutBinding2
            };

            VkDescriptorSetLayoutCreateInfo layoutInfo{};
            layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layoutInfo.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
            layoutInfo.pBindings = setLayoutBindings.data();

            if (vkCreateDescriptorSetLayout(_device, &layoutInfo, nullptr, &_descriptorSetLayout) != VK_SUCCESS) 
            {
                throw std::runtime_error("Failed to create descriptor set layout.");
            }
        }

        void createPipelineCache()
        {
            VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
            pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
            if(vkCreatePipelineCache(_device, &pipelineCacheCreateInfo, nullptr, &_pipelineCache) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create pipeline cache.");
            }
        }

        void createCommandPool()
        {
            QueueFamilyIndices queueFamilyIndices = findQueueFamilies(_physicalDevice);

            VkCommandPoolCreateInfo poolInfo{};
            poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            poolInfo.queueFamilyIndex = queueFamilyIndices.computeFamily.value();

            if (vkCreateCommandPool(_device, &poolInfo, nullptr, &_commandPool) != VK_SUCCESS) 
            {
                throw std::runtime_error("Failed to create graphics command pool.");
            }
        }

        void createDescriptorSets()
        {
            VkDescriptorSetAllocateInfo descriptorSetAllocateInfo {};
			descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			descriptorSetAllocateInfo.descriptorPool = _descriptorPool;
			descriptorSetAllocateInfo.pSetLayouts = &_descriptorSetLayout;
			descriptorSetAllocateInfo.descriptorSetCount = 1;

            if(vkAllocateDescriptorSets(_device, &descriptorSetAllocateInfo, &_descriptorSet) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to allocate descriptor sets.");
            }

            VkWriteDescriptorSet writeDescriptorSet {};
			writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			writeDescriptorSet.dstSet = _descriptorSet;
			writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			writeDescriptorSet.dstBinding = 0;
			writeDescriptorSet.pImageInfo = &_viewDescriptor;
			writeDescriptorSet.descriptorCount = 1;

            VkWriteDescriptorSet writeDescriptorSet2 {};
			writeDescriptorSet2.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			writeDescriptorSet2.dstSet = _descriptorSet;
			writeDescriptorSet2.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			writeDescriptorSet2.dstBinding = 1;
			writeDescriptorSet2.pImageInfo = &_targetDescriptor;
			writeDescriptorSet2.descriptorCount = 1;

            std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets = 
            {
                writeDescriptorSet,
                writeDescriptorSet2
            };

            vkUpdateDescriptorSets(_device, computeWriteDescriptorSets.size(), computeWriteDescriptorSets.data(), 0, NULL);
        }

        void createCommandBuffers()
        {
            VkCommandBufferAllocateInfo commandBufferAllocateInfo {};
			commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			commandBufferAllocateInfo.commandPool = _commandPool;
			commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			commandBufferAllocateInfo.commandBufferCount = 1;	

		    if(vkAllocateCommandBuffers(_device, &commandBufferAllocateInfo, &_commandBuffer) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to allocate command buffer.");
            }

            vkQueueWaitIdle(_computeQueue);

            VkCommandBufferBeginInfo cmdBufferBeginInfo {};
			cmdBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

            if(vkBeginCommandBuffer(_commandBuffer, &cmdBufferBeginInfo) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to begin command buffer.");
            }

            vkCmdBindPipeline(_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _computePipeline);
            vkCmdBindDescriptorSets(_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipelineLayout, 0, 1, &_descriptorSet, 0, 0);

            vkCmdDispatch(_commandBuffer, _width / 16, _height / 16, 1);

            vkEndCommandBuffer(_commandBuffer);
        }

        void createSyncObjects() 
        {
            _inFlightFences.resize(1);

            VkFenceCreateInfo fenceInfo{};
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

            for (size_t i = 0; i < 1; i++) 
            {
                if(vkCreateFence(_device, &fenceInfo, nullptr, &_inFlightFences[i]) != VK_SUCCESS) 
                {
                    throw std::runtime_error("failed to create synchronization objects for a frame!");
                }
            }
        }

        void createComputePipeline()
        {
            auto compShaderCode = readFile("../shaders/greyscale.comp.spv");

            VkShaderModule shaderModule = createShaderModule(compShaderCode);

            VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
            pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &_descriptorSetLayout;

            if (vkCreatePipelineLayout(_device, &pipelineLayoutInfo, nullptr, &_pipelineLayout) != VK_SUCCESS) 
            {
                throw std::runtime_error("failed to create pipeline layout.");
            }

            VkPipelineShaderStageCreateInfo shaderStage = {};
            shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            shaderStage.module = shaderModule;
            shaderStage.pName = "main";
            assert(shaderStage.module != VK_NULL_HANDLE);

            VkComputePipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipelineInfo.basePipelineHandle = _computePipeline;
            pipelineInfo.layout = _pipelineLayout;
            pipelineInfo.flags = 0;
            pipelineInfo.stage = shaderStage;

            if (vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &_computePipeline) != VK_SUCCESS) 
            {
                throw std::runtime_error("Failed to create graphics pipeline.");
            }

            vkDestroyShaderModule(_device, shaderModule, nullptr);
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

        void setupDescriptorPool()
        {
            VkDescriptorPoolSize descriptorPoolSize {};
            descriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorPoolSize.descriptorCount = 2;

            std::vector<VkDescriptorPoolSize> poolSizes = {
                descriptorPoolSize
            };
            VkDescriptorPoolCreateInfo descriptorPoolInfo{};
			descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
			descriptorPoolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
			descriptorPoolInfo.pPoolSizes = poolSizes.data();
			descriptorPoolInfo.maxSets = 3;
            if(vkCreateDescriptorPool(_device, &descriptorPoolInfo, nullptr, &_descriptorPool) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create descriptor pool.");
            }
        }

        void createTextureImage() 
        {
            VkFormatProperties formatProperties;
            vkGetPhysicalDeviceFormatProperties(_physicalDevice, VK_FORMAT_R8G8B8A8_UNORM, &formatProperties);
            // Check if requested image format supports image storage operations
            assert(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT);

            VkImageCreateInfo imageCreateInfo {};
			imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
            imageCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
            imageCreateInfo.extent.width = _width;
            imageCreateInfo.extent.height = _height;
            imageCreateInfo.extent.depth = 1;
            imageCreateInfo.mipLevels = 1;
            imageCreateInfo.arrayLayers = 1;
            imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageCreateInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
            imageCreateInfo.flags = 0;
            QueueFamilyIndices indices = findQueueFamilies(_physicalDevice);
            imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            imageCreateInfo.queueFamilyIndexCount = 1;
            imageCreateInfo.pQueueFamilyIndices = &indices.computeFamily.value();

            VkMemoryAllocateInfo memAllocInfo {};
			memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            VkMemoryRequirements memReqs;

            if(vkCreateImage(_device, &imageCreateInfo, nullptr, &_targetImage) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create image.");
            }

            vkGetImageMemoryRequirements(_device, _targetImage, &memReqs);
            memAllocInfo.allocationSize = memReqs.size;
            memAllocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if(vkAllocateMemory(_device, &memAllocInfo, nullptr, &_targetImageDeviceMemory) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to allocate memory.");
            }
            if(vkBindImageMemory(_device, _targetImage, _targetImageDeviceMemory, 0) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to bind image memory.");
            }

            VkCommandBufferAllocateInfo commandBufferAllocateInfo {};
			commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			commandBufferAllocateInfo.commandPool = _commandPool;
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
                throw std::runtime_error("Failed to record command buffer.");
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
			imageMemoryBarrier.image = _targetImage;
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

            flushCommandBuffer(cmdBuffer, _computeQueue, _commandPool);

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
            if(vkCreateSampler(_device, &sampler, nullptr, &_targetSampler) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create sampler.");
            }

            VkImageViewCreateInfo view {};
			view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            view.image = VK_NULL_HANDLE;
            view.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view.format = VK_FORMAT_R8G8B8A8_UNORM;
            view.components = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A };
            view.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            view.image = _targetImage;
            if(vkCreateImageView(_device, &view, nullptr, &_targetImageView) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create image view.");
            }

            _targetDescriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            _targetDescriptor.imageView = _targetImageView;
            _targetDescriptor.sampler = _targetSampler;
        }

        void createViewImage()
        {
            int texChannels;
            stbi_uc* pixels = stbi_load("../data/vulkan.png", &_width, &_height, &texChannels, STBI_rgb_alpha);
            VkDeviceSize imageSize = _width * _height * 4;

            if (!pixels) 
            {
                throw std::runtime_error("Failed to load texture image.");
            }

            VkBuffer stagingBuffer;
            createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, stagingBuffer, _viewImageDeviceMemory);

            void* data;
            vkMapMemory(_device, _viewImageDeviceMemory, 0, imageSize, 0, &data);
            memcpy(data, pixels, static_cast<size_t>(imageSize));
            vkUnmapMemory(_device, _viewImageDeviceMemory);

            stbi_image_free(pixels);

            createImage(_width, _height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, _viewImage, _viewImageDeviceMemory, _viewImageView, _viewSampler, _viewDescriptor);

            transitionImageLayout(_viewImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            copyBufferToImage(stagingBuffer, _viewImage, static_cast<uint32_t>(_width), static_cast<uint32_t>(_height));
            transitionImageLayout(_viewImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);

            vkDestroyBuffer(_device, stagingBuffer, nullptr);
        }

        void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) 
        {
            VkBufferCreateInfo bufferInfo{};
            bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferInfo.size = size;
            bufferInfo.usage = usage;
            bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            if (vkCreateBuffer(_device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) 
            {
                throw std::runtime_error("Failed to create buffer.");
            }

            VkMemoryRequirements memRequirements;
            vkGetBufferMemoryRequirements(_device, buffer, &memRequirements);

            VkMemoryAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.allocationSize = memRequirements.size;
            allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

            if (vkAllocateMemory(_device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) 
            {
                throw std::runtime_error("Failed to allocate buffer memory.");
            }

            vkBindBufferMemory(_device, buffer, bufferMemory, 0);
        }

        void flushCommandBuffer(VkCommandBuffer commandBuffer, VkQueue queue, VkCommandPool pool)
	    {
            if (commandBuffer == VK_NULL_HANDLE)
            {
                return;
            }

            if(vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to end command buffer.");
            }

            VkSubmitInfo submitInfo {};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffer;

            VkFenceCreateInfo fenceCreateInfo {};
			fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
			fenceCreateInfo.flags = 0;
            VkFence fence;
            if(vkCreateFence(_device, &fenceCreateInfo, nullptr, &fence) != VK_SUCCESS)
            {
                std::runtime_error("Failed to create fence.");
            }

            if(vkQueueSubmit(queue, 1, &submitInfo, fence) != VK_SUCCESS)
            {
                std::runtime_error("Failed to submit queue.");
            }

            if(vkWaitForFences(_device, 1, &fence, VK_TRUE, 100000000000));
            vkDestroyFence(_device, fence, nullptr);

            vkFreeCommandBuffers(_device, pool, 1, &commandBuffer);
        }

        void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory, VkImageView& imageView, VkSampler& sampler, VkDescriptorImageInfo& descriptor) 
        {
            VkImageCreateInfo imageInfo{};
            imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.extent.width = width;
            imageInfo.extent.height = height;
            imageInfo.extent.depth = 1;
            imageInfo.mipLevels = 1;
            imageInfo.arrayLayers = 1;
            imageInfo.format = format;
            imageInfo.tiling = tiling;
            imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            if (vkCreateImage(_device, &imageInfo, nullptr, &image) != VK_SUCCESS) 
            {
                throw std::runtime_error("Failed to create image.");
            }

            VkMemoryRequirements memRequirements;
            vkGetImageMemoryRequirements(_device, image, &memRequirements);

            VkMemoryAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.allocationSize = memRequirements.size;
            allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

            if (vkAllocateMemory(_device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) 
            {
                throw std::runtime_error("Failed to allocate image memory.");
            }

            vkBindImageMemory(_device, image, imageMemory, 0);

            VkCommandBufferAllocateInfo commandBufferAllocateInfo {};
			commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			commandBufferAllocateInfo.commandPool = _commandPool;
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
                throw std::runtime_error("Failed to begin command buffer recording.");
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
			imageMemoryBarrier.image = image;
			imageMemoryBarrier.subresourceRange = subresourceRange;
            imageMemoryBarrier.srcAccessMask = 0;

            vkCmdPipelineBarrier(
				cmdBuffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				0,
				0, nullptr,
				0, nullptr,
				1, &imageMemoryBarrier);

            flushCommandBuffer(cmdBuffer, _computeQueue, _commandPool);
            
            VkSamplerCreateInfo samplerInfo {};
			samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
			samplerInfo.maxAnisotropy = 1.0f;
            samplerInfo.magFilter = VK_FILTER_LINEAR;
            samplerInfo.minFilter = VK_FILTER_LINEAR;
            samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            samplerInfo.addressModeV = samplerInfo.addressModeU;
            samplerInfo.addressModeW = samplerInfo.addressModeU;
            samplerInfo.mipLodBias = 0.0f;
            samplerInfo.maxAnisotropy = 1.0f;
            samplerInfo.compareOp = VK_COMPARE_OP_NEVER;
            samplerInfo.minLod = 0.0f;
            samplerInfo.maxLod = 1;
            samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
            if(vkCreateSampler(_device, &samplerInfo, nullptr, &sampler) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create sampler.");
            }

            VkImageViewCreateInfo view {};
			view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            view.image = VK_NULL_HANDLE;
            view.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view.format = format;
            view.components = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A };
            view.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            view.image = image;
            if(vkCreateImageView(_device, &view, nullptr, &imageView) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create image view.");
            }

            descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            descriptor.imageView = imageView;
            descriptor.sampler = sampler;
        }

        void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) 
        {
            VkCommandBuffer commandBuffer = beginSingleTimeCommands();

            VkImageMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout = oldLayout;
            barrier.newLayout = newLayout;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = image;
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount = 1;

            VkPipelineStageFlags sourceStage;
            VkPipelineStageFlags destinationStage;

            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;

            vkCmdPipelineBarrier(
                commandBuffer,
                sourceStage, destinationStage,
                0,
                0, nullptr,
                0, nullptr,
                1, &barrier
            );

            endSingleTimeCommands(commandBuffer);
        }

        void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) 
        {
            VkCommandBuffer commandBuffer = beginSingleTimeCommands();

            VkBufferImageCopy region{};
            region.bufferOffset = 0;
            region.bufferRowLength = 0;
            region.bufferImageHeight = 0;
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = 0;
            region.imageSubresource.baseArrayLayer = 0;
            region.imageSubresource.layerCount = 1;
            region.imageOffset = {0, 0, 0};
            region.imageExtent = {
                width,
                height,
                1
            };

            vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            endSingleTimeCommands(commandBuffer);
        }

        VkCommandBuffer beginSingleTimeCommands() 
        {
            VkCommandBufferAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandPool = _commandPool;
            allocInfo.commandBufferCount = 1;

            VkCommandBuffer commandBuffer;
            vkAllocateCommandBuffers(_device, &allocInfo, &commandBuffer);

            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            vkBeginCommandBuffer(commandBuffer, &beginInfo);

            return commandBuffer;
        }

        void endSingleTimeCommands(VkCommandBuffer commandBuffer) 
        {
            vkEndCommandBuffer(commandBuffer);

            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffer;

            vkQueueSubmit(_computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
            vkQueueWaitIdle(_computeQueue);

            vkFreeCommandBuffers(_device, _commandPool, 1, &commandBuffer);
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

        void compute()
        {
            VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

            VkSubmitInfo computeSubmitInfo {};
			computeSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            computeSubmitInfo.commandBufferCount = 1;
            computeSubmitInfo.pCommandBuffers = &_commandBuffer;
            computeSubmitInfo.pWaitDstStageMask = &waitStageMask;
            if(vkQueueSubmit(_computeQueue, 1, &computeSubmitInfo, VK_NULL_HANDLE) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to submit to compute queue.");
            }

            VkDeviceSize imageSize = _width * _height * 4;

            unsigned char* image = new unsigned char[_width*_height*4];
            void* data;
            vkMapMemory(_device, _viewImageDeviceMemory, 0, imageSize, 0, &data);
            memcpy(image, data, static_cast<size_t>(imageSize));
            vkUnmapMemory(_device, _viewImageDeviceMemory);

            stbi_write_png("stbpng.png", _width, _height, 4, image, _width*4);
        }

    private:
        VkInstance _instance;    
        VkDebugUtilsMessengerEXT _debugMessenger;
        VkPhysicalDevice _physicalDevice = VK_NULL_HANDLE;
        VkDevice _device;
        VkQueue _computeQueue;
        VkImage _viewImage;
        VkImage _targetImage;
        VkImageView _viewImageView;
        VkImageView _targetImageView;
        VkSampler _viewSampler;
        VkSampler _targetSampler;
        VkDeviceMemory _viewImageDeviceMemory;
        VkDeviceMemory _targetImageDeviceMemory;
        VkCommandPool _commandPool;
        VkCommandBuffer _commandBuffer;
        VkPipeline _computePipeline;
        VkPipelineLayout _pipelineLayout;
        VkPipelineCache _pipelineCache;
        VkDescriptorSetLayout _descriptorSetLayout;
        VkDescriptorPool _descriptorPool;
        VkDescriptorSet _descriptorSet;
        VkDescriptorImageInfo _viewDescriptor;
        VkDescriptorImageInfo _targetDescriptor;

        std::vector<VkFence> _inFlightFences;

        int _width, _height;
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