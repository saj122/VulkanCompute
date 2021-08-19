#include "vulkan/vulkan.hpp"

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
            createComputePipeline();
        }

        void cleanUp()
        {
            for(auto& pipeline : _computePipelines)
            {
                vkDestroyPipeline(_device, pipeline, nullptr);
            }

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

            VkComputePipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipelineInfo.layout = _pipelineLayout;
            pipelineInfo.flags = 0;

            VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
            pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
            if(vkCreatePipelineCache(_device, &pipelineCacheCreateInfo, nullptr, &_pipelineCache) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create pipeline cache.");
            }

            _shaderNames = { "edgedetect" };
            for(auto& shaderName : _shaderNames)
            {
			    pipelineInfo.stage = compShaderStageInfo;
			    VkPipeline pipeline;
			    if(vkCreateComputePipelines(_device, _pipelineCache, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS)
                {
                    throw std::runtime_error("Failed to create compute pipeplines.");
                }

                _computePipelines.push_back(pipeline);
            } 

            QueueFamilyIndices indices = findQueueFamilies(_physicalDevice);

            VkCommandPoolCreateInfo cmdPoolInfo = {};
            cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            cmdPoolInfo.queueFamilyIndex =  indices.computeFamily.value();
            cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            if(vkCreateCommandPool(_device, &cmdPoolInfo, nullptr, &_computeCommandPool) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create compute command pool.");
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

        std::vector<VkPipeline> _computePipelines;
        std::vector<std::string> _shaderNames;
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