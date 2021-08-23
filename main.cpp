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
            loadImage();
        }

        void cleanUp()
        {
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

        void loadImage()
        {
            int width, height, channels;
            FILE* pFile;
            pFile = fopen ("../data/vulkan.png" , "r");
            unsigned char* pixels = stbi_load_from_file(pFile, &width, &height, &channels, STBI_rgb_alpha);
            if(!pixels)
            {
                throw std::runtime_error("Failed to load texture file.");
            }

            VkDeviceSize imageSize = width*height*channels;

            VkFormat image_format = VK_FORMAT_R8G8B8A8_SRGB;

            QueueFamilyIndices indices = findQueueFamilies(_physicalDevice);

            const VkBufferCreateInfo bufferCreateInfo = {
            VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            0,
            0,
            imageSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_SHARING_MODE_EXCLUSIVE,
            1,
            &indices.computeFamily.value()
            };

            VkBuffer in_buffer;
            if(vkCreateBuffer(_device, &bufferCreateInfo, 0, &in_buffer) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create buffer.");
            }

            uint32_t memoryTypeIndex = VK_MAX_MEMORY_TYPES;

            VkPhysicalDeviceMemoryProperties properties;

            vkGetPhysicalDeviceMemoryProperties(_physicalDevice, &properties);

             for (uint32_t k = 0; k < properties.memoryTypeCount; k++) 
             {
                if ((VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT & properties.memoryTypes[k].propertyFlags) &&
                    (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT & properties.memoryTypes[k].propertyFlags) &&
                    (imageSize < properties.memoryHeaps[properties.memoryTypes[k].heapIndex].size)) 
                {
                    memoryTypeIndex = k;
                    break;
                }
            }

            if(memoryTypeIndex == VK_MAX_MEMORY_TYPES ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_SUCCESS)
            {
                throw std::runtime_error("Failed to get memory type from device.");
            }

            const VkMemoryAllocateInfo memoryAllocateInfo = {
                VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                0,
                imageSize,
                memoryTypeIndex
            };

            VkDeviceMemory memory;
            if(vkAllocateMemory(_device, &memoryAllocateInfo, 0, &memory) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to allocate memory.");
            }

            void* data;
            if(vkMapMemory(_device, memory, 0, imageSize, 0, (void **)data) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to map memory.");
            }

            memcpy(data, pixels, static_cast<size_t>(imageSize));

            vkUnmapMemory(_device, memory);
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