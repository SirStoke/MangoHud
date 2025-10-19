#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// ===== Utilities =====
static std::vector<char> readFile(const std::string &path) {
  std::ifstream file(path, std::ios::ate | std::ios::binary);
  if (!file)
    throw std::runtime_error("Failed to open file: " + path);
  size_t size = (size_t)file.tellg();
  std::vector<char> buffer(size);
  file.seekg(0);
  file.read(buffer.data(), size);
  return buffer;
}

struct QueueFamilyIndices {
  std::optional<uint32_t> graphics;
  std::optional<uint32_t> present;
  bool complete() const { return graphics.has_value() && present.has_value(); }
};

struct Vertex {
  float pos[3];
  float color[3];
};

// 8 cube vertices (pos, color)
static const std::array<Vertex, 8> cubeVertices = {
    Vertex{{-1, -1, -1}, {1, 0, 0}}, Vertex{{1, -1, -1}, {0, 1, 0}},
    Vertex{{1, 1, -1}, {0, 0, 1}},   Vertex{{-1, 1, -1}, {1, 1, 0}},
    Vertex{{-1, -1, 1}, {1, 0, 1}},  Vertex{{1, -1, 1}, {0, 1, 1}},
    Vertex{{1, 1, 1}, {1, 1, 1}},    Vertex{{-1, 1, 1}, {0.2, 0.2, 0.2}}};

static const std::array<uint16_t, 36> cubeIndices = {
    0, 1, 2, 2, 3, 0, // back
    4, 5, 6, 6, 7, 4, // front
    0, 4, 7, 7, 3, 0, // left
    1, 5, 6, 6, 2, 1, // right
    3, 2, 6, 6, 7, 3, // top
    0, 1, 5, 5, 4, 0  // bottom
};

// Very small 4x4 matrix helpers
struct Mat4 {
  float m[16];
};
static Mat4 matMul(const Mat4 &a, const Mat4 &b) {
  Mat4 r{};
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) {
      r.m[i * 4 + j] = 0;
      for (int k = 0; k < 4; k++)
        r.m[i * 4 + j] += a.m[i * 4 + k] * b.m[k * 4 + j];
    }
  return r;
}
static Mat4 matIdentity() {
  Mat4 r{};
  for (int i = 0; i < 16; i++)
    r.m[i] = (i % 5 == 0) ? 1.f : 0.f;
  return r;
}
static Mat4 matPerspective(float fovy, float aspect, float znear, float zfar) {
  float f = 1.0f / tanf(fovy / 2.f);
  Mat4 r{};
  r.m[0] = f / aspect;
  r.m[5] = f;
  r.m[10] = (zfar + znear) / (znear - zfar);
  r.m[11] = -1;
  r.m[14] = (2 * zfar * znear) / (znear - zfar);
  return r;
}
static Mat4 matTranslate(float x, float y, float z) {
  Mat4 r = matIdentity();
  r.m[12] = x;
  r.m[13] = y;
  r.m[14] = z;
  return r;
}
static Mat4 matRotateY(float a) {
  Mat4 r = matIdentity();
  float c = cosf(a), s = sinf(a);
  r.m[0] = c;
  r.m[2] = s;
  r.m[8] = -s;
  r.m[10] = c;
  return r;
}
static Mat4 matRotateX(float a) {
  Mat4 r = matIdentity();
  float c = cosf(a), s = sinf(a);
  r.m[5] = c;
  r.m[6] = s;
  r.m[9] = -s;
  r.m[10] = c;
  return r;
}

struct UBO {
  Mat4 mvp;
};

// ===== Vulkan helpers =====
static VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
              VkDebugUtilsMessageTypeFlagsEXT type,
              const VkDebugUtilsMessengerCallbackDataEXT *data, void *user) {
  std::cerr << "[Vulkan] " << data->pMessage << std::endl;
  return VK_FALSE;
}

class App {
public:
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

private:
  GLFWwindow *window{};

  VkInstance instance{};
  VkDebugUtilsMessengerEXT debugMessenger{};
  VkSurfaceKHR surface{};

  VkPhysicalDevice physicalDevice{};
  VkDevice device{};
  VkQueue graphicsQueue{};
  VkQueue presentQueue{};

  VkSwapchainKHR swapchain{};
  VkFormat swapFormat{};
  VkExtent2D swapExtent{};
  std::vector<VkImage> swapImages;
  std::vector<VkImageView> swapViews;

  VkRenderPass renderPass{};
  VkDescriptorSetLayout descSetLayout{};
  VkPipelineLayout pipelineLayout{};
  VkPipeline pipeline{};

  std::vector<VkFramebuffer> framebuffers;

  VkCommandPool cmdPool{};
  std::vector<VkCommandBuffer> cmdBufs;

  VkBuffer vertexBuffer{};
  VkDeviceMemory vertexMemory{};
  VkBuffer indexBuffer{};
  VkDeviceMemory indexMemory{};

  std::vector<VkBuffer> uniformBuffers;
  std::vector<VkDeviceMemory> uniformMemories;

  VkDescriptorPool descPool{};
  std::vector<VkDescriptorSet> descSets;

  VkSemaphore imageAvailable{};
  VkSemaphore renderFinished{};
  VkFence inFlight{};

  const uint32_t WIDTH = 800, HEIGHT = 600;

  void initWindow() {
    if (!glfwInit())
      throw std::runtime_error("glfwInit failed");
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    window =
        glfwCreateWindow(WIDTH, HEIGHT, "Vulkan X11 Cube", nullptr, nullptr);
    if (!window)
      throw std::runtime_error("Failed to create window");
  }

  void initVulkan() {
    createInstance();
    setupDebug();
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) !=
        VK_SUCCESS)
      throw std::runtime_error("Failed to create surface");
    pickPhysicalDevice();
    createDevice();
    createSwapchain();
    createImageViews();
    createRenderPass();
    createDescriptorSetLayout();
    createPipeline();
    createFramebuffers();
    createCommandPool();
    createBuffers();
    createUniforms();
    createDescriptorPoolAndSets();
    createCommandBuffers();
    createSync();
  }

  void mainLoop() {
    auto start = std::chrono::high_resolution_clock::now();
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
      auto now = std::chrono::high_resolution_clock::now();
      float t = std::chrono::duration<float>(now - start).count();
      updateUniforms(t);
      drawFrame();
    }
    vkDeviceWaitIdle(device);
  }

  void cleanup() {
    vkDestroyFence(device, inFlight, nullptr);
    vkDestroySemaphore(device, renderFinished, nullptr);
    vkDestroySemaphore(device, imageAvailable, nullptr);

    vkDestroyDescriptorPool(device, descPool, nullptr);
    for (size_t i = 0; i < uniformBuffers.size(); ++i) {
      vkDestroyBuffer(device, uniformBuffers[i], nullptr);
      vkFreeMemory(device, uniformMemories[i], nullptr);
    }

    vkDestroyBuffer(device, indexBuffer, nullptr);
    vkFreeMemory(device, indexMemory, nullptr);
    vkDestroyBuffer(device, vertexBuffer, nullptr);
    vkFreeMemory(device, vertexMemory, nullptr);

    vkDestroyCommandPool(device, cmdPool, nullptr);
    for (auto fb : framebuffers)
      vkDestroyFramebuffer(device, fb, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, descSetLayout, nullptr);
    vkDestroyRenderPass(device, renderPass, nullptr);
    for (auto v : swapViews)
      vkDestroyImageView(device, v, nullptr);
    vkDestroySwapchainKHR(device, swapchain, nullptr);
    vkDestroyDevice(device, nullptr);
    auto DestroyDebugUtilsMessengerEXT =
        (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            instance, "vkDestroyDebugUtilsMessengerEXT");
    if (DestroyDebugUtilsMessengerEXT)
      DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);
    glfwDestroyWindow(window);
    glfwTerminate();
  }

  // --- Instance & Debug ---
  void createInstance() {
    VkDebugUtilsMessengerCreateInfoEXT debugInfo{
        VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};

    debugInfo.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;

    debugInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;

    debugInfo.pfnUserCallback = debugCallback;

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "Vulkan X11 Cube";
    app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app.pEngineName = "NoEngine";
    app.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app.apiVersion = VK_API_VERSION_1_2;

    uint32_t count = 0;
    const char **glfwExt = glfwGetRequiredInstanceExtensions(&count);
    std::vector<const char *> extensions(glfwExt, glfwExt + count);
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    //const std::vector<const char *> layers = {"VK_LAYER_KHRONOS_validation"};
    const std::vector<const char *> layers = {};

    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = (uint32_t)extensions.size();
    ci.ppEnabledExtensionNames = extensions.data();
    ci.enabledLayerCount = (uint32_t)layers.size();
    ci.ppEnabledLayerNames = layers.data();

    // Attach to pNext so vkCreateInstance can log issues
    ci.pNext = &debugInfo;

    if (vkCreateInstance(&ci, nullptr, &instance) != VK_SUCCESS)
      throw std::runtime_error("vkCreateInstance failed");
  }

  void setupDebug() {
    VkDebugUtilsMessengerCreateInfoEXT ci{
        VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    ci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    ci.pfnUserCallback = debugCallback;
    auto CreateDebugUtilsMessengerEXT =
        (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            instance, "vkCreateDebugUtilsMessengerEXT");
    if (CreateDebugUtilsMessengerEXT)
      CreateDebugUtilsMessengerEXT(instance, &ci, nullptr, &debugMessenger);
  }

  // --- Physical/Logical Device ---
  QueueFamilyIndices findQueueFamilies(VkPhysicalDevice dev) {
    QueueFamilyIndices indices;
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, nullptr);
    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, props.data());
    for (uint32_t i = 0; i < count; i++) {
      if (props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
        indices.graphics = i;
      VkBool32 presentSupport = false;
      vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface, &presentSupport);
      if (presentSupport)
        indices.present = i;
      if (indices.complete())
        break;
    }
    return indices;
  }

  void pickPhysicalDevice() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (count == 0)
      throw std::runtime_error("No Vulkan devices");
    std::vector<VkPhysicalDevice> devs(count);
    vkEnumeratePhysicalDevices(instance, &count, devs.data());
    for (auto d : devs) {
      auto idx = findQueueFamilies(d);
      if (idx.complete()) {
        physicalDevice = d;
        break;
      }
    }
    if (!physicalDevice)
      throw std::runtime_error("No suitable GPU");
  }

  void createDevice() {
    auto idx = findQueueFamilies(physicalDevice);
    std::vector<VkDeviceQueueCreateInfo> qci;
    std::set<uint32_t> unique = {idx.graphics.value(), idx.present.value()};
    float prio = 1.0f;
    for (uint32_t qf : unique) {
      VkDeviceQueueCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
      ci.queueFamilyIndex = qf;
      ci.queueCount = 1;
      ci.pQueuePriorities = &prio;
      qci.push_back(ci);
    }
    const std::vector<const char *> devExt = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    VkPhysicalDeviceFeatures feats{};
    feats.samplerAnisotropy = VK_FALSE;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = (uint32_t)qci.size();
    dci.pQueueCreateInfos = qci.data();
    dci.enabledExtensionCount = (uint32_t)devExt.size();
    dci.ppEnabledExtensionNames = devExt.data();
    dci.pEnabledFeatures = &feats;
    if (vkCreateDevice(physicalDevice, &dci, nullptr, &device) != VK_SUCCESS)
      throw std::runtime_error("vkCreateDevice failed");
    vkGetDeviceQueue(device, idx.graphics.value(), 0, &graphicsQueue);
    vkGetDeviceQueue(device, idx.present.value(), 0, &presentQueue);
  }

  // --- Swapchain
  void createSwapchain() {
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &caps);
    uint32_t fcount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &fcount,
                                         nullptr);
    std::vector<VkSurfaceFormatKHR> formats(fcount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &fcount,
                                         formats.data());
    uint32_t mcount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &mcount,
                                              nullptr);
    std::vector<VkPresentModeKHR> modes(mcount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &mcount,
                                              modes.data());

    VkSurfaceFormatKHR chosenFmt = formats[0];
    for (auto &f : formats)
      if (f.format == VK_FORMAT_B8G8R8A8_UNORM &&
          f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        chosenFmt = f;
    VkPresentModeKHR chosenMode = VK_PRESENT_MODE_FIFO_KHR; // always available
    for (auto m : modes)
      if (m == VK_PRESENT_MODE_MAILBOX_KHR)
        chosenMode = m;
    VkExtent2D extent = caps.currentExtent.width != UINT32_MAX
                            ? caps.currentExtent
                            : VkExtent2D{WIDTH, HEIGHT};

    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount)
      imageCount = caps.maxImageCount;

    auto idx = findQueueFamilies(physicalDevice);
    uint32_t qIdx[] = {idx.graphics.value(), idx.present.value()};

    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface = surface;
    ci.minImageCount = imageCount;
    ci.imageFormat = chosenFmt.format;
    ci.imageColorSpace = chosenFmt.colorSpace;
    ci.imageExtent = extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (idx.graphics != idx.present) {
      ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
      ci.queueFamilyIndexCount = 2;
      ci.pQueueFamilyIndices = qIdx;
    } else {
      ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = chosenMode;
    ci.clipped = VK_TRUE;

    if (vkCreateSwapchainKHR(device, &ci, nullptr, &swapchain) != VK_SUCCESS)
      throw std::runtime_error("vkCreateSwapchainKHR failed");

    uint32_t scCount = 0;
    vkGetSwapchainImagesKHR(device, swapchain, &scCount, nullptr);
    swapImages.resize(scCount);
    vkGetSwapchainImagesKHR(device, swapchain, &scCount, swapImages.data());
    swapFormat = chosenFmt.format;
    swapExtent = extent;
  }

  void createImageViews() {
    swapViews.resize(swapImages.size());
    for (size_t i = 0; i < swapImages.size(); ++i) {
      VkImageViewCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      ci.image = swapImages[i];
      ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
      ci.format = swapFormat;
      ci.components = {
          VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
          VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
      ci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      ci.subresourceRange.levelCount = 1;
      ci.subresourceRange.layerCount = 1;
      if (vkCreateImageView(device, &ci, nullptr, &swapViews[i]) != VK_SUCCESS)
        throw std::runtime_error("vkCreateImageView failed");
    }
  }

  // --- Render pass & pipeline ---
  void createRenderPass() {
    VkAttachmentDescription color{};
    color.format = swapFormat;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &colorRef;

    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    ci.attachmentCount = 1;
    ci.pAttachments = &color;
    ci.subpassCount = 1;
    ci.pSubpasses = &sub;
    ci.dependencyCount = 1;
    ci.pDependencies = &dep;
    if (vkCreateRenderPass(device, &ci, nullptr, &renderPass) != VK_SUCCESS)
      throw std::runtime_error("vkCreateRenderPass failed");
  }

  void createDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding ubo{};
    ubo.binding = 0;
    ubo.descriptorCount = 1;
    ubo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ubo.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    VkDescriptorSetLayoutCreateInfo ci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    ci.bindingCount = 1;
    ci.pBindings = &ubo;
    if (vkCreateDescriptorSetLayout(device, &ci, nullptr, &descSetLayout) !=
        VK_SUCCESS)
      throw std::runtime_error("vkCreateDescriptorSetLayout failed");
  }

  VkShaderModule loadShader(const std::string &path) {
    auto bytes = readFile(path);
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = bytes.size();
    ci.pCode = reinterpret_cast<const uint32_t *>(bytes.data());
    VkShaderModule mod{};
    if (vkCreateShaderModule(device, &ci, nullptr, &mod) != VK_SUCCESS)
      throw std::runtime_error("vkCreateShaderModule failed: " + path);
    return mod;
  }

  void createPipeline() {
    VkShaderModule vert = loadShader("shaders/bin/cube.vert.spv");
    VkShaderModule frag = loadShader("shaders/bin/cube.frag.spv");

    VkPipelineShaderStageCreateInfo vs{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    vs.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vs.module = vert;
    vs.pName = "main";
    VkPipelineShaderStageCreateInfo fs{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    fs.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fs.module = frag;
    fs.pName = "main";
    VkPipelineShaderStageCreateInfo stages[2] = {vs, fs};

    VkVertexInputBindingDescription bind{};
    bind.binding = 0;
    bind.stride = sizeof(Vertex);
    bind.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    std::array<VkVertexInputAttributeDescription, 2> attrs{};
    attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)};
    attrs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)};
    VkPipelineVertexInputStateCreateInfo vi{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &bind;
    vi.vertexAttributeDescriptionCount = (uint32_t)attrs.size();
    vi.pVertexAttributeDescriptions = attrs.data();

    VkPipelineInputAssemblyStateCreateInfo ia{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport vp{};
    vp.x = 0;
    vp.y = 0;
    vp.width = (float)swapExtent.width;
    vp.height = (float)swapExtent.height;
    vp.minDepth = 0;
    vp.maxDepth = 1;
    VkRect2D sc{{0, 0}, swapExtent};
    VkPipelineViewportStateCreateInfo vpState{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vpState.viewportCount = 1;
    vpState.pViewports = &vp;
    vpState.scissorCount = 1;
    vpState.pScissors = &sc;

    VkPipelineRasterizationStateCreateInfo rs{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_BACK_BIT;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState cbAtt{};
    cbAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    cbAtt.blendEnable = VK_FALSE;
    VkPipelineColorBlendStateCreateInfo cb{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments = &cbAtt;

    VkPipelineLayoutCreateInfo plci{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &descSetLayout;
    if (vkCreatePipelineLayout(device, &plci, nullptr, &pipelineLayout) !=
        VK_SUCCESS)
      throw std::runtime_error("vkCreatePipelineLayout failed");

    VkGraphicsPipelineCreateInfo gp{
        VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gp.stageCount = 2;
    gp.pStages = stages;
    gp.pVertexInputState = &vi;
    gp.pInputAssemblyState = &ia;
    gp.pViewportState = &vpState;
    gp.pRasterizationState = &rs;
    gp.pMultisampleState = &ms;
    gp.pColorBlendState = &cb;
    gp.layout = pipelineLayout;
    gp.renderPass = renderPass;
    gp.subpass = 0;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gp, nullptr,
                                  &pipeline) != VK_SUCCESS)
      throw std::runtime_error("vkCreateGraphicsPipelines failed");

    vkDestroyShaderModule(device, frag, nullptr);
    vkDestroyShaderModule(device, vert, nullptr);
  }

  // --- Framebuffers ---
  void createFramebuffers() {
    framebuffers.resize(swapViews.size());
    for (size_t i = 0; i < swapViews.size(); ++i) {
      VkImageView atts[] = {swapViews[i]};
      VkFramebufferCreateInfo ci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
      ci.renderPass = renderPass;
      ci.attachmentCount = 1;
      ci.pAttachments = atts;
      ci.width = swapExtent.width;
      ci.height = swapExtent.height;
      ci.layers = 1;
      if (vkCreateFramebuffer(device, &ci, nullptr, &framebuffers[i]) !=
          VK_SUCCESS)
        throw std::runtime_error("vkCreateFramebuffer failed");
    }
  }

  // --- Command pool ---
  void createCommandPool() {
    auto idx = findQueueFamilies(physicalDevice);
    VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    ci.queueFamilyIndex = idx.graphics.value();
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(device, &ci, nullptr, &cmdPool) != VK_SUCCESS)
      throw std::runtime_error("vkCreateCommandPool failed");
  }

  uint32_t findMemType(uint32_t typeBits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mem{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &mem);
    for (uint32_t i = 0; i < mem.memoryTypeCount; i++)
      if ((typeBits & (1 << i)) &&
          (mem.memoryTypes[i].propertyFlags & props) == props)
        return i;
    throw std::runtime_error("No suitable memory type");
  }

  void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags props, VkBuffer &buf,
                    VkDeviceMemory &mem) {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bi, nullptr, &buf) != VK_SUCCESS)
      throw std::runtime_error("vkCreateBuffer failed");
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device, buf, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemType(req.memoryTypeBits, props);
    if (vkAllocateMemory(device, &ai, nullptr, &mem) != VK_SUCCESS)
      throw std::runtime_error("vkAllocateMemory failed");
    vkBindBufferMemory(device, buf, mem, 0);
  }

  void createBuffers() {
    VkDeviceSize vsize = sizeof(cubeVertices);
    VkDeviceSize isize = sizeof(cubeIndices);
    createBuffer(vsize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 vertexBuffer, vertexMemory);
    createBuffer(isize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 indexBuffer, indexMemory);
    void *p;
    vkMapMemory(device, vertexMemory, 0, vsize, 0, &p);
    memcpy(p, cubeVertices.data(), (size_t)vsize);
    vkUnmapMemory(device, vertexMemory);
    vkMapMemory(device, indexMemory, 0, isize, 0, &p);
    memcpy(p, cubeIndices.data(), (size_t)isize);
    vkUnmapMemory(device, indexMemory);
  }

  void createUniforms() {
    uniformBuffers.resize(swapImages.size());
    uniformMemories.resize(swapImages.size());
    for (size_t i = 0; i < swapImages.size(); ++i) {
      createBuffer(sizeof(UBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   uniformBuffers[i], uniformMemories[i]);
    }
  }

  void createDescriptorPoolAndSets() {
    VkDescriptorPoolSize pool{};
    pool.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool.descriptorCount = (uint32_t)uniformBuffers.size();
    VkDescriptorPoolCreateInfo pci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = (uint32_t)uniformBuffers.size();
    pci.poolSizeCount = 1;
    pci.pPoolSizes = &pool;
    if (vkCreateDescriptorPool(device, &pci, nullptr, &descPool) != VK_SUCCESS)
      throw std::runtime_error("vkCreateDescriptorPool failed");
    std::vector<VkDescriptorSetLayout> layouts(uniformBuffers.size(),
                                               descSetLayout);
    VkDescriptorSetAllocateInfo ai{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = descPool;
    ai.descriptorSetCount = (uint32_t)layouts.size();
    ai.pSetLayouts = layouts.data();
    descSets.resize(layouts.size());
    if (vkAllocateDescriptorSets(device, &ai, descSets.data()) != VK_SUCCESS)
      throw std::runtime_error("vkAllocateDescriptorSets failed");
    for (size_t i = 0; i < descSets.size(); ++i) {
      VkDescriptorBufferInfo bi{};
      bi.buffer = uniformBuffers[i];
      bi.offset = 0;
      bi.range = sizeof(UBO);
      VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      write.dstSet = descSets[i];
      write.dstBinding = 0;
      write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      write.descriptorCount = 1;
      write.pBufferInfo = &bi;
      vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }
  }

  void createCommandBuffers() {
    cmdBufs.resize(framebuffers.size());
    VkCommandBufferAllocateInfo ai{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = cmdPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = (uint32_t)cmdBufs.size();
    if (vkAllocateCommandBuffers(device, &ai, cmdBufs.data()) != VK_SUCCESS)
      throw std::runtime_error("vkAllocateCommandBuffers failed");
  }

  void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imgIndex) {
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmd, &bi);
    VkClearValue clear{};
    clear.color = {{0.05f, 0.07f, 0.1f, 1.0f}};
    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = renderPass;
    rp.framebuffer = framebuffers[imgIndex];
    rp.renderArea = {{0, 0}, swapExtent};
    rp.clearValueCount = 1;
    rp.pClearValues = &clear;
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    VkDeviceSize offs = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer, &offs);
    vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT16);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout, 0, 1, &descSets[imgIndex], 0,
                            nullptr);
    vkCmdDrawIndexed(cmd, (uint32_t)cubeIndices.size(), 1, 0, 0, 0);
    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
  }

  void createSync() {
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateSemaphore(device, &si, nullptr, &imageAvailable);
    vkCreateSemaphore(device, &si, nullptr, &renderFinished);
    vkCreateFence(device, &fi, nullptr, &inFlight);
  }

  void updateUniforms(float t) {
    Mat4 proj = matPerspective(
        45.0f * (3.14159f / 180.f),
        (float)swapExtent.width / (float)swapExtent.height, 0.1f, 100.0f);
    Mat4 view = matTranslate(0, 0, -5);
    Mat4 model = matMul(matRotateY(t * 0.9f), matRotateX(t * 0.5f));
    Mat4 mvp = matMul(proj, matMul(view, model));
    for (size_t i = 0; i < uniformBuffers.size(); ++i) {
      UBO u{};
      u.mvp = mvp;
      void *p;
      vkMapMemory(device, uniformMemories[i], 0, sizeof(UBO), 0, &p);
      memcpy(p, &u, sizeof(UBO));
      vkUnmapMemory(device, uniformMemories[i]);
    }
  }

  void recreateCommandBuffer(uint32_t idx) {
    // Record on demand; buffers were allocated up front
    recordCommandBuffer(cmdBufs[idx], idx);
  }

  void drawFrame() {
    vkWaitForFences(device, 1, &inFlight, VK_TRUE, UINT64_MAX);
    vkResetFences(device, 1, &inFlight);

    uint32_t imageIndex;
    VkResult res =
        vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, imageAvailable,
                              VK_NULL_HANDLE, &imageIndex);
    if (res != VK_SUCCESS)
      throw std::runtime_error("vkAcquireNextImageKHR failed");

    vkResetCommandBuffer(cmdBufs[imageIndex], 0);
    recreateCommandBuffer(imageIndex);

    VkPipelineStageFlags waitStage =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &imageAvailable;
    submit.pWaitDstStageMask = &waitStage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmdBufs[imageIndex];
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &renderFinished;
    if (vkQueueSubmit(graphicsQueue, 1, &submit, inFlight) != VK_SUCCESS)
      throw std::runtime_error("vkQueueSubmit failed");

    VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &renderFinished;
    present.swapchainCount = 1;
    present.pSwapchains = &swapchain;
    present.pImageIndices = &imageIndex;
    vkQueuePresentKHR(presentQueue, &present);
  }
};

int main() {
  try {
    App().run();
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
