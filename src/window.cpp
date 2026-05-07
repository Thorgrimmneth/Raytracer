#include <algorithm>
#include "window.hpp"

namespace RT
{	Window::Window(int p_width, int p_height)
    : width(p_width),
      height(p_height)
{
}

GLuint Window::createShader(GLenum type, const char* source)
{
    GLuint shader = glCreateShader(type);

    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    if (!success)
    {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);

        std::cout << "Shader compilation error:\n"
                  << infoLog << std::endl;
    }

    return shader;
}

void Window::initShaders()
{
    GLuint vertexShader =
        createShader(GL_VERTEX_SHADER, vertexShaderSource);

    GLuint fragmentShader =
        createShader(GL_FRAGMENT_SHADER, fragmentShaderSource);

    shaderProgram = glCreateProgram();

    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);

    glLinkProgram(shaderProgram);

    int success;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);

    if (!success)
    {
        char infoLog[512];
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);

        std::cout << "Program linking error:\n"
                  << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void Window::initTexture(int width, int height)
{
    glGenTextures(1, &texture);

    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(GL_TEXTURE_2D,
                    GL_TEXTURE_MIN_FILTER,
                    GL_LINEAR);

    glTexParameteri(GL_TEXTURE_2D,
                    GL_TEXTURE_MAG_FILTER,
                    GL_LINEAR);

    glTexParameteri(GL_TEXTURE_2D,
                    GL_TEXTURE_WRAP_S,
                    GL_CLAMP_TO_EDGE);

    glTexParameteri(GL_TEXTURE_2D,
                    GL_TEXTURE_WRAP_T,
                    GL_CLAMP_TO_EDGE);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB8,
        width,
        height,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        nullptr
    );

    glBindTexture(GL_TEXTURE_2D, 0);
}

void Window::initQuad()
{
    float quadVertices[] =
    {
        // positions   // uv
        -1.f, -1.f, 0.f, 0.f,
         1.f, -1.f, 1.f, 0.f,
         1.f,  1.f, 1.f, 1.f,

        -1.f, -1.f, 0.f, 0.f,
         1.f,  1.f, 1.f, 1.f,
        -1.f,  1.f, 0.f, 1.f
    };

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glBufferData(
        GL_ARRAY_BUFFER,
        sizeof(quadVertices),
        quadVertices,
        GL_STATIC_DRAW
    );

    glEnableVertexAttribArray(0);

    glVertexAttribPointer(
        0,
        2,
        GL_FLOAT,
        GL_FALSE,
        4 * sizeof(float),
        (void*)0
    );

    glEnableVertexAttribArray(1);

    glVertexAttribPointer(
        1,
        2,
        GL_FLOAT,
        GL_FALSE,
        4 * sizeof(float),
        (void*)(2 * sizeof(float))
    );

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void Window::uploadTexture(
    unsigned char* framebuffer,
    int width,
    int height)
{
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexSubImage2D(
        GL_TEXTURE_2D,
        0,
        0,
        0,
        width,
        height,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        framebuffer
    );
}

void Window::draw()
{
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(shaderProgram);
    glUniform1i(
    glGetUniformLocation(shaderProgram, "uTexture"),
        0
    );
    glBindVertexArray(vao);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);

    glDrawArrays(GL_TRIANGLES, 0, 6);
}

unsigned char* Window::cumulativeRendering(Vec3f sunDir, int width, int height)
{
    glfwInit();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);

    glfwWindowHint(
        GLFW_OPENGL_PROFILE,
        GLFW_OPENGL_CORE_PROFILE
    );
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);


    GLFWmonitor* targetMonitor = nullptr;

    int count;
    GLFWmonitor** monitors = glfwGetMonitors(&count);

    for (int i = 0; i < count; i++)
    {
        const GLFWvidmode* mode =
            glfwGetVideoMode(monitors[i]);

        if (mode->refreshRate >= 165)
        {
            targetMonitor = monitors[i];
            break;
        }
    }

    if (!targetMonitor)
    {
        targetMonitor = glfwGetPrimaryMonitor();
    }

    const GLFWvidmode* mode = glfwGetVideoMode(targetMonitor);
    std::cout
    << "Refresh rate: "
    << mode->refreshRate
    << " Hz"
    << std::endl;
    int windowWidth = width;
    int windowHeight = height;

    if (mode)
    {
        windowWidth = std::min(windowWidth, mode->width);
        windowHeight = std::min(windowHeight, mode->height);
    }

    GLFWwindow* window = glfwCreateWindow(
        windowWidth,
        windowHeight,
        "Path Tracer",
        nullptr,
        nullptr
    );

    if (!window)
    {
        std::cout << "Failed to create window" << std::endl;
        return nullptr;
    }

    if (mode)
    {
        int monitorX, monitorY;
        glfwGetMonitorPos(targetMonitor, &monitorX, &monitorY);
        int posX = monitorX + (mode->width - windowWidth) / 2;
        int posY = monitorY + (mode->height - windowHeight) / 2;
        glfwSetWindowPos(window, posX, posY);
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); // VSync : 0 = off, 1 = on
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return nullptr;
    }
    std::cout << glGetString(GL_VERSION) << std::endl;

    int framebufferWidth, framebufferHeight;
    glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
    glViewport(0, 0, framebufferWidth, framebufferHeight);
    glfwShowWindow(window);

    initShaders();
    initTexture(width, height);
    initQuad();
    Renderer renderer;
    renderer.init(width, height, sunDir.x, sunDir.y, sunDir.z);
    glClearColor(0.f, 0.f, 0.f, 1.f);
    double lastTime = glfwGetTime();

    int frames = 0;

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        renderer.renderFrame();

        frames++;

        double currentTime = glfwGetTime();

        if (currentTime - lastTime >= 1.0)
        {
            double fps =
                frames / (currentTime - lastTime);

            std::string title =
                "Path Tracer | FPS: " +
                std::to_string((int)fps) +
                " | SPP: " +
                std::to_string(renderer.getFrameNumber());

            glfwSetWindowTitle(window, title.c_str());

            frames = 0;
            lastTime = currentTime;
        }

        unsigned char* framebuffer =
            renderer.getFramebuffer();

        uploadTexture(framebuffer, width, height);

        draw();

        glfwSwapBuffers(window);
    }
    
    unsigned char* finalImage = new unsigned char[width * height * 3];

    memcpy(
        finalImage,
        renderer.getFramebuffer(),
        width * height * 3 * sizeof(unsigned char)
    );

    glDeleteTextures(1, &texture);

    glDeleteBuffers(1, &vbo);

    glDeleteVertexArrays(1, &vao);

    glDeleteProgram(shaderProgram);

    glfwDestroyWindow(window);
    glfwTerminate();

    return finalImage;
}
}// namespace RT