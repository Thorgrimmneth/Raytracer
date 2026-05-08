#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <vector>
#include <cstring>
#include "renderingGPU/hello_cuda.hpp"
#include "defines.hpp"

namespace RT
{	
    class Window
    {
      public:
        Window() = delete;
        ~Window();
        Window(int width, int height);
        unsigned char* cumulativeRendering(Vec3f sunDir, int width, int height);

        private:

            GLuint createShader(GLenum type, const char* source);
            void initShaders();
            void initTexture(int width, int height);
            void initQuad();
            void draw();
            
        int width  = 1920;
        int height = 1080;
        
        Renderer renderer;

        cudaGraphicsResource* cudaTextureResource = nullptr;
        
        GLuint texture;
        GLuint vao;
        GLuint vbo;
        GLuint shaderProgram;

        unsigned char* image = nullptr;

        const char* vertexShaderSource = R"(
        #version 450 core

        layout(location = 0) in vec2 aPos;
        layout(location = 1) in vec2 aUV;

        out vec2 uv;

        void main()
        {
            uv = aUV;
            gl_Position = vec4(aPos, 0.0, 1.0);
        }
        )";

        const char* fragmentShaderSource = R"(
        #version 450 core

        in vec2 uv;

        out vec4 FragColor;

        uniform sampler2D uTexture;

        void main()
        {
            vec2 flippedUV = vec2(uv.x, 1.0 - uv.y);

            vec3 color = texture(uTexture, flippedUV).rgb;

            FragColor = vec4(color, 1.0);
        }
        )";
    };

}