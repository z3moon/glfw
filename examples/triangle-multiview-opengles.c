//========================================================================
// OpenGL ES 2.0 triangle example
// Copyright (c) Camilla Löwy <elmindreda@glfw.org>
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would
//    be appreciated but is not required.
//
// 2. Altered source versions must be plainly marked as such, and must not
//    be misrepresented as being the original software.
//
// 3. This notice may not be removed or altered from any source
//    distribution.
//
//========================================================================

#define GLAD_GLES2_IMPLEMENTATION
#include <glad/gles2_30.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "linmath.h"

#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <stdbool.h>

#define M_PI       3.14159265358979323846   // pi
#define LOGI(...) fprintf(stdout, __VA_ARGS__)
#define LOGE(...) fprintf(stderr, __VA_ARGS__)
#define GL_CHECK(x)                                         \
    x;                                                      \
    {                                                       \
        GLenum glError = glGetError();                      \
        if (glError != GL_NO_ERROR) {                       \
            LOGE("glGetError() = %i (0x%.8x) at %s:%i\n",   \
                glError, glError, __FILE__, __LINE__);      \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    }


//---------------------------------------------------------------
// Test variables
#define MIP_NUMS     4 // the number of mipmaps for the multiview FBO
#define MIP_INDEX    0 // the mipmap index to be used. this should be 0 <= mipIndex < mipNums.
static_assert(0 <= MIP_INDEX && MIP_INDEX < MIP_NUMS);
//---------------------------------------------------------------


GLuint fboWidth = 1920;
GLuint fboHeight = 1080;
GLuint screenWidth;
GLuint screenHeight;
GLuint frameBufferTextureId;
GLuint frameBufferDepthTextureId;
GLuint frameBufferObjectId[MIP_NUMS];

GLuint multiviewProgram;
GLuint multiviewVertexLocation;
GLuint multiviewVertexNormalLocation;
GLuint multiviewModelViewProjectionLocation;
GLuint multiviewModelLocation;

GLuint texturedQuadProgram;
GLuint texturedQuadVertexLocation;
GLuint texturedQuadLowResTexCoordLocation;
GLuint texturedQuadHighResTexCoordLocation;
GLuint texturedQuadSamplerLocation;
GLuint texturedQuadLayerIndexLocation;

mat4x4 projectionMatrix[4];
mat4x4 viewMatrix[4];
mat4x4 viewProjectionMatrix[4];
mat4x4 modelViewProjectionMatrix[4];
mat4x4 modelMatrix;
float angle = 0;

typedef void (*PFNGLFRAMEBUFFERTEXTUREMULTIVIEWOVR)(GLenum, GLenum, GLuint, GLint, GLint, GLsizei);
PFNGLFRAMEBUFFERTEXTUREMULTIVIEWOVR glFramebufferTextureMultiviewOVR;

/* Multiview vertexShader */
static const char multiviewVertexShader[] =
"#version 300 es\n"
"#extension GL_OVR_multiview : enable\n"
"layout(num_views = 4) in;\n"

"in vec3 vertexPosition;\n"
"in vec3 vertexNormal;\n"
"uniform mat4 modelViewProjection[4];\n"
"uniform mat4 model;\n"
"out vec3 v_normal;\n"

"void main()\n"
"{\n"
"    gl_Position = modelViewProjection[gl_ViewID_OVR] * vec4(vertexPosition, 1.0);\n"
"    v_normal = (model * vec4(vertexNormal, 0.0f)).xyz;\n"
"}\n";

/* Multiview fragmentShader */
static const char multiviewFragmentShader[] =
"#version 300 es\n"
"precision highp float;\n"

"in vec3 v_normal;\n"
"out vec4 f_color;\n"

"vec3 light(vec3 n, vec3 l, vec3 c)\n"
"{\n"
"    float ndotl = max(dot(n, l), 0.0);\n"
"    return ndotl * c;\n"
"}\n"

"void main()\n"
"{\n"
"    vec3 n = normalize(v_normal);\n"
"    f_color.rgb = vec3(0.0);\n"
"    f_color.rgb += light(n, normalize(vec3(1.0, 0.0, 0.0)), vec3(0.23, 0.91, 0.12));\n"
"    f_color.rgb += light(n, normalize(vec3(-1.0, 0.0, 0.0)), vec3(0.8, 0.23, 0.35));\n"

"    f_color.a = 1.0;\n"
"}\n";

/* Textured quad vertexShader */
static const char  texturedQuadVertexShader[] =
"#version 300 es\n"
"in vec3 attributePosition;\n"
"in vec2 attributeLowResTexCoord;\n"
"in vec2 attributeHighResTexCoord;\n"
"out vec2 vLowResTexCoord;\n"
"out vec2 vHighResTexCoord;\n"
"void main()\n"
"{\n"
"    vLowResTexCoord = attributeLowResTexCoord;\n"
"    vHighResTexCoord = attributeHighResTexCoord;\n"
"    gl_Position = vec4(attributePosition, 1.0);\n"
"}\n";

/* Textured quad fragmentShader */
static const char texturedQuadFragmentShader[] =
"#version 300 es\n"
"precision mediump float;\n"
"precision mediump int;\n"
"precision mediump sampler2DArray;\n"
"in vec2 vLowResTexCoord;\n"
"in vec2 vHighResTexCoord;\n"
"out vec4 fragColor;\n"
"uniform sampler2DArray tex;\n"
"uniform int layerIndex;\n"
"void main()\n"
"{\n"
"    vec4 lowResSample = texture(tex, vec3(vLowResTexCoord, layerIndex));\n"
"    vec4 highResSample = texture(tex, vec3(vHighResTexCoord, layerIndex + 2));\n"
"    // Using squared distance to middle of screen for interpolating.\n"
"    vec2 distVec = vec2(0.5) - vHighResTexCoord;\n"
"    float squaredDist = dot(distVec, distVec);\n"
"    // Using the high res texture when distance from center is less than 0.5 in texture coordinates (0.25 is 0.5 squared).\n"
"    // When the distance is less than 0.2 (0.04 is 0.2 squared), only the high res texture will be used.\n"
"    float lerpVal = smoothstep(-0.25, -0.04, -squaredDist);\n"
"    fragColor = mix(lowResSample, highResSample, lerpVal);\n"
"}\n";

/* Vertices for cube drawn with multiview. */
GLfloat multiviewVertices[] =
{   // Front face
    -1.0f, -1.0f, 1.0f,
     1.0f, -1.0f, 1.0f,
     1.0f,  1.0f, 1.0f,
    -1.0f,  1.0f, 1.0f,

    // Right face
     1.0f, -1.0f,  1.0f,
     1.0f, -1.0f, -1.0f,
     1.0f,  1.0f, -1.0f,
     1.0f,  1.0f,  1.0f,

     // Back face
      1.0f, -1.0f,  -1.0f,
     -1.0f, -1.0f,  -1.0f,
     -1.0f,  1.0f,  -1.0f,
      1.0f,  1.0f,  -1.0f,

      // Left face
      -1.0f, -1.0f, -1.0f,
      -1.0f, -1.0f,  1.0f,
      -1.0f,  1.0f,  1.0f,
      -1.0f,  1.0f, -1.0f,

      // Top face
       1.0f,  1.0f,  1.0f,
       1.0f,  1.0f, -1.0f,
      -1.0f,  1.0f, -1.0f,
      -1.0f,  1.0f,  1.0f,

      // Bottom face
      -1.0f, -1.0f, -1.0f,
       1.0f, -1.0f, -1.0f,
       1.0f, -1.0f,  1.0f,
      -1.0f, -1.0f,  1.0f,
};

/* Normals for cube drawn with multiview. */
GLfloat multiviewNormals[] =
{
    // Front face
    0.0f,  0.0f,  1.0f,
    0.0f,  0.0f,  1.0f,
    0.0f,  0.0f,  1.0f,
    0.0f,  0.0f,  1.0f,

    // Right face
    1.0f,  0.0f, 0.0f,
    1.0f,  0.0f, 0.0f,
    1.0f,  0.0f, 0.0f,
    1.0f,  0.0f, 0.0f,

    // Back face
    0.0f,  0.0f, -1.0f,
    0.0f,  0.0f, -1.0f,
    0.0f,  0.0f, -1.0f,
    0.0f,  0.0f, -1.0f,

    // Left face
    -1.0f,  0.0f, 0.0f,
    -1.0f,  0.0f, 0.0f,
    -1.0f,  0.0f, 0.0f,
    -1.0f,  0.0f, 0.0f,

    // Top face
    0.0f,  1.0f, 0.0f,
    0.0f,  1.0f, 0.0f,
    0.0f,  1.0f, 0.0f,
    0.0f,  1.0f, 0.0f,

    // Bottom face
    0.0f, -1.0f, 0.0f,
    0.0f, -1.0f, 0.0f,
    0.0f, -1.0f, 0.0f,
    0.0f, -1.0f, 0.0f
};

/* Indices for cube drawn with multiview. */
GLushort multiviewIndices[] =
{
    //Front face
    0, 1, 2,
    0, 2, 3,

    // Right face
    4, 5, 6,
    4, 6, 7,

    // Back face
    8, 9, 10,
    8, 10, 11,

    // Left face
    12, 13, 14,
    12, 14, 15,

    // Top face
    16, 17, 18,
    16, 18, 19,

    // Bottom face
    20, 21, 22,
    20, 22, 23
};

/* Textured quad geometry */
float texturedQuadCoordinates[] =
{
    -1.0f, -1.0f, 0.0f,
     1.0f, -1.0f, 0.0f,
     1.0f,  1.0f, 0.0f,

    -1.0f, -1.0f, 0.0f,
     1.0f,  1.0f, 0.0f,
    -1.0f,  1.0f, 0.0f
};

/* Textured quad low resolution texture coordinates */
float texturedQuadLowResTexCoordinates[] =
{
    0, 0,
    1, 0,
    1, 1,

    0, 0,
    1, 1,
    0, 1
};

/* Textured quad high resolution texture coordinates */
float texturedQuadHighResTexCoordinates[] =
{
    -0.5, -0.5,
     1.5, -0.5,
     1.5,  1.5,

    -0.5, -0.5,
     1.5,  1.5,
    -0.5,  1.5
};

GLuint loadShader(GLenum shaderType, const char* shaderSource)
{
    GLuint shader = GL_CHECK(glCreateShader(shaderType));
    if (shader != 0)
    {
        GL_CHECK(glShaderSource(shader, 1, &shaderSource, NULL));
        GL_CHECK(glCompileShader(shader));
        GLint compiled = 0;
        GL_CHECK(glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled));
        if (compiled != GL_TRUE)
        {
            GLint infoLen = 0;
            GL_CHECK(glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen));

            if (infoLen > 0)
            {
                char* logBuffer = (char*)malloc(infoLen);

                if (logBuffer != NULL)
                {
                    GL_CHECK(glGetShaderInfoLog(shader, infoLen, NULL, logBuffer));
                    LOGE("Could not Compile Shader %d:\n%s\n", shaderType, logBuffer);
                    free(logBuffer);
                    logBuffer = NULL;
                }

                GL_CHECK(glDeleteShader(shader));
                shader = 0;
            }
        }
    }

    return shader;
}

GLuint createProgram(const char* vertexSource, const char* fragmentSource)
{
    GLuint vertexShader = loadShader(GL_VERTEX_SHADER, vertexSource);
    if (vertexShader == 0)
    {
        return 0;
    }

    GLuint fragmentShader = loadShader(GL_FRAGMENT_SHADER, fragmentSource);
    if (fragmentShader == 0)
    {
        return 0;
    }

    GLuint program = GL_CHECK(glCreateProgram());

    if (program != 0)
    {
        GL_CHECK(glAttachShader(program, vertexShader));
        GL_CHECK(glAttachShader(program, fragmentShader));
        GL_CHECK(glLinkProgram(program));
        GLint linkStatus = GL_FALSE;
        GL_CHECK(glGetProgramiv(program, GL_LINK_STATUS, &linkStatus));
        if (linkStatus != GL_TRUE)
        {
            GLint bufLength = 0;
            GL_CHECK(glGetProgramiv(program, GL_INFO_LOG_LENGTH, &bufLength));
            if (bufLength > 0)
            {
                char* logBuffer = (char*)malloc(bufLength);

                if (logBuffer != NULL)
                {
                    GL_CHECK(glGetProgramInfoLog(program, bufLength, NULL, logBuffer));
                    LOGE("Could not link program:\n%s\n", logBuffer);
                    free(logBuffer);
                    logBuffer = NULL;
                }
            }
            GL_CHECK(glDeleteProgram(program));
            program = 0;
        }
    }
    return program;
}

bool setupFBO(int width, int height)
{
    // Create array texture
    GL_CHECK(glGenTextures(1, &frameBufferTextureId));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D_ARRAY, frameBufferTextureId));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    GL_CHECK(glTexStorage3D(GL_TEXTURE_2D_ARRAY, MIP_NUMS, GL_RGBA8, width, height, 4));

    /* Create array depth texture */
    GL_CHECK(glGenTextures(1, &frameBufferDepthTextureId));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D_ARRAY, frameBufferDepthTextureId));
    GL_CHECK(glTexStorage3D(GL_TEXTURE_2D_ARRAY, MIP_NUMS, GL_DEPTH_COMPONENT24, width, height, 4));

    /* Create framebuffers */
    for (int i = 0; i < MIP_NUMS; ++i) {
        /* Initialize FBO. */
        GL_CHECK(glGenFramebuffers(1, &frameBufferObjectId[i]));
        /* Bind our framebuffer for rendering. */
        GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, frameBufferObjectId[i]));
        /* Attach texture for the current mipmap level to the framebuffer. */
        GL_CHECK(glFramebufferTextureMultiviewOVR(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
            frameBufferTextureId, i, 0, 4));
        /* Attach depth texture for the current mipmap level to the framebuffer. */
        GL_CHECK(glFramebufferTextureMultiviewOVR(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
            frameBufferDepthTextureId, i, 0, 4));
    }

    /* Check FBO is OK. */
    GLenum result = GL_CHECK(glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER));
    if (result != GL_FRAMEBUFFER_COMPLETE)
    {
        LOGE("Framebuffer incomplete at %s:%i\n", __FILE__, __LINE__);
        /* Unbind framebuffer. */
        GL_CHECK(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
        return false;
    }
    return true;
}

bool setupGraphics(int width, int height)
{
    /*
     * Make sure the required multiview extension is present.
     */
    const GLubyte* extensions = GL_CHECK(glGetString(GL_EXTENSIONS));
    const char* found_extension = strstr((const char*)extensions, "GL_OVR_multiview");
    if (NULL == found_extension)
    {
        LOGI("OpenGL ES 3.0 implementation does not support GL_OVR_multiview extension.\n");
        exit(EXIT_FAILURE);
    }
    else
    {
        glFramebufferTextureMultiviewOVR =
            (PFNGLFRAMEBUFFERTEXTUREMULTIVIEWOVR)glfwGetProcAddress("glFramebufferTextureMultiviewOVR");
        if (!glFramebufferTextureMultiviewOVR)
        {
            LOGI("Can not get proc address for glFramebufferTextureMultiviewOVR.\n");
            exit(EXIT_FAILURE);
        }
    }

    /* Enable culling and depth testing. */
    GL_CHECK(glDisable(GL_CULL_FACE));
    GL_CHECK(glEnable(GL_DEPTH_TEST));
    GL_CHECK(glDepthFunc(GL_LEQUAL));

    /* Setting screen width and height for use when rendering. */
    screenWidth = width;
    screenHeight = height;

    if (!setupFBO(fboWidth, fboHeight))
    {
        LOGE("Could not create multiview FBO");
        return false;
    }

    /* Creating program for drawing textured quad. */
    texturedQuadProgram = createProgram(texturedQuadVertexShader, texturedQuadFragmentShader);
    if (texturedQuadProgram == 0)
    {
        LOGE("Could not create textured quad program");
        return false;
    }

    /* Get attribute and uniform locations for textured quad program. */
    texturedQuadVertexLocation = GL_CHECK(glGetAttribLocation(texturedQuadProgram, "attributePosition"));
    texturedQuadLowResTexCoordLocation = GL_CHECK(glGetAttribLocation(texturedQuadProgram, "attributeLowResTexCoord"));
    texturedQuadHighResTexCoordLocation = GL_CHECK(glGetAttribLocation(texturedQuadProgram, "attributeHighResTexCoord"));
    texturedQuadSamplerLocation = GL_CHECK(glGetUniformLocation(texturedQuadProgram, "tex"));
    texturedQuadLayerIndexLocation = GL_CHECK(glGetUniformLocation(texturedQuadProgram, "layerIndex"));

    /* Creating program for drawing object with multiview. */
    multiviewProgram = createProgram(multiviewVertexShader, multiviewFragmentShader);
    if (multiviewProgram == 0)
    {
        LOGE("Could not create multiview program");
        return false;
    }

    /* Get attribute and uniform locations for multiview program. */
    multiviewVertexLocation = GL_CHECK(glGetAttribLocation(multiviewProgram, "vertexPosition"));
    multiviewVertexNormalLocation = GL_CHECK(glGetAttribLocation(multiviewProgram, "vertexNormal"));
    multiviewModelViewProjectionLocation = GL_CHECK(glGetUniformLocation(multiviewProgram, "modelViewProjection"));
    multiviewModelLocation = GL_CHECK(glGetUniformLocation(multiviewProgram, "model"));

    /*
     * Set up the perspective matrices for each view. Rendering is done twice in each eye position with different
     * field of view. The narrower field of view should give half the size for the near plane in order to
     * render the center of the scene at a higher resolution. The resulting high resolution and low resolution
     * images will later be interpolated to create an image with higher resolution in the center of the screen
     * than on the outer parts of the screen.
     * 1.5707963268 rad = 90 degrees.
     * 0.9272952188 rad = 53.1301024 degrees. This angle gives half the size for the near plane.
     */
    mat4x4_perspective(projectionMatrix[0], 1.5707963268f, (float)width / (float)height, 0.1f, 100.f);
    mat4x4_perspective(projectionMatrix[1], 1.5707963268f, (float)width / (float)height, 0.1f, 100.f);
    mat4x4_perspective(projectionMatrix[2], 0.9272952188f, (float)width / (float)height, 0.1f, 100.f);
    mat4x4_perspective(projectionMatrix[3], 0.9272952188f, (float)width / (float)height, 0.1f, 100.f);

    GL_CHECK(glViewport(0, 0, width, height));

    /* Setting up model view matrices for each of the */
    vec3 leftCameraPos = { -1.5f, 2.0f, 5.0f };
    vec3 rightCameraPos = { 1.5f, -2.0f, 5.0f };
    vec3 lookAt = { 0.0f, 0.0f, 0.0f };
    vec3 upVec = { 0.0f, 1.0f, 0.0f };
    mat4x4_look_at(viewMatrix[0], leftCameraPos, lookAt, upVec);
    mat4x4_look_at(viewMatrix[1], rightCameraPos, lookAt, upVec);
    mat4x4_look_at(viewMatrix[2], leftCameraPos, lookAt, upVec);
    mat4x4_look_at(viewMatrix[3], rightCameraPos, lookAt, upVec);

    mat4x4_mul(viewProjectionMatrix[0], projectionMatrix[0], viewMatrix[0]);
    mat4x4_mul(viewProjectionMatrix[1], projectionMatrix[1], viewMatrix[1]);
    mat4x4_mul(viewProjectionMatrix[2], projectionMatrix[2], viewMatrix[2]);
    mat4x4_mul(viewProjectionMatrix[3], projectionMatrix[3], viewMatrix[3]);

    return true;
}

void renderToFBO(int width, int height)
{
    /* Rendering to FBO. */
    GL_CHECK(glViewport(0, 0, width, height));

    /* Bind our framebuffer for rendering. */
    GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, frameBufferObjectId[MIP_INDEX]));

    // z3moon: test to see if this crashes
    //GL_CHECK(glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE));
    //GL_CHECK(glDepthMask(GL_TRUE))
    //GLfloat clr[4] = { 0.0, 0.0, 0.0, 1.0 };
    //GL_CHECK(glClearBufferfv(GL_COLOR, 0, clr));
    //GL_CHECK(glClearBufferfv(GL_DEPTH, 0, clr));
    //---

    GL_CHECK(glClearColor(0.5f, 0.5f, 0.5f, 1.0f));
    GL_CHECK(glClearDepthf(1.0));
    GL_CHECK(glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));

    /* Rotating the cube. */
    float angleRad = M_PI * angle / 180.0f;
    mat4x4_identity(modelMatrix);
    mat4x4_rotate_Y(modelMatrix, modelMatrix, angleRad);
    mat4x4_rotate_X(modelMatrix, modelMatrix, angleRad * 1.5f);

    mat4x4_mul(modelViewProjectionMatrix[0], viewProjectionMatrix[0], modelMatrix);
    mat4x4_mul(modelViewProjectionMatrix[1], viewProjectionMatrix[1], modelMatrix);
    mat4x4_mul(modelViewProjectionMatrix[2], viewProjectionMatrix[2], modelMatrix);
    mat4x4_mul(modelViewProjectionMatrix[3], viewProjectionMatrix[3], modelMatrix);

    GL_CHECK(glUseProgram(multiviewProgram));

    /* Upload vertex attributes. */
    GL_CHECK(glVertexAttribPointer(multiviewVertexLocation, 3, GL_FLOAT, GL_FALSE, 0, multiviewVertices));
    GL_CHECK(glEnableVertexAttribArray(multiviewVertexLocation));
    GL_CHECK(glVertexAttribPointer(multiviewVertexNormalLocation, 3, GL_FLOAT, GL_FALSE, 0, multiviewNormals));
    GL_CHECK(glEnableVertexAttribArray(multiviewVertexNormalLocation));

    /* Upload model view projection matrices. */
    GL_CHECK(glUniformMatrix4fv(multiviewModelViewProjectionLocation, 4, GL_FALSE, (const GLfloat*)&modelViewProjectionMatrix[0]));
    GL_CHECK(glUniformMatrix4fv(multiviewModelLocation, 1, GL_FALSE, (const GLfloat*)&modelMatrix));

    /* Draw a cube. */
    GL_CHECK(glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, multiviewIndices));

    /* Draw a translated cube. */
    mat4x4 translatedModelMatrix;
    mat4x4_translate(translatedModelMatrix, -3.5f, 0.f, 0.f);
    mat4x4_mul(translatedModelMatrix, translatedModelMatrix, modelMatrix);
    mat4x4_mul(modelViewProjectionMatrix[0], viewProjectionMatrix[0], translatedModelMatrix);
    mat4x4_mul(modelViewProjectionMatrix[1], viewProjectionMatrix[1], translatedModelMatrix);
    mat4x4_mul(modelViewProjectionMatrix[2], viewProjectionMatrix[2], translatedModelMatrix);
    mat4x4_mul(modelViewProjectionMatrix[3], viewProjectionMatrix[3], translatedModelMatrix);
    GL_CHECK(glUniformMatrix4fv(multiviewModelViewProjectionLocation, 4, GL_FALSE, (const GLfloat*)&modelViewProjectionMatrix[0]));
    GL_CHECK(glUniformMatrix4fv(multiviewModelLocation, 1, GL_FALSE, (const GLfloat*)translatedModelMatrix));
    GL_CHECK(glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, multiviewIndices));

    /* Draw another translated cube. */
    mat4x4_translate(translatedModelMatrix, 3.5f, 0.f, 0.f);
    mat4x4_mul(translatedModelMatrix, translatedModelMatrix, modelMatrix);
    mat4x4_mul(modelViewProjectionMatrix[0], viewProjectionMatrix[0], translatedModelMatrix);
    mat4x4_mul(modelViewProjectionMatrix[1], viewProjectionMatrix[1], translatedModelMatrix);
    mat4x4_mul(modelViewProjectionMatrix[2], viewProjectionMatrix[2], translatedModelMatrix);
    mat4x4_mul(modelViewProjectionMatrix[3], viewProjectionMatrix[3], translatedModelMatrix);
    GL_CHECK(glUniformMatrix4fv(multiviewModelViewProjectionLocation, 4, GL_FALSE, (const GLfloat*)&modelViewProjectionMatrix[0]));
    GL_CHECK(glUniformMatrix4fv(multiviewModelLocation, 1, GL_FALSE, (const GLfloat*)translatedModelMatrix));
    GL_CHECK(glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, multiviewIndices));

    //angle += 0.1;
    //if (angle > 360)
    //{
    //    angle -= 360;
    //}

    /* Go back to the backbuffer for rendering to the screen. */
    GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, 0));
}

void renderFrame()
{
    /*
     * Calculate the size of the mipmap to be used.
     */
    GLuint fboMipWidth = fboWidth / (1 << MIP_INDEX);
    GLuint fboMipHeight = fboHeight / (1 << MIP_INDEX);

    /*
     * Render the scene to the multiview texture. This will render to 4 different layers of the texture,
     * using different projection and view matrices for each layer.
     */
    renderToFBO(fboMipWidth, fboMipHeight);

    GL_CHECK(glClearColor(0.0f, 0.0f, 0.0f, 1.0f));
    GL_CHECK(glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));

    /*
     * Render the multiview texture layers to separate viewports. Each viewport corresponds to one eye,
     * and will use two different texture layers from the multiview texture, one with a wide field of view
     * and one with a narrow field of view.
     */
    for (int i = 0; i < 2; i++)
    {
        glViewport(i * screenWidth / 2, 0, screenWidth / 2, screenHeight);

        /* Use the texture array that was drawn to using multiview. */
        GL_CHECK(glActiveTexture(GL_TEXTURE0));
        GL_CHECK(glBindTexture(GL_TEXTURE_2D_ARRAY, frameBufferTextureId));

        /* Use the specified mipmap level for this array texture. */
        GL_CHECK(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_BASE_LEVEL, MIP_INDEX));
        GL_CHECK(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAX_LEVEL, MIP_INDEX));

        GL_CHECK(glUseProgram(texturedQuadProgram));

        /* Upload vertex attributes. */
        GL_CHECK(glVertexAttribPointer(texturedQuadVertexLocation, 3, GL_FLOAT, GL_FALSE, 0, texturedQuadCoordinates));
        GL_CHECK(glEnableVertexAttribArray(texturedQuadVertexLocation));
        GL_CHECK(glVertexAttribPointer(texturedQuadLowResTexCoordLocation, 2, GL_FLOAT,
            GL_FALSE, 0, texturedQuadLowResTexCoordinates));
        GL_CHECK(glEnableVertexAttribArray(texturedQuadLowResTexCoordLocation));
        GL_CHECK(glVertexAttribPointer(texturedQuadHighResTexCoordLocation, 2, GL_FLOAT,
            GL_FALSE, 0, texturedQuadHighResTexCoordinates));
        GL_CHECK(glEnableVertexAttribArray(texturedQuadHighResTexCoordLocation));

        /*
         * Upload uniforms. The layerIndex is used to choose what layer of the array texture to sample from.
         * The shader will use the given layerIndex and layerIndex + 2, where layerIndex gives the layer with
         * the wide field of view, where the entire scene has been rendered, and layerIndex + 2 gives the layer
         * with the narrow field of view, where only the center of the scene has been rendered.
         */
        GL_CHECK(glUniform1i(texturedQuadSamplerLocation, 0));
        GL_CHECK(glUniform1i(texturedQuadLayerIndexLocation, i));

        /* Draw textured quad using the multiview texture. */
        GL_CHECK(glDrawArrays(GL_TRIANGLES, 0, 6));
    }
}

static void error_callback(int error, const char* description)
{
    LOGE("GLFW Error: %s\n", description);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}

int main(void)
{
    glfwSetErrorCallback(error_callback);

    if (!glfwInit())
        exit(EXIT_FAILURE);

    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_EGL_CONTEXT_API);

    GLFWwindow* window = glfwCreateWindow(fboWidth, fboHeight, "OpenGL ES 2.0 Triangle (EGL)", NULL, NULL);
    if (!window)
    {
        glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_NATIVE_CONTEXT_API);
        window = glfwCreateWindow(fboWidth, fboHeight, "OpenGL ES 2.0 Triangle", NULL, NULL);
        if (!window)
        {
            glfwTerminate();
            exit(EXIT_FAILURE);
        }
    }

    glfwSetKeyCallback(window, key_callback);

    glfwMakeContextCurrent(window);
    gladLoadGLES2(glfwGetProcAddress);
    glfwSwapInterval(1);

    // Set up all necessary resources for multiview rendering.
    setupGraphics(fboWidth, fboHeight);

    while (!glfwWindowShouldClose(window))
    {
        // Render multiview frames.
        renderFrame();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);

    glfwTerminate();
    exit(EXIT_SUCCESS);
}
