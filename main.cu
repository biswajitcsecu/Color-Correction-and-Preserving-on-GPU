
#include <iostream>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <GL/gl.h>
#include <GL/freeglut.h>
#include <cuda.h>
#include <cudaGL.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <cuda_surface_types.h>
#include <cuda_texture_types.h>
#include <device_launch_parameters.h>
#include <cuda_device_runtime_api.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h> 
#include <helper_functions.h>
#include <cutil.h>
#include <omp.h>
#include <helper_gl.h> 
#include <helper_math.h>

using namespace std;

#pragma pack(1)
#define Tx 16
#define Ty 16

#define MAX_EPSILON_ERROR   5.0f
#define REFRESH_DELAY       10 
#define MIN_EUCLIDEAN_D     0.01f
#define MAX_EUCLIDEAN_D     5.f
#define MAX_FILTER_RADIUS   25
#define GL_TEXTURE_TYPE GL_TEXTURE_2D


//cuda graphics resources
struct uchar4;
GLuint pbo;
GLuint texid;  
GLuint shader;
struct cudaGraphicsResource *cuda_pbo_resource;


//Timer counter
StopWatchInterface *timer = NULL;
StopWatchInterface *kernel_timer = NULL;

//Parameters
float gaussian_delta = 4;
float euclidean_delta = 0.1f;
const char **pArgv = NULL;
unsigned int *pArgc = NULL;
unsigned int iterations = 1;
unsigned int filter_radius = 5;
unsigned int width;
unsigned int height;
unsigned int  *hImage  = NULL;
unsigned int fpsCount = 0;
unsigned int fpsLimit = 1;
unsigned int g_TotalErrors = 0;
unsigned int devID=0;
unsigned int  dev;
bool g_bInteractive = false;


//BMP Data structure
typedef struct
{
    short type;
    int size;
    short reserved1;
    short reserved2;
    int offset;
} BMPHeader;

typedef struct
{
    int size;
    int width;
    int height;
    short planes;
    short bitsPerPixel;
    unsigned compression;
    unsigned imageSize;
    int xPelsPerMeter;
    int yPelsPerMeter;
    int clrUsed;
    int clrImportant;
} BMPInfoHeader;



//Function prototype;
static double imageFilterRGBA(unsigned int *d_dest, int width, int height,float e_d, 
        int radius, int iterations, StopWatchInterface *timer);
static void updateGaussian(float delta, int radius);
static void LoadBMPFile(uchar4 **dst, unsigned int *width, unsigned int *height, const char *name);



__constant__ float cGaussian[64]; 
cudaTextureObject_t rgbaTexdImage;
cudaTextureObject_t rgbaTexdTemp;

uint *dImage  = NULL;  
uint *dTemp   = NULL;   
size_t pitch;


//Euclidean Distance measure
__device__ float euclideanLen(float4 a, float4 b, float d)
{

    float mod = (b.x - a.x) * (b.x - a.x) +  (b.y - a.y) * (b.y - a.y) +  (b.z - a.z) * (b.z - a.z);

    return (__expf(-mod / (2.f * d * d)));
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(fabs(rgba.x)); 
    rgba.y = __saturatef(fabs(rgba.y));
    rgba.z = __saturatef(fabs(rgba.z));
    rgba.w = __saturatef(fabs(rgba.w));
    return (uint(rgba.w * 255.0f) << 24) | (uint(rgba.z * 255.0f) << 16) 
            | (uint(rgba.y * 255.0f) << 8) | uint(rgba.x * 255.0f);
}

__device__ float4 rgbaIntToFloat(uint c)
{
    float4 rgba;
    rgba.x = (c & 0xff) * 0.003921568627f; 
    rgba.y = ((c>>8) & 0xff) * 0.003921568627f;  
    rgba.z = ((c>>16) & 0xff) * 0.003921568627f; 
    rgba.w = ((c>>24) & 0xff) * 0.003921568627f; 
    return rgba;
}

//Kernel for filter 
__global__ void
rgbImagefilter(uint *od, int w, int h,  float e_d,  int r, cudaTextureObject_t rgbaTex)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    const int i= y * w + x;

    if (x >= w || y >= h)
    {
        return;
    }

    float sum = 0.0f;
    float factor;
    float4 t = {0.f, 0.f, 0.f, 0.f};
    float4 center = tex2D<float4>(rgbaTex, x, y);

    for (int i = -r; i <= r; i++){
        for (int j = -r; j <= r; j++){
            float4 curPix = tex2D<float4>(rgbaTex, x + j, y + i);
            factor = cGaussian[i + r] * cGaussian[j + r] * euclideanLen(curPix, center, e_d); 
            t += factor * curPix;
            sum += factor;
        }
    }
    od[i] = rgbaFloatToInt(t/sum);
}


static void initTexture(int width, int height, uint *hImage){
    // copy image data to cuda array
    cudaMallocPitch(&dImage, &pitch, sizeof(uint)*width, height);
    cudaMallocPitch(&dTemp,  &pitch, sizeof(uint)*width, height);
    cudaMemcpy2D(dImage, pitch, hImage, sizeof(uint)*width,sizeof(uint)*width, height,cudaMemcpyHostToDevice);
    cudaStreamSynchronize(0);
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    cudaResourceDesc  texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType  = cudaResourceTypePitch2D;
    texRes.res.pitch2D.devPtr   = dImage;
    texRes.res.pitch2D.desc     = desc;
    texRes.res.pitch2D.width    = width;
    texRes.res.pitch2D.height   = height;
    texRes.res.pitch2D.pitchInBytes = pitch;
    cudaTextureDesc   texDescr;
    memset(&texDescr,0,sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode   = cudaFilterModePoint;
    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.addressMode[1] = cudaAddressModeWrap;
    texDescr.readMode = cudaReadModeNormalizedFloat;

    cudaCreateTextureObject(&rgbaTexdImage, &texRes, &texDescr, NULL);
    memset(&texRes,0,sizeof(cudaResourceDesc));

    texRes.resType   = cudaResourceTypePitch2D;
    texRes.res.pitch2D.devPtr   = dTemp;
    texRes.res.pitch2D.desc     = desc;
    texRes.res.pitch2D.width    = width;
    texRes.res.pitch2D.height   = height;
    texRes.res.pitch2D.pitchInBytes = pitch;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode   = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.addressMode[1] = cudaAddressModeWrap;
    texDescr.readMode = cudaReadModeNormalizedFloat;
    cudaCreateTextureObject(&rgbaTexdTemp, &texRes, &texDescr, NULL);
    
}


static void updateGaussian(float delta, int radius){
    float  fGaussian[64];
    #pragma omp parallel  for
    for (int i = 0; i < 2*radius + 1; ++i)
    {
        float x = i-radius;
        fGaussian[i] = expf(-(x*x) / (2*delta*delta));
    }
    cudaMemcpyToSymbol(cGaussian, fGaussian, sizeof(float)*(2*radius+1));
    cudaStreamSynchronize(0);
}

static double imageFilterRGBA(uint *dDest, int width, int height, float e_d, int radius, 
        int iterations, StopWatchInterface *timer)
{
    double dKernelTime;
    for (int i=0; i<iterations; i++)
    {
        dKernelTime = 0.0;
        cudaDeviceSynchronize();
        sdkResetTimer(&timer);
        dim3 gridSize((width + Tx - 1) / Ty, (height + Ty - 1) / Ty);
        dim3 blockSize(Tx, Ty);
        rgbImagefilter<<< gridSize, blockSize>>>(dDest, width, height, e_d, radius, rgbaTexdImage);       
        cudaDeviceSynchronize();
        dKernelTime += sdkGetTimerValue(&timer);      
    }
    return ((dKernelTime/1000.)/(double)iterations);
}

static void varyEuclidean(){
    static float factor = 1.01f;
    if (euclidean_delta > MAX_EUCLIDEAN_D)
    {
        factor = 1/1.01f;
    }

    if (euclidean_delta < MIN_EUCLIDEAN_D)
    {
        factor = 1.01f;
    }
    euclidean_delta *= factor;
}

static void computeFPS(){
    fpsCount++;
    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.0f / (sdkGetAverageTimerValue(&timer) / 1000.0f);
        sprintf(fps, "CUDA Color Image Filter:fps(%2.f)", ifps);
        glutSetWindowTitle(fps);
        fpsCount = 0;
        fpsLimit = (int)MAX(ifps, 1.0f);
        sdkResetTimer(&timer);
    }

    if (!g_bInteractive)
    {
        varyEuclidean();
    }
}

// Display
static void display(){
    sdkStartTimer(&timer);
    unsigned int *dResult;
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    cudaStreamSynchronize(0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void **)&dResult, &num_bytes, cuda_pbo_resource);
    imageFilterRGBA(dResult, width, height, euclidean_delta, filter_radius, iterations, kernel_timer);
    cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
    cudaStreamSynchronize(0);

    {
        //Clear window
        glClearColor (0.0, 0.0, 0.0, 0.0);
        glClearDepth(1.0f);
        glEnable(GL_DEPTH_TEST);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
        glEnable(GL_TEXTURE_2D);
        glEnable(GL_FRAGMENT_PROGRAM_ARB);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        
        //  texture from pbo
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); 
        glBindTexture(GL_TEXTURE_2D, texid);
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE); 
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
        
        // fragment program 
        glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, shader);
        glEnable(GL_FRAGMENT_PROGRAM_ARB);
        glDisable(GL_DEPTH_TEST);
        
        //Draw textures
        glPushMatrix();
        glBegin(GL_QUADS);
        {
            glTexCoord2f(0, 0); glVertex2f(0, 0);
            glTexCoord2f(1, 0); glVertex2f(1, 0);
            glTexCoord2f(1, 1); glVertex2f(1, 1);
            glTexCoord2f(0, 1); glVertex2f(0, 1);
        }
        glEnd();
        glPopMatrix();
        glBindTexture(GL_TEXTURE_TYPE, 0);
        glDisable(GL_FRAGMENT_PROGRAM_ARB);
        glDisable(GL_TEXTURE_2D);
        glDisable(GL_DEPTH_TEST); 
        
    }
    glutSwapBuffers();
    glutReportErrors();
    sdkStopTimer(&timer);
    computeFPS();
    glutPostRedisplay();
}

//Key board interactive
static void keyboard(unsigned char key, int x, int y){
    if(x!=y) {x=y=0;}    
    switch (key)
    {
        case 27:
                glutDestroyWindow(glutGetWindow());
                return;
            break;

        case 'a':
        case 'A':
            g_bInteractive = !g_bInteractive;
            printf("> Animation is %s\n", !g_bInteractive ? "ON" : "OFF");
            break;

        case ']':
            iterations++;
            break;

        case '[':
            iterations--;
            if (iterations < 1)
            {
                iterations = 1;
            }
            break;
        case '=':
        case '+':
            filter_radius++;
            if (filter_radius > MAX_FILTER_RADIUS)
            {
                filter_radius = MAX_FILTER_RADIUS;
            }
            updateGaussian(gaussian_delta, filter_radius);
            break;

        case '-':
            filter_radius--;
            if (filter_radius < 1)
            {
                filter_radius = 1;
            }
            updateGaussian(gaussian_delta, filter_radius);
            break;

        case 'E':
            euclidean_delta *= 1.5;
            break;

        case 'e':
            euclidean_delta /= 1.5;
            break;

        case 'g':
            if (gaussian_delta > 0.1)
            {
                gaussian_delta /= 2;
            }
            updateGaussian(gaussian_delta, filter_radius);
            break;

        case 'G':
            gaussian_delta *= 2;
            updateGaussian(gaussian_delta, filter_radius);
            break;

        default:
            break;
    }   
    glutPostRedisplay();
}

//Timer 
static void timerEvent(int value)
{
    if(glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    }
}

//Reshape window
static void reshape(int x, int y)
{
    glViewport(0, 0,(GLint)x, (GLint) y);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f);
    
}
//Clean Device memory
static void cleanup(){
    sdkDeleteTimer(&timer);
    sdkDeleteTimer(&kernel_timer);    
    cudaDestroyTextureObject(rgbaTexdImage);
    cudaDestroyTextureObject(rgbaTexdTemp);
    cudaFree(dImage);
    cudaFree(dTemp);
    free(hImage); 
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &texid);
    glDeleteProgramsARB(1, &shader);
}

static const char *shader_code =
    "!!ARBfp1.0\n"
    "TEX result.color, fragment.texcoord, texture[0], 2D; \n"
    "END";

GLuint compileASMShader(GLenum program_type, const char *code)
{
    GLuint program_id;
    glGenProgramsARB(1, &program_id);
    glBindProgramARB(program_type, program_id);
    glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei) strlen(code), (GLubyte *) code);    
    
    return program_id;
}

//Initiate GL resource
static void initGLResources()
{
    //PBO
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, hImage, GL_STREAM_DRAW_ARB);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    //TID
    glGenTextures(1, &texid);
    glBindTexture(GL_TEXTURE_2D, texid);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);   
    
    cudaGraphicsResourceSetMapFlags(cuda_pbo_resource, cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo,  cudaGraphicsMapFlagsWriteDiscard);
    shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);
}

static void LoadBMPFile(uchar4 **dst, unsigned int *width,  unsigned int *height, const char *name)
{
    BMPHeader hdr;
    BMPInfoHeader infoHdr;
    int x, y;
    FILE *fd;
    printf("Loading %s...\n", name);

    fd = fopen(name,"rb");
    fread(&hdr, sizeof(hdr), 1, fd);

    if (hdr.type != 0x4D42)
    {
        printf("***BMP load error: bad file format***\n");
        exit(EXIT_SUCCESS);
    }
    fread(&infoHdr, sizeof(infoHdr), 1, fd);

    if (infoHdr.bitsPerPixel != 24)
    {
        printf("***BMP load error: invalid color depth***\n");
        exit(EXIT_SUCCESS);
    }
    if (infoHdr.compression)
    {
        printf("***BMP load error: compressed image***\n");
        exit(EXIT_SUCCESS);
    }
    //Dim of image
    *width  = infoHdr.width;
    *height = infoHdr.height;
    *dst    = (uchar4 *)malloc(*width **height * 4);
    printf("BMP width: %u\n", infoHdr.width);
    printf("BMP height: %u\n", infoHdr.height);
    fseek(fd, hdr.offset - sizeof(hdr) - sizeof(infoHdr), SEEK_CUR);
    
#pragma omp parallel 
    for (y = 0; y < infoHdr.height; y++){
#pragma omp parallel  for
        for (x = 0; x < infoHdr.width; x++){
            (*dst)[(y * infoHdr.width + x)].w = 0;
            (*dst)[(y * infoHdr.width + x)].z = fgetc(fd);
            (*dst)[(y * infoHdr.width + x)].y = fgetc(fd);
            (*dst)[(y * infoHdr.width + x)].x = fgetc(fd);
        }
#pragma omp parallel  for
        for (x = 0; x < (4 - (3 * infoHdr.width) % 4) % 4; x++)
        {
            fgetc(fd);
        }
    }
    if (ferror(fd))
    {
        printf("***Unknown BMP load error.***\n");
        free(*dst);
        exit(EXIT_SUCCESS);
    }
    else
    {
        printf("BMP file loaded successfully!\n");
    }
    fclose(fd);
}

int main(int argc, char **argv)
{
    printf("Starting Application..\n");
    setenv ("DISPLAY", ":0", 0);    
    try{ 
        //Device info.
        devID = findCudaDevice(argc, (const char **)argv);
        int runtimeVersion = 0;   
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);   
        cudaRuntimeGetVersion(&runtimeVersion);
        fprintf(stderr,"\nDevice %d: \"%s\"\n", dev, deviceProp.name);
        fprintf(stderr,"  CUDA Runtime Version     :\t%d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10);
        fprintf(stderr,"  CUDA Compute Capability  :\t%d.%d\n", deviceProp.major, deviceProp.minor);

        //Load Image
        char *image_path = NULL;
        const char *image_filename = "src.bmp";
        image_path = sdkFindFilePath(image_filename, 0);
        LoadBMPFile((uchar4 **)&hImage, &width, &height, image_path);
        printf("Loaded '%s', %d x %d pixels\n", image_path, width, height);

        //Initialization of Glut model
        glutInit(&argc, argv);      
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE|GLUT_DEPTH);
        glutInitWindowSize(width, height);
        glutInitWindowPosition(100,100);
        glutCreateWindow("CUDA Bilateral Filter");
        glutDisplayFunc(display);
        glutKeyboardFunc(keyboard);
        glutReshapeFunc(reshape);
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0); 
        //Call methods
        updateGaussian(gaussian_delta, filter_radius);
        initTexture(width, height, hImage);
        sdkCreateTimer(&timer);
        sdkCreateTimer(&kernel_timer);
        initGLResources();
        //Instruction
        printf("Running Standard Demonstration with GLUT loop...\n");
        printf("Press '+' and '-' to change filter width\n"
            "Press ']' and '[' to change number of iterations\n"
            "Press 'e' and 'E' to change Euclidean delta\n"
            "Press 'g' and 'G' to change Gaussian delta\n"
            "Press 'a' or  'A' to change Animation mode ON/OFF\n");
        
        glutCloseFunc(cleanup);
        glutMainLoop();        
    }catch(exception &erb){
        std::cerr<<"Error found"<<erb.what()<<"\n";
    }
    return EXIT_SUCCESS;   
}
