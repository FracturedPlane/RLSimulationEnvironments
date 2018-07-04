/**************************************************************************
 *
 * Copyright 2008 VMware, Inc.
 * All Rights Reserved.
 *
 **************************************************************************/

/*
 * Draw a triangle with X/EGL and OpenGL ES 2.x
 */

#define USE_FULL_GL 0



#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#if USE_FULL_GL
#include "gl_wrap.h"  /* use full OpenGL */
#else
#include <GLES2/gl2.h>  /* use OpenGL ES 3.x */
#endif
#define EGL_EGLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <iostream>


#include <exception>

class EGLException:public std::exception
{
  public:
    const char* message;
    EGLException(const char* mmessage): message(mmessage) {}

    virtual const char* what() const throw()
    {
      return this->message;
    }
};

class EGLReturnException: private EGLException
{
  using EGLException::EGLException;
};

class EGLErrorException: private EGLException
{
  using EGLException::EGLException;
};

#define checkEglError(message){ \
    EGLint err = eglGetError(); \
    if (err != EGL_SUCCESS) \
    { \
        std::cerr << "EGL Error " << std::hex << err << std::dec << " on line " <<  __LINE__ << std::endl; \
        throw EGLErrorException(message); \
    } \
}

#define checkEglReturn(x, message){ \
    if (x != EGL_TRUE) \
    { \
        std::cerr << "EGL returned not true on line " << __LINE__ << std::endl; \
        throw EGLReturnException(message); \
    } \
}

#define FLOAT_TO_FIXED(X)   ((X) * 65535.0)

const int winWidth = 300, winHeight = 300;
EGLSurface egl_surf;
EGLContext egl_ctx;
EGLDisplay egl_dpy;
char *dpyName = NULL;
GLboolean printInfo = GL_FALSE;
EGLint egl_major, egl_minor;
int i;
const char *s;
int cudaIndexDesired = 0;

static GLfloat view_rotx = 0.0, view_roty = 0.0;
static GLfloat view_transx = 0.0, view_transy = 0.0, view_transz = 0.0;
static GLfloat view_transx2 = 0.0, view_transy2 = 0.0, view_transz2 = 0.0;

static GLint u_matrix = -1;
static GLint attr_pos = 0, attr_color = 1;

static void setPosition(GLfloat xs, GLfloat ys, GLfloat zs)
{
	view_transx = xs;
	view_transy = ys;
	view_transz = zs;
}

static void setPosition2(GLfloat xs, GLfloat ys, GLfloat zs)
{
	view_transx2 = xs;
	view_transy2 = ys;
	view_transz2 = zs;
}


static void
make_z_rot_matrix(GLfloat angle, GLfloat *m)
{
   float c = cos(angle * M_PI / 180.0);
   float s = sin(angle * M_PI / 180.0);
   int i;
   for (i = 0; i < 16; i++)
      m[i] = 0.0;
   m[0] = m[5] = m[10] = m[15] = 1.0;

   m[0] = c;
   m[1] = s;
   m[4] = -s;
   m[5] = c;
}

static void
make_scale_matrix(GLfloat xs, GLfloat ys, GLfloat zs, GLfloat *m)
{
   int i;
   for (i = 0; i < 16; i++)
      m[i] = 0.0;
   m[0] = xs;
   m[5] = ys;
   m[10] = zs;
   m[15] = 1.0;
}

/*
 * row major matrix?
 */
static void
make_translation_matrix(GLfloat x, GLfloat y, GLfloat z, GLfloat *m)
{
   int i;
   for (i = 0; i < 16; i++)
      m[i] = 0.0;

   m[0] = m[5] = m[10] = m[15] = 1.0;
	// m[3] = x;
	// m[7] = y;
	// m[11] = z;
	m[12] = x;
	m[13] = y;
	m[14] = z;
}


static void
mul_matrix(GLfloat *prod, const GLfloat *a, const GLfloat *b)
{
#define A(row,col)  a[(col<<2)+row]
#define B(row,col)  b[(col<<2)+row]
#define P(row,col)  p[(col<<2)+row]
   GLfloat p[16];
   GLint i;
   for (i = 0; i < 4; i++) {
      const GLfloat ai0=A(i,0),  ai1=A(i,1),  ai2=A(i,2),  ai3=A(i,3);
      P(i,0) = ai0 * B(0,0) + ai1 * B(1,0) + ai2 * B(2,0) + ai3 * B(3,0);
      P(i,1) = ai0 * B(0,1) + ai1 * B(1,1) + ai2 * B(2,1) + ai3 * B(3,1);
      P(i,2) = ai0 * B(0,2) + ai1 * B(1,2) + ai2 * B(2,2) + ai3 * B(3,2);
      P(i,3) = ai0 * B(0,3) + ai1 * B(1,3) + ai2 * B(2,3) + ai3 * B(3,3);
   }
   memcpy(prod, p, sizeof(p));
#undef A
#undef B
#undef PROD
}

// dumps a PPM raw (P6) file on an already allocated memory array
void DumpPPM(FILE *fp, int width, int height)
{
    const int maxVal=255;
    register int y;
    int r = 0;
    int sum = 0;
    int b_width = 3*width;
    //printf("width = %d height = %d\n",width, height) ;
    fprintf(fp,	"P6 ");
    fprintf(fp,	"%d %d ", width, height);
    fprintf(fp,	"%d\n",	maxVal);
    unsigned char m_pixels[3*width];

	// glReadBuffer(GL_FRONT) ;

	for	( y = height-1;	y>=0; y-- )
	{
		// bzero(m_pixels, 3*width);
		glReadPixels(0,y,width,1,GL_RGB,GL_UNSIGNED_BYTE,
			(GLvoid *) m_pixels);
		fwrite(m_pixels, 3, width, fp);
	}
}
int m_frameCount = 0;
int save_PPM()
{
	int width = 300;
	int height = 300;

	char fname[128] ;
	sprintf(fname,"frame%d.ppm", m_frameCount) ;
	m_frameCount++ ;
	FILE *fp = fopen(fname,"wb") ;
	if( fp == NULL )
	{
		fprintf(stderr, "Cannot open file %s\n", fname) ;
		return -1 ;
	}
	DumpPPM(fp,width,height) ;
	fclose(fp) ;
	return 1 ;
}

static void
draw(void)
{
   static const GLfloat verts[3][2] = {
      { -1, -1 },
      {  1, -1 },
      {  0,  1 }
   };
   static const GLfloat colors[3][3] = {
      { 1, 0, 0 },
      { 0, 1, 0 },
      { 0, 0, 1 }
   };
   GLfloat mat[16], rot[16], scale[16], trans[16];
   GLfloat mat2[16], rot2[16], scale2[16], trans2[16];

   /* Set modelview/projection matrix */
   make_z_rot_matrix(view_rotx, rot);
   make_scale_matrix(0.5, 0.5, 0.5, scale);
   make_translation_matrix(view_transx, view_transy, view_transz, trans);
   mul_matrix(mat, trans, rot);
   mul_matrix(mat, mat, scale);

    make_z_rot_matrix(view_rotx, rot2);
	make_scale_matrix(0.5, 0.5, 0.5, scale2);
	make_translation_matrix(view_transx2, view_transy2, view_transz2, trans2);
	mul_matrix(mat2, trans2, rot2);
	mul_matrix(mat2, mat2, scale2);

   glUniformMatrix4fv(u_matrix, 1, GL_FALSE, mat);

   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

   {
      glVertexAttribPointer(attr_pos, 2, GL_FLOAT, GL_FALSE, 0, verts);
      glVertexAttribPointer(attr_color, 3, GL_FLOAT, GL_FALSE, 0, colors);
      glEnableVertexAttribArray(attr_pos);
      glEnableVertexAttribArray(attr_color);

      glDrawArrays(GL_TRIANGLES, 0, 3);

      glDisableVertexAttribArray(attr_pos);
      glDisableVertexAttribArray(attr_color);
   }

   glUniformMatrix4fv(u_matrix, 1, GL_FALSE, mat2);

   {
      glVertexAttribPointer(attr_pos, 2, GL_FLOAT, GL_FALSE, 0, verts);
      glVertexAttribPointer(attr_color, 3, GL_FLOAT, GL_FALSE, 0, colors);
      glEnableVertexAttribArray(attr_pos);
      glEnableVertexAttribArray(attr_color);

      glDrawArrays(GL_TRIANGLES, 0, 3);

      glDisableVertexAttribArray(attr_pos);
      glDisableVertexAttribArray(attr_color);
   }
	eglSwapBuffers(egl_dpy, egl_surf);

}


/* new window size or exposure */
static void
reshape(int width, int height)
{
   glViewport(0, 0, (GLint) width, (GLint) height);
}


static void
create_shaders(void)
{
   static const char *fragShaderText =
      "precision mediump float;\n"
      "varying vec4 v_color;\n"
      "void main() {\n"
      "   gl_FragColor = v_color;\n"
      "}\n";
   static const char *vertShaderText =
      "uniform mat4 modelviewProjection;\n"
      "attribute vec4 pos;\n"
      "attribute vec4 color;\n"
      "varying vec4 v_color;\n"
      "void main() {\n"
      "   gl_Position = modelviewProjection * pos;\n"
      "   v_color = color;\n"
      "}\n";

   GLuint fragShader, vertShader, program;
   GLint stat;

   fragShader = glCreateShader(GL_FRAGMENT_SHADER);
   glShaderSource(fragShader, 1, (const char **) &fragShaderText, NULL);
   glCompileShader(fragShader);
   glGetShaderiv(fragShader, GL_COMPILE_STATUS, &stat);
   if (!stat) {
      printf("Error: fragment shader did not compile!\n");
      exit(1);
   }

   vertShader = glCreateShader(GL_VERTEX_SHADER);
   glShaderSource(vertShader, 1, (const char **) &vertShaderText, NULL);
   glCompileShader(vertShader);
   glGetShaderiv(vertShader, GL_COMPILE_STATUS, &stat);
   if (!stat) {
      printf("Error: vertex shader did not compile!\n");
      exit(1);
   }

   program = glCreateProgram();
   glAttachShader(program, fragShader);
   glAttachShader(program, vertShader);
   glLinkProgram(program);

   glGetProgramiv(program, GL_LINK_STATUS, &stat);
   if (!stat) {
      char log[1000];
      GLsizei len;
      glGetProgramInfoLog(program, 1000, &len, log);
      printf("Error: linking:\n%s\n", log);
      exit(1);
   }

   glUseProgram(program);

   if (1) {
      /* test setting attrib locations */
      glBindAttribLocation(program, attr_pos, "pos");
      glBindAttribLocation(program, attr_color, "color");
      glLinkProgram(program);  /* needed to put attribs into effect */
   }
   else {
      /* test automatic attrib locations */
      attr_pos = glGetAttribLocation(program, "pos");
      attr_color = glGetAttribLocation(program, "color");
   }

   u_matrix = glGetUniformLocation(program, "modelviewProjection");
   printf("Uniform modelviewProjection at %d\n", u_matrix);
   printf("Attrib pos at %d\n", attr_pos);
   printf("Attrib color at %d\n", attr_color);
}


static void
init(void)
{
   typedef void (*proc)();

#if 1 /* test code */
   proc p = eglGetProcAddress("glMapBufferOES");
   assert(p);
#endif

   glClearColor(0.9, 0.9, 0.9, 0.0);

   create_shaders();
}


/*
 * Create an RGB, double-buffered Headless window.
 * context handle.
 */
static void
make_headless_window(EGLDisplay egl_dpy,
              const char *name,
              int x, int y, int width, int height,
              EGLContext *ctxRet,
              EGLSurface *surfRet)
{
	/*
    const EGLint attribs[] ={
        EGL_RED_SIZE,           8,
        EGL_GREEN_SIZE,         8,
        EGL_BLUE_SIZE,          8,
        EGL_ALPHA_SIZE,         8,
        EGL_DEPTH_SIZE,         24,
        EGL_STENCIL_SIZE,       8,
        EGL_COLOR_BUFFER_TYPE,  EGL_RGB_BUFFER,
        EGL_SURFACE_TYPE,       EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE,    EGL_OPENGL_BIT,
        EGL_NONE
    };
    */

   static const EGLint attribs[] = {
      EGL_RED_SIZE, 1,
      EGL_GREEN_SIZE, 1,
      EGL_BLUE_SIZE, 1,
      EGL_DEPTH_SIZE, 1,
      EGL_SURFACE_TYPE,       EGL_PBUFFER_BIT,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
      EGL_NONE
   };

#if USE_FULL_GL
   static const EGLint ctx_attribs[] = {
       EGL_NONE
   };
#else
   static const EGLint ctx_attribs[] = {
      EGL_CONTEXT_CLIENT_VERSION, 2,
      EGL_NONE
   };
#endif

   unsigned long mask;
   int num_visuals;
   EGLContext ctx;
   EGLConfig config;
   EGLint num_configs;
   EGLint vid;


   if (!eglChooseConfig( egl_dpy, attribs, &config, 1, &num_configs)) {
      printf("Error: couldn't get an EGL visual config\n");
      exit(1);
   }

   assert(num_configs > 0);
   assert(config);

   if (!eglGetConfigAttrib(egl_dpy, config, EGL_NATIVE_VISUAL_ID, &vid)) {
      printf("Error: eglGetConfigAttrib() failed\n");
      exit(1);
   }

#if USE_FULL_GL /* XXX fix this when eglBindAPI() works */
   eglBindAPI(EGL_OPENGL_API);
#else
   eglBindAPI(EGL_OPENGL_ES_API);
#endif

   ctx = eglCreateContext(egl_dpy, config, EGL_NO_CONTEXT, ctx_attribs );
   if (!ctx) {
      printf("Error: eglCreateContext failed\n");
      exit(1);
   }

#if !USE_FULL_GL
   /* test eglQueryContext() */
   {
      EGLint val;
      eglQueryContext(egl_dpy, ctx, EGL_CONTEXT_CLIENT_VERSION, &val);
      printf("qeurry EGL_CONTEXT_CLIENT_VERSION %d\n", val);
      assert(val != 0);
   }
#endif
   /// Could not get A WindowSurface to work
   // *surfRet = eglCreateWindowSurface(egl_dpy, config, (EGLNativeWindowType)NULL, NULL);
   static const EGLint pbufferAttribs[] = {
         EGL_WIDTH, width,
         EGL_HEIGHT, height,
         EGL_NONE,
   };
   *surfRet = eglCreatePbufferSurface(egl_dpy, config,
                                                      pbufferAttribs);
   if (!*surfRet) {
	  std::cout << glGetError() << std::endl;
      printf("Error: eglCreateWindowSurface failed\n");
      exit(1);
   }

   /* sanity checks */
   {
      EGLint val;
      eglQuerySurface(egl_dpy, *surfRet, EGL_WIDTH, &val);
      assert(val == width);
      eglQuerySurface(egl_dpy, *surfRet, EGL_HEIGHT, &val);
      assert(val == height);
      assert(eglGetConfigAttrib(egl_dpy, config, EGL_SURFACE_TYPE, &val));
      // assert(val & EGL_WINDOW_BIT);
      assert(val);
   }

   *ctxRet = ctx;
}

EGLDisplay eglGetDisplay_(NativeDisplayType nativeDisplay=EGL_DEFAULT_DISPLAY)
{
  EGLDisplay eglDisplay = eglGetDisplay(nativeDisplay);
  checkEglError("Failed to Get Display: eglGetDisplay");
  std::cerr << "Failback to eglGetDisplay" << std::endl;
  return eglDisplay;
}

int _init()
{
   int desiredGPUDeviceIndex = 0;

   std::cout << "Desired device: " << desiredGPUDeviceIndex << std::endl;

   PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT = (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
   checkEglError("Failed to get EGLEXT: eglQueryDevicesEXT");
   PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT = (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
   checkEglError("Failed to get EGLEXT: eglGetPlatformDisplayEXT");
   PFNEGLQUERYDEVICEATTRIBEXTPROC eglQueryDeviceAttribEXT = (PFNEGLQUERYDEVICEATTRIBEXTPROC)eglGetProcAddress("eglQueryDeviceAttribEXT");
   checkEglError("Failed to get EGLEXT: eglQueryDeviceAttribEXT");

   if (desiredGPUDeviceIndex >= 0)
	 {
	   EGLDeviceEXT *eglDevs;
	   EGLint numberDevices;

	   //Get number of devices
	   checkEglReturn(
		 eglQueryDevicesEXT(0, NULL, &numberDevices),
		 "Failed to get number of devices. Bad parameter suspected"
	   );
	   checkEglError("Error getting number of devices: eglQueryDevicesEXT");

	   std::cerr << numberDevices << " devices found" << std::endl;

	   if (numberDevices)
	   {
		 EGLAttrib cudaIndex;

		 //Get devices
		 eglDevs = new EGLDeviceEXT[numberDevices];
		 checkEglReturn(
		   eglQueryDevicesEXT(numberDevices, eglDevs, &numberDevices),
		   "Failed to get devices. Bad parameter suspected"
		 );
		 checkEglError("Error getting number of devices: eglQueryDevicesEXT");

		 /*
		 for(i=0; i<numberDevices; i++)
		 {
		   checkEglReturn(
			 eglQueryDeviceAttribEXT(eglDevs[i], EGL_CUDA_DEVICE_NV, &cudaIndex),
			 "Failed to get EGL_CUDA_DEVICE_NV attribute for device"
		   );
		   checkEglError("Error retreiving EGL_CUDA_DEVICE_NV attribute for device");
		   std::cout << "Device index: " << cudaIndex << std::endl;
		   if (cudaIndex == desiredGPUDeviceIndex)
			 break;
		 }*/
		 if (desiredGPUDeviceIndex < numberDevices)
		 {
		   egl_dpy = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, eglDevs[desiredGPUDeviceIndex], 0);
		   checkEglError("Error getting Platform Display: eglGetPlatformDisplayEXT");
		   std::cerr << "Got Cuda device " << desiredGPUDeviceIndex << std::endl;
		 }
		 else
		 {
		   egl_dpy = eglGetDisplay_();
		 }
	   }
	   else
	   {//If no devices were found, or a matching cuda not found, get a Display the normal way
		 egl_dpy = eglGetDisplay_();
	   }
	 }
	 else
	 {
	   egl_dpy = eglGetDisplay_();
	 }

	 if (egl_dpy == EGL_NO_DISPLAY)
	   throw EGLException("No Disply Found");
   // unsetenv("DISPLAY"); //Force Headless
	// egl_dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   if (!egl_dpy) {
	  printf("Error: eglGetDisplay() failed\n");
	  return -1;
   }

   if (!eglInitialize(egl_dpy, &egl_major, &egl_minor)) {
	  printf("Error: eglInitialize() failed\n");
	  return -1;
   }

   s = eglQueryString(egl_dpy, EGL_VERSION);
   printf("EGL_VERSION = %s\n", s);

   s = eglQueryString(egl_dpy, EGL_VENDOR);
   printf("EGL_VENDOR = %s\n", s);

   s = eglQueryString(egl_dpy, EGL_EXTENSIONS);
   printf("EGL_EXTENSIONS = %s\n", s);

   s = eglQueryString(egl_dpy, EGL_CLIENT_APIS);
   printf("EGL_CLIENT_APIS = %s\n", s);

   make_headless_window(egl_dpy,
				 "OpenGL ES 2.x tri", 0, 0, winWidth, winHeight,
				&egl_ctx, &egl_surf);

   if (!eglMakeCurrent(egl_dpy, egl_surf, egl_surf, egl_ctx)) {
	  printf("Error: eglMakeCurrent() failed\n");
	  return -1;
   }

   if (printInfo) {
	  printf("GL_RENDERER   = %s\n", (char *) glGetString(GL_RENDERER));
	  printf("GL_VERSION    = %s\n", (char *) glGetString(GL_VERSION));
	  printf("GL_VENDOR     = %s\n", (char *) glGetString(GL_VENDOR));
	  printf("GL_EXTENSIONS = %s\n", (char *) glGetString(GL_EXTENSIONS));
   }

   init();

   /* Set initial projection/viewing transformation.
	* We can't be sure we'll get a ConfigureNotify event when the window
	* first appears.
	*/
   reshape(winWidth, winHeight);

}

void finish()
{
	eglDestroyContext(egl_dpy, egl_ctx);
	eglDestroySurface(egl_dpy, egl_surf);
	eglTerminate(egl_dpy);
}

int main(int argc, char *argv[])
{
	_init();

   draw();
   for (size_t i = 0; i < 5; i++)
   {
		draw();
		save_PPM();
		view_rotx += 5.0;
		setPosition(i * 0.2 + 0.2, i * 0.2 + 0.2, 0);
		setPosition2(i * -0.2 + -0.2, i * -0.2 + -0.2, 0);
   }

   finish();
   return 0;
}
