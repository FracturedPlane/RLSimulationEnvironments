/**************************************************************************
 *
 * Copyright 2018 Glen Berth.
 * All Rights Reserved.
 *
 **************************************************************************/

#include "EGLRender.h"


#define FLOAT_TO_FIXED(X)   ((X) * 65535.0)

EGLRender::EGLRender()
{
	int m_frameCount = 0;
	winWidth = 1000, winHeight = 1000;
	char *dpyName = NULL;
	GLboolean printInfo = GL_FALSE;
	int i;
	const char *s;
	cudaIndexDesired = 0;

	drawAgent = true, drawObject = true;

	view_rotx = 0.0, view_roty = 0.0;
	view_transx = 0.0, view_transy = 0.0, view_transz = 0.0;
	view_transx2 = 0.0, view_transy2 = 0.0, view_transz2 = 0.0;
	setCameraPosition(0.0, 0.0, 0.0);

	u_matrix = -1;
	attr_pos = 0, attr_color = 1;
}

EGLRender::~EGLRender()
{
	this->finish();
}

void EGLRender::setPosition(float xs, float ys, float zs)
{
	view_transx = xs;
	view_transy = ys;
	view_transz = zs;
}

void EGLRender::setPosition2(float xs, float ys, float zs)
{
	view_transx2 = xs;
	view_transy2 = ys;
	view_transz2 = zs;
}

void EGLRender::setCameraPosition(float xs, float ys, float zs)
{
	camPos[0] = xs;
	camPos[1] = ys;
	camPos[2] = zs;
}

void EGLRender::setDrawAgent(bool draw_)
{
	drawAgent = draw_;
}
void EGLRender::setDrawObject(bool draw_)
{
	drawObject = draw_;
}

// dumps a PPM raw (P6) file on an already allocated memory array
void EGLRender::DumpPPM(FILE *fp, int width, int height)
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

int EGLRender::save_PPM()
{

	char fname[128] ;
	sprintf(fname,"frame%d.ppm", m_frameCount) ;
	m_frameCount++ ;
	FILE *fp = fopen(fname,"wb") ;
	if( fp == NULL )
	{
		fprintf(stderr, "Cannot open file %s\n", fname) ;
		return -1 ;
	}
	DumpPPM(fp,winWidth,winHeight) ;
	fclose(fp) ;
	return 1 ;
}

void EGLRender::draw(void)
{
	float tri_scale = 0.05;
   const GLfloat verts[3][2] = {
      { -1, -1 },
      {  1, -1 },
      {  0,  1 }
   };
   GLfloat colors[3][3] = {
      { 0, 0, 1 },
      { 0, 0, 1 },
      { 0, 0, 1 }
   };
   GLfloat mat[16], rot[16], scale[16], trans[16], camMat[16];
   GLfloat mat2[16], rot2[16], scale2[16], trans2[16];

   /* Set modelview/projection matrix */
   make_z_rot_matrix(view_rotx, rot);
   make_scale_matrix(tri_scale, tri_scale, tri_scale, scale);
   make_translation_matrix(view_transx, view_transy, view_transz, trans);
   make_translation_matrix(-camPos[0],-camPos[1], -camPos[2], camMat);
   mul_matrix(mat, trans, rot);
   mul_matrix(mat, mat, camMat);
   mul_matrix(mat, mat, scale);


    make_z_rot_matrix(view_rotx, rot2);
	make_scale_matrix(tri_scale, tri_scale, tri_scale, scale2);
	make_translation_matrix(view_transx2, view_transy2, view_transz2, trans2);
	mul_matrix(mat2, trans2, rot2);
	mul_matrix(mat2, mat2, camMat);
	mul_matrix(mat2, mat2, scale2);

   glUniformMatrix4fv(u_matrix, 1, GL_FALSE, mat);

   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

   if (drawAgent)
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
   GLfloat colors2[3][3] = {
         { 1, 0, 0 },
         { 1, 0, 0 },
         { 1, 0, 0 }
      };
   if (drawObject)
   {
      glVertexAttribPointer(attr_pos, 2, GL_FLOAT, GL_FALSE, 0, verts);
      glVertexAttribPointer(attr_color, 3, GL_FLOAT, GL_FALSE, 0, colors2);
      glEnableVertexAttribArray(attr_pos);
      glEnableVertexAttribArray(attr_color);

      glDrawArrays(GL_TRIANGLES, 0, 3);

      glDisableVertexAttribArray(attr_pos);
      glDisableVertexAttribArray(attr_color);
   }
	eglSwapBuffers(egl_dpy, egl_surf);

}


/* new window size or exposure */
void EGLRender::reshape(int width, int height)
{
   glViewport(0, 0, (GLint) width, (GLint) height);
}

std::vector<unsigned char> EGLRender::getPixels(size_t x_start, size_t y_start, size_t width, size_t height)
{
	// drawAgent = true;
	// drawObject = false;
	// draw();
	std::vector<unsigned char > out;
	size_t num_pixels = 3*width*height;
	unsigned char m_pixels[num_pixels];
	glReadPixels(x_start, y_start, width, height,
			GL_RGB,GL_UNSIGNED_BYTE, (GLvoid *) m_pixels);
	for (size_t i = 0; i < num_pixels; i ++)
	{
		out.push_back(m_pixels[i]);
	}
	return out;
}

void EGLRender::create_shaders(void)
{
   const char *fragShaderText =
      "precision mediump float;\n"
      "varying vec4 v_color;\n"
      "void main() {\n"
      "   gl_FragColor = v_color;\n"
      "}\n";
   const char *vertShaderText =
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
   checkEglError("Create frag shader");
   printGLError();
   glShaderSource(fragShader, 1, (const char **) &fragShaderText, NULL);
   checkEglError("Get Shader source");
   printGLError();
   glCompileShader(fragShader);
   checkEglError("Compile Shader source");
   printGLError();
   glGetShaderiv(fragShader, GL_COMPILE_STATUS, &stat);
   if (!stat) {
      printf("Error: fragment shader did not compile!\n");
      checkEglError("Get Shader compile status");
      printGLError();
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


void
EGLRender::init(void)
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
void EGLRender::make_headless_window(EGLDisplay egl_dpy,
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

EGLDisplay EGLRender::eglGetDisplay_(NativeDisplayType nativeDisplay)
{
  EGLDisplay eglDisplay = eglGetDisplay(nativeDisplay);
  checkEglError("Failed to Get Display: eglGetDisplay");
  std::cerr << "Failback to eglGetDisplay" << std::endl;
  return eglDisplay;
}

int EGLRender::_init()
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

void EGLRender::finish()
{
	eglDestroyContext(egl_dpy, egl_ctx);
	eglDestroySurface(egl_dpy, egl_surf);
	eglTerminate(egl_dpy);
}
