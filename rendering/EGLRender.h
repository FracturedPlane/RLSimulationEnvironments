/**************************************************************************
 *
 * Copyright 2018 Glen Berth.
 * All Rights Reserved.
 *
 **************************************************************************/

/*
 * Draw some triangles with EGL and OpenGL ES 2.x
 */
#ifndef EGLRENDER_H_
#define EGLRENDER_H_

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <GLES2/gl2.h>  /* use OpenGL ES 3.x */
#define EGL_EGLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <vector>
#include <iostream>

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
 * row major matrix? I guess not....
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

/*
 * row major matrix? I guess not....
 */
static void
make_identity_matrix(GLfloat *m)
{
   int i;
   for (i = 0; i < 16; i++)
      m[i] = 0.0;

   m[0] = m[5] = m[10] = m[15] = 1.0;
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

static inline const char * GetGLErrorString(GLenum error)
{
	const char *str;
	switch( error )
	{
		case GL_NO_ERROR:
			str = "GL_NO_ERROR";
			break;
		case GL_INVALID_ENUM:
			str = "GL_INVALID_ENUM";
			break;
		case GL_INVALID_VALUE:
			str = "GL_INVALID_VALUE";
			break;
		case GL_INVALID_OPERATION:
			str = "GL_INVALID_OPERATION";
			break;
#ifdef __gl_h_
		case GL_STACK_OVERFLOW:
			str = "GL_STACK_OVERFLOW";
			break;
		case GL_STACK_UNDERFLOW:
			str = "GL_STACK_UNDERFLOW";
			break;
		case GL_OUT_OF_MEMORY:
			str = "GL_OUT_OF_MEMORY";
			break;
		case GL_TABLE_TOO_LARGE:
			str = "GL_TABLE_TOO_LARGE";
			break;
#endif
#if GL_EXT_framebuffer_object
		case GL_INVALID_FRAMEBUFFER_OPERATION_EXT:
			str = "GL_INVALID_FRAMEBUFFER_OPERATION_EXT";
			break;
#endif
		default:
			str = "(ERROR: Unknown Error Enum)";
			break;
	}
	return str;
}

#define printGLError(){ \
	GLenum  err = glGetError();; \
    if (err != GL_NO_ERROR) \
    { \
    	const char *str = GetGLErrorString(err); \
		std::cout << "error: " << str << std::endl; \
    } \
}
/*
#define printGLError()
{
// #if DEBUG
	GLenum  err = glGetError();
    if (err != GL_NO_ERROR)
	{
		const char *str = GetGLErrorString(err);
		std::cout << "error: " << str << std::endl;
		// NSLog(@"%@", [NSString stringWithCString:str encoding:NSUTF8StringEncoding]);
	}
// #endif
}
*/

#define FLOAT_TO_FIXED(X)   ((X) * 65535.0)

class EGLRender
{

public:

	EGLRender();
	virtual ~EGLRender();

	virtual void setPosition(float xs, float ys, float zs);
	virtual void setPosition2(float xs, float ys, float zs);
	virtual void setCameraPosition(float xs, float ys, float zs);

	virtual void setDrawAgent(bool draw_);
	virtual void setDrawObject(bool draw_);

	// dumps a PPM raw (P6) file on an already allocated memory array
	virtual void DumpPPM(FILE *fp, int width, int height);
	virtual int save_PPM();

	virtual void draw(void);

/* new window size or exposure */
	virtual void reshape(int width, int height);

	virtual std::vector<unsigned char> getPixels(size_t x_start, size_t y_start, size_t width, size_t height);
	virtual void create_shaders(void);
	virtual void init(void);

/*
	 * Create an RGB, double-buffered Headless window.
	 * context handle.
	 */
	virtual void make_headless_window(EGLDisplay egl_dpy,
				  const char *name,
				  int x, int y, int width, int height,
				  EGLContext *ctxRet,
				  EGLSurface *surfRet);

	virtual EGLDisplay eglGetDisplay_(NativeDisplayType nativeDisplay=EGL_DEFAULT_DISPLAY);
	virtual int _init();
	virtual void finish();

	int m_frameCount;
	size_t winWidth = 1000, winHeight = 1000;
	EGLSurface egl_surf;
	EGLContext egl_ctx;
	EGLDisplay egl_dpy;
	char *dpyName = NULL;
	GLboolean printInfo = GL_FALSE;
	EGLint egl_major, egl_minor;
	int i;
	const char *s;
	int cudaIndexDesired = 0;

	bool drawAgent = true, drawObject = true;

	GLfloat view_rotx = 0.0, view_roty = 0.0;
	GLfloat view_transx = 0.0, view_transy = 0.0, view_transz = 0.0;
	GLfloat view_transx2 = 0.0, view_transy2 = 0.0, view_transz2 = 0.0;
	GLfloat camPos[3] = {0.0, 0.0, 0.0};

	GLint u_matrix = -1;
	GLint attr_pos = 0, attr_color = 1;

};
#endif /* EGLRender_H_ */
