
#include "eglRender.h"

int main(int argc, char *argv[])
{
	_init();

   draw();
   for (size_t i = 0; i < 5; i++)
   {
		// view_rotx += 5.0;
		setPosition(i * 0.02, i * 0.02, 0);
		setPosition2(i * -0.02, i * -0.02, 0);
		draw();
		save_PPM();
   }

   finish();
   return 0;
}
