
 /* simAdapter.i */
 %module eglRenderer
 %{
 /* Put header files here or function declarations like below */
 #define SWIG_FILE_WITH_INIT
 #include <vector>

#include "eglRender.h"
 
%}
%include "std_string.i"
%include "std_vector.i"
 // %include "std_pair.i"
 // Instantiate templates used by example
namespace std {
   %template(IntVector) vector<int>;
   %template(DoubleVector) vector<double>;
   %template(StringVector) vector<string>;
   %template(ConstCharVector) vector<const char*>;
}

%include "eglRender.h"
// %include "../sim/TerrainRLCharController.h"
// %include "../sim/SimCharacter.h"
// %include "../util/MathUtil.h"

 