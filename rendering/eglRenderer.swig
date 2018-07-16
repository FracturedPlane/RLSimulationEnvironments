
 /* simAdapter.i */
 %module eglRenderer
 %{
 /* Put header files here or function declarations like below */
 #define SWIG_FILE_WITH_INIT
 #include <vector>

#include "EGLRender.h"
 
%}
%include "std_string.i"
%include "std_vector.i"
 // %include "std_pair.i"
 // Instantiate templates used by example
namespace std {
   %template(IntVector) vector<int>;
   %template(DoubleVector) vector<double>;
   %template(FloatVector) vector<float>;
   %template(StringVector) vector<string>;
   %template(ConstCharVector) vector<const char*>;
   %template(UnsighnedCharVector) vector<unsigned char>;
   %template(UnsighnedCharVectorVector) vector<vector<unsigned char> >;
   %template(UnsighnedCharVectorVectorVector) vector<vector<vector<unsigned char> > >;
}

%include "EGLRender.h"

 