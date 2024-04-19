%module relokate
%include "typemaps.i"
%{
    #define SWIG_FILE_WITH_INIT
    #include "relokate.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}
%apply (double* ARGOUT_ARRAY1, int DIM1) {(double* Res_Tx, int n1),(double* Res_Ty, int n2),(double* ErrMin, int n3)};
%apply (double* IN_ARRAY1, int DIM1) {(double* Px, int PSize)};
%apply (double* IN_ARRAY1, int DIM1) {(double* Py, int PSize2)};
%apply (double* IN_ARRAY1, int DIM1) {(double* Tx, int TSize)};
%apply (double* IN_ARRAY1, int DIM1) {(double* Ty, int TSize2)};
%include "relokate.h"