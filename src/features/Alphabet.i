%{
#include "features/Alphabet.h" 
%}

#ifdef HAVE_PYTHON
%include "lib/numpy.i"
%apply (LONG** ARGOUT1, INT* DIM1) {(LONG** h, INT* len)};
#endif

%rename(Alphabet) CAlphabet;

%include "features/Alphabet.h" 
