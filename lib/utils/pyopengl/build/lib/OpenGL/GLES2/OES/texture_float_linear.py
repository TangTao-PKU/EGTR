'''OpenGL extension OES.texture_float_linear

This module customises the behaviour of the 
OpenGL.raw.GLES2.OES.texture_float_linear to provide a more 
Python-friendly API

Overview (from the spec)
	
	These extensions expand upon the OES_texture_half_float and
	OES_texture_float extensions by allowing support for LINEAR
	magnification filter and LINEAR, NEAREST_MIPMAP_LINEAR,
	LINEAR_MIPMAP_NEAREST and LINEAR_MIPMAP_NEAREST minification
	filters.
	
	When implemented against OpenGL ES 3.0 or later versions, sized 32-bit
	floating-point formats become texture-filterable. This should be noted
	by, for example, checking the ``TF'' column of table 8.13 in the ES 3.1
	Specification (``Correspondence of sized internal formats to base
	internal formats ... and use cases ...'') for the R32F, RG32F, RGB32F,
	and RGBA32F formats.

The official definition of this extension is available here:
http://www.opengl.org/registry/specs/OES/texture_float_linear.txt
'''
from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GLES2 import _types, _glgets
from OpenGL.raw.GLES2.OES.texture_float_linear import *
from OpenGL.raw.GLES2.OES.texture_float_linear import _EXTENSION_NAME

def glInitTextureFloatLinearOES():
    '''Return boolean indicating whether this extension is available'''
    from OpenGL import extensions
    return extensions.hasGLExtension( _EXTENSION_NAME )


### END AUTOGENERATED SECTION