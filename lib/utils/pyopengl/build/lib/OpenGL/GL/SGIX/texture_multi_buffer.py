'''OpenGL extension SGIX.texture_multi_buffer

This module customises the behaviour of the 
OpenGL.raw.GL.SGIX.texture_multi_buffer to provide a more 
Python-friendly API

Overview (from the spec)
	
	This extension provides an API for the application to specify that
	the OpenGL should handle multiple textures in such a way that,
	wherever possible, a texture definition or redefinition can occur
	in parallel with rendering that uses a different texture.
	
	The texture_object extension allows the simultaneous definition
	of multiple textures; any texture that is not being used for 
	rendering can, in principle, have its definition or operations
	in its definition (e.g. downloading to hardware) occur in parallel
	with the use of another texture. This is true as long as all
	redefinitions strictly follow any use of the previous definition.
	
	Conceptually this is similar to frame buffer double-buffering,
	except that the intent here is to simply provide a hint to the
	OpenGL to promote such double-buffering if and wherever possible.
	The effect of such a hint is to speed up operations without
	affecting the result. The user on any particular system must be
	knowledgable and prepared to accept any trade-offs which follow
	from such a hint.
	
	GL_FASTEST in this context means that texture multi-buffering
	is being used whenever possible to improve performance. 
	Generally, textures that are adjacent in a sequence of multiple
	texture definitions have the greatest chance of being in 
	different buffers. The number of buffers available at any time
	depends on various factors, such as the machine being used and
	the textures' internal formats.

The official definition of this extension is available here:
http://www.opengl.org/registry/specs/SGIX/texture_multi_buffer.txt
'''
from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.SGIX.texture_multi_buffer import *
from OpenGL.raw.GL.SGIX.texture_multi_buffer import _EXTENSION_NAME

def glInitTextureMultiBufferSGIX():
    '''Return boolean indicating whether this extension is available'''
    from OpenGL import extensions
    return extensions.hasGLExtension( _EXTENSION_NAME )


### END AUTOGENERATED SECTION