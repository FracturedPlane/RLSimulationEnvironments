local linuxLibraryLoc = "../external/"
local windowsLibraryLoc = "../../library/"

local action = _ACTION or ""
local todir = "./" .. action
-- local todir = "./"

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

local linuxLibraryLoc = "./external/"
local windowsLibraryLoc = "../library/"

solution "eglRender"
	configurations { 
		"Debug",
		"Release"
	}
	
	platforms {
		"x64"
	}
	location ("./")

	-- extra warnings, no exceptions or rtti
	flags { 
		"Symbols"
	}
	-- debug configs
	configuration { "Debug*"}
		defines { "DEBUG" }
		flags {
			"Symbols",
			Optimize = Off
		}
		targetdir ( "./x64/Debug" )
 
 	-- release configs
	configuration {"Release*"}
		defines { "NDEBUG" }
		flags { "Optimize" }
		targetdir ( "./x64/Release" )

	-- windows specific
	configuration { "linux", "Debug*", "gmake"}
        buildoptions { "-ggdb -fPIC" }
		linkoptions { 
			-- "-stdlib=libc++" ,
			"-Wl,-rpath," .. path.getabsolute("lib")
		}
		targetdir ( "./x64/Debug" )

	configuration { "linux", "Release*", "gmake"}
        buildoptions { "-ggdb -fPIC" }
		linkoptions { 
			-- "-stdlib=libc++" ,
			"-Wl,-rpath," .. path.getabsolute("lib")
		}
		targetdir ( "./x64/Release" )

project "eglRenderer"
	language "C++"
	kind "SharedLib"

	targetdir ( "../lib" )
	targetname ("_eglRenderer")
	targetprefix ("")
	files { 
		-- Source files for this project
		"./*.cpp",
	}
	excludes 
	{
		"Main.cpp",
	}	
	includedirs { 
		"./",
	}
	links {
	}

	defines {
		"_CRT_SECURE_NO_WARNINGS",
		"_SCL_SECURE_NO_WARNINGS",
        "__Python_Export__",
		"USE_OpenGLES",
		"HAVE_BUILTIN_SINCOS"
	}

	-- targetdir "./"
	buildoptions("-std=c++0x -ggdb -fPIC" )	

	-- linux library cflags and libs
	configuration { "linux", "gmake" }
		buildoptions { 
			" -fPIC",
		}
		linkoptions { 
			"-Wl,-rpath," .. path.getabsolute("../lib") ,
			" -fPIC",
		}
		libdirs { 
			"/usr/lib/nvidia-390",
			"/usr/lib/nvidia-396",
		}
		
		includedirs { 
            "/usr/include/python3.6m",
		}
		defines {
			"_LINUX_",
		}
			-- debug configs
		configuration { "linux", "Debug*", "gmake"}
			links {
				"m",
				"glut",
				"X11",
				"python3.6m",
				"EGL",
				"OpenGL",
			}
	 
	 	-- release configs
		configuration { "linux", "Release*", "gmake"}
			defines { "NDEBUG" }
			links {
				"m",
				"glut",
				"X11",
				"python3.6m",
				"EGL",
				"OpenGL",
			}
	
