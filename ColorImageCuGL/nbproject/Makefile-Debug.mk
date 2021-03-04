#
# Generated Makefile - do not edit!
#
# Edit the Makefile in the project folder instead (../Makefile). Each target
# has a -pre and a -post target defined where you can add customized code.
#
# This makefile implements configuration specific macros and targets.


# Environment
MKDIR=mkdir
CP=cp
GREP=grep
NM=nm
CCADMIN=CCadmin
RANLIB=ranlib
CC=gcc
CCC=nvcc
CXX=nvcc
FC=gfortran
AS=as

# Macros
CND_PLATFORM=GNU-Linux
CND_DLIB_EXT=so
CND_CONF=Debug
CND_DISTDIR=dist
CND_BUILDDIR=build

# Include project Makefile
include Makefile

# Object Directory
OBJECTDIR=${CND_BUILDDIR}/${CND_CONF}/${CND_PLATFORM}

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/main.o


# C Compiler Flags
CFLAGS=

# CC Compiler Flags
CCFLAGS=
CXXFLAGS=

# Fortran Compiler Flags
FFLAGS=

# Assembler Flags
ASFLAGS=

# Link Libraries and Options
LDLIBSOPTIONS=-L/opt/arrayfire/lib -lGL -lGLEW -lGLEW -lGLU -lGLX -lNvAppBase -lNvAppBaseD -lNvAssetLoader -lNvAssetLoaderD -lNvGLUtils -lNvGLUtilsD -lNvGamepad -lNvGamepadD -lNvModel -lNvModelD -lNvUI -lNvUID -lOpenCL -laccinj64 -laf -lafcpu -lafcuda -lafopencl -lcublas -lcublasLt -lcuda -lcudart -lcufft -lcufftw -lcuinj64 -lcurand -lcusolver -lcusolverMg -lcusparse -lforge -lfreeimage-3.18.0 -lglad -lglfw -lglfw3 -lglm_shared -lglut -lgomp -lharfbuzz -lharfbuzzD -liomp5 -lmkl_avx -lmkl_avx2 -lmkl_avx512 -lmkl_core -lmkl_def -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_mc -lmkl_mc3 -lnppc -lnppial -lnppicc -lnppicom -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc -lnpps -lnvToolsExt -lnvblas -lnvgraph -lnvidia-ml -lnvjpeg -lnvrtc-builtins -lnvrtc -lnvvm -lomp -ltrace

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/colorimagecugl

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/colorimagecugl: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/colorimagecugl ${OBJECTFILES} ${LDLIBSOPTIONS}

${OBJECTDIR}/main.o: main.cu
	${MKDIR} -p ${OBJECTDIR}
	$(COMPILE.cc) -g -I/opt/arrayfire/include -o ${OBJECTDIR}/main.o main.cu

# Subprojects
.build-subprojects:

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}

# Subprojects
.clean-subprojects:
