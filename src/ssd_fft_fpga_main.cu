/*
Template-based speed-sign detection is the proprietary property of The Regents of the University of California ("The Regents") and is copyright © 2018 
The Regents of the University of California, Davis campus. All Rights Reserved. Redistribution and use in source and binary forms, with or without 
modification, are permitted by nonprofit educational or research institutions for noncommercial use only, provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation 
and/or other materials provided with the distribution. 
* The name or other trademarks of The Regents may not be used to endorse or promote products derived from this software without specific prior written 
permission.
The end-user understands that the program was developed for research purposes and is advised not to rely exclusively on the program for any reason.
THE SOFTWARE PROVIDED IS ON AN "AS IS" BASIS, AND THE REGENTS HAVE NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS. 
THE REGENTS SPECIFICALLY DISCLAIM ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, EXEMPLARY OR 
CONSEQUENTIAL DAMAGES, INCLUDING BUT NOT LIMITED TO  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES, LOSS OF USE, DATA OR PROFITS, OR BUSINESS INTERRUPTION, 
HOWEVER CAUSED AND UNDER ANY THEORY OF LIABILITY WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT 
OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
If you do not agree to these terms, do not download or use the software.  This license may be modified only in a writing signed by authorized signatory of 
both parties.
For license information please contact copyright@ucdavis.edu re T11-005.
*/




////////////////////////////////////////////////////////////////////////////////////////////////////////////////// AOCL
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

using namespace aocl_utils;

#define STRING_BUFFER_LEN 1024
#define AOCL_ALIGNMENT 64


// Runtime constants
// Used to define the work set over which this kernel will execute.
static const size_t work_group_size = 8;  // 8 threads in the demo workgroup
										  // Defines kernel argument value, which is the workitem ID that will
										  // execute a printf call
static const int thread_id_to_output = 2;

// OpenCL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_command_queue queue2 = NULL;
static cl_command_queue queue3 = NULL;
static cl_command_queue queue4 = NULL;
static cl_kernel fetch_kernel0 = NULL;
static cl_kernel fetch_kernel1 = NULL;
static cl_kernel fft_kernel0 = NULL;
static cl_kernel fft_kernel1 = NULL;
static cl_kernel transpose_kernel0 = NULL;
static cl_kernel transpose_kernel1 = NULL;
static cl_kernel extraCol_kernel0 = NULL;
static cl_kernel extraCol_kernel1 = NULL;
static cl_kernel kthLaw_kernel = NULL;
static cl_kernel pointWiseMul_kernel = NULL;
static cl_kernel pointWiseMul_kernel2 = NULL;
static cl_kernel ComplexScale_kernel = NULL;
static cl_kernel convertChar4ToFloatDoConGam_kernel = NULL;
static cl_kernel fixDeadPixels_kernel = NULL;
static cl_kernel max_k_kernel = NULL;
static cl_program program = NULL;

// Function prototypes
bool init();
void AOCLcleanup();
static void device_info_ulong(cl_device_id device, cl_device_info param, const char* name);
static void device_info_uint(cl_device_id device, cl_device_info param, const char* name);
static void device_info_bool(cl_device_id device, cl_device_info param, const char* name);
static void device_info_string(cl_device_id device, cl_device_info param, const char* name);
static void display_device_info(cl_device_id device);
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//#define US_SIGNS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cublas.h>
#include <cufft.h>
//#include <cutil.h>
#include "ssd_fft_gpu_kernel.cu"
//#define BUILD_DLL
#include <GL/glew.h>
#include <GL/glut.h>
//#include "include/ssd_fft_gpu_dll.h"
#include <ssd_fft_gpu_common.h>
#include "include/ssd_fft_gpu.h"
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_functions.h>
#include <helper_timer.h>

#define CUTFalse false
#define CUTTrue true
#define CUTBoolean bool
#define CUDA_SAFE_CALL checkCudaErrors
#define CUFFT_SAFE_CALL checkCudaErrors
#define CUT_SAFE_CALL checkCudaErrors
#define CUT_CHECK_ERROR getLastCudaError
#define cutComparefe sdkCompareL2fe
#define cutCreateTimer sdkCreateTimer


extern "C"
int CLAHE(unsigned char* pImage, unsigned int uiXRes, unsigned int uiYRes, unsigned char Min,
	unsigned char Max, unsigned int uiNrX, unsigned int uiNrY,
	unsigned int uiNrBins, float fCliplimit);

#define gd_afCompFlt(iIPIndx, iSzIndx, iFltIndx)  gd_afCompFlt + ((giTplH * giTplW * giNumIPRot * giNumSz) * iFltIndx) +  ((giTplH * giTplW * giNumIPRot) * iSzIndx) + ((giTplH * giTplW) * iIPIndx)
#define d_pafWholeTplFFT(iIPAbsIndx, iSzIndx, iFltAbsIndx) d_pafWholeTplFFT + ((giPadScnH * giPadScnW * giNumIPInFirst * giNumSz) * iFltAbsIndx) + ((giPadScnH * giPadScnW * giNumIPInFirst) * iSzIndx) + ((giPadScnH * giPadScnW) * iIPAbsIndx)
#define gd_afWholeTplFFT(iIPAbsIndx, iSzIndx, iFltAbsIndx) gd_afWholeTplFFT + ((giPadScnH * giPadScnW * giNumIPInFirst * giNumSz) * iFltAbsIndx) + ((giPadScnH * giPadScnW * giNumIPInFirst) * iSzIndx) + ((giPadScnH * giPadScnW) * iIPAbsIndx)
#define d_pafPartTplFFT(iIPIndx, iSzIndx, iFltIndx) d_pafPartTplFFT + ((giTplH * giTplW * giNumIPRot * giNumSz) * iFltIndx) + ((giTplH * giTplW * giNumIPRot) * iSzIndx) + ((giTplH * giTplW) * iIPIndx)
#define gd_afPartTplFFT(iIPIndx, iSzIndx, iFltIndx) gd_afPartTplFFT + ((giTplH * giTplW * giNumIPRot * giNumSz) * iFltIndx) + ((giTplH * giTplW * giNumIPRot) * iSzIndx) + ((giTplH * giTplW) * iIPIndx)
// OpenCL host can't see unify address, so these addresses are presented as offsets when needed 
#define cl_gd_afCompFlt(iIPIndx, iSzIndx, iFltIndx) (((giTplH * giTplW * giNumIPRot * giNumSz) * iFltIndx) +  ((giTplH * giTplW * giNumIPRot) * iSzIndx) + ((giTplH * giTplW) * iIPIndx))* sizeof(float)
#define cl_gd_afWholeTplFFT(iIPAbsIndx, iSzIndx, iFltAbsIndx) (((giPadScnH * giPadScnW * giNumIPInFirst * giNumSz) * iFltAbsIndx) + ((giPadScnH * giPadScnW * giNumIPInFirst) * iSzIndx) + ((giPadScnH * giPadScnW) * iIPAbsIndx))* sizeof(float)
#define cl_gd_afPartTplFFT(iIPIndx, iSzIndx, iFltIndx) (((giTplH * giTplW * giNumIPRot * giNumSz) * iFltIndx) + ((giTplH * giTplW * giNumIPRot) * iSzIndx) + ((giTplH * giTplW) * iIPIndx))* sizeof(float)
////////////////////////////////////////////////////////////////////////////////
// Global vars
////////////////////////////////////////////////////////////////////////////////
cl_int status;
size_t wgSize[3] = { 1, 1, 1 };
size_t gSize[3] = { 1, 1, 1 };
//trashold for the PSR (might be different for day and night)
const float gfPSRTrashold = 7.5f;
//params related to Majority Voting
//keep track of PSRs for giTrackingLen frames
const float giTrackingLen = 10;
float giFrameNo = 0;
int giNumFramesInAcc = 0; //number of frames that contribute to AccPSR
						  //max acc psr should be greater than gfAccPSRTrashold so that we can conclude that speed sign is recognized
float gfAccPSRTrashold = 0;
//factor which determines additional confidence due to IP (if IP is equal to prevIP increase conf). 
//makes sense when different IP Rots are defined.
const float gfAddConfIPFac = 0.25;
//factor which determines additional confidence due to Sz (if Sz is larger to prevSz increase conf). 
const float gfAddConfEqSzFac = 0.5;
const float gfAddConfGrSzFac = 1.25;

typedef struct AccRes_struct
{
	float fAccConf;
	int iPrevIP;
	int iPrevSz;
}AccRes_struct_t;

AccRes_struct_t* gastAccRes;

//for PSR calculation define sidelobe
//area = frame+mask
const int giAreaH = 20;
const int giMaskH = 4;

const int	giScnSz = giScnW * giScnH;
const int	giScnSzPad = giPadScnW * giPadScnH;
const int	giScnMemSzReal = giScnSz * sizeof(cufftReal);
const int	giScnMemSzRealPad = giScnSzPad * sizeof(cufftReal);
const int   giScnMemSzCmplx = giScnSz * sizeof(cufftComplex);
const int   giScnMemSzCmplxPad = giScnSzPad * sizeof(cufftComplex);
const int   giScnMemSzUChar = giScnSz * sizeof(unsigned char);
const int	giAreaMemSzReal = giAreaH * giAreaH * sizeof(cufftReal);
const int	giScnOffset = giScnBegY * giScnW;
const int   giOrigScnMemSzUChar = giOrigScnSz * sizeof(unsigned char);

//directory where scene and templates are
char g_sPathBegin[50] = "../../cpuResults/";
char g_sPath[100];
//directory where stats files will be stored
char g_sStatsPathBegin[50] = "../stats/ssd_gpu_stats/fft_results/";
char g_sStatsPath[100];
FILE* g_fStatsFile;
//directory where scnbin files will be stored
#ifdef US_SIGNS
char g_sScnBinPathBegin[50] = "../convert_pgm_to_RawVideo/raw/";
#else
char g_sScnBinPathBegin[60] = "../../../copied15May17/EU_raw(savedRealisFilesAsBin)/";
#endif
char g_sScnBinPath[100];
FILE* g_fScnBin;

#ifdef REALTIME
unsigned long g_ulPrevTimeStamp = 0;
const int g_iRuntime = 124; //update this if you make performance improvements
const float	gfAccPSRTrasholdSpecialReal = 11.0f;
#endif

//unsigned int guiParTim;
StopWatchInterface *guiParTim;
//unsigned int guiKerTim;
StopWatchInterface *guiKerTim;
double g_dRunsOnGPUTotalTime;
double g_dTotalKerTime;
double g_dClaheTime;
double g_time;
int giTplH, giTplW, giTplSz, giTplWMemSz, giTplMemSzReal, giTplMemSzCmplx;
int giTplSzPad, giTplWMemSzPad, giTplMemSzRealPad, giTplMemSzCmplxPad;
int giNumIPRot, giNumSz, giNumOrigFlt, giNumSngCompFlt;

typedef struct CompFlt_struct
{
	float* h_afData;
	int iH;
	int iW;
	int iNumIPRot;
	int iNumSz;
	int iNumOrigFlt;
	int iNumMulCompFlt;
	int iDataSz;
	int iDataMemSz;
	int* aiIPAngs;
	int* aiTplCols;
	int* aiTpl_no;
}CompFlt_struct_t;

CompFlt_struct_t gstCompFlt;

int giPartMaxGDx, giWholeMaxGDx;
cufftReal
*gd_pfMax,
*gd_afBlockMaxs;
int
*gd_piMaxIdx,
*gd_aiBlockMaxIdxs;
////////////////////////////////////////////////////////////////////////////////
// Following variables have been made global, so that we can divide the main function
// to init, fingBestTpl, and exit
////////////////////////////////////////////////////////////////////////////////


//typedef float cufftReal;
cufftReal
*gd_afScnPartIn,
*gh_afArea,
*gd_afCompFlt,
*gd_afPadTplIn,
*gd_afPadScnIn,
*gd_afPadScnInPad,
*gd_afCorr;
cl_mem
cl_afPadScnInPad,
cl_afPadScnOutPad,
d_tmp0, d_tmp1,// for FFTs
d_tmp02, d_tmp12,
cl_gd_ac4Scn,
cl_gd_afCompFlt,
cl_gd_afPadScnIn,
cl_gd_afCorr,
cl_gd_afPadScnOut,
cl_gd_afMul,
cl_gd_afScnPartIn,
cl_gd_afScnPartOut,
cl_gd_pfMax,
cl_gd_piMaxIdx,
cl_gd_afWholeTplFFT,
cl_gd_afPartTplFFT; 

//typedef float cufftComplex[2];
cufftComplex
*gd_afPadScnInPadC,
*gd_afScnPartInC,
*gd_afScnPartOut,
*gd_afPadTplOut,
*gd_afPadScnOut,
*gd_afPadScnOutPad,
*gd_afWholeTplFFT,
*gd_afPartTplFFT,
*gd_afMul,
*gd_afCorrC;

unsigned char
*gh_acScn;

uchar4
*gd_ac4Scn;

cufftHandle
ghFFTplanWholeFwd,
ghFFTplanWholeInv,
ghFFTplanPartFwd,
ghFFTplanPartInv;

dim3 gdThreadsConv(1, 1, 1);
dim3 gdBlocksConv(1, 1);
dim3 gdThreadsDead(1, 1, 1);
dim3 gdBlocksDead(1, 1);
dim3 gdThreadsWhole(1, 1, 1);
dim3 gdBlocksWhole(1, 1);
dim3 gdThreadsPart(1, 1, 1);
dim3 gdBlocksPart(1, 1);

int
giBegIdxIPInFirst,
giEndIdxIPInFirst,
giNumIPInFirst,
giBegIdxIPInSecond,
giEndIdxIPInSecond;

//adjust contrast and do gamma correction 
bool gbConGam = 0;
//fix the dead pixels in the given scene if we are processing a video 
bool gbFixDead = 1;

//params related to ConGam
#define LUTSIZE 256
float gfLUT[LUTSIZE];
unsigned char gacLUT[LUTSIZE];
float gfLIn = 0.2f;//0.4f;//0.2f;
float gfHIn = 0.8f;//0.6f;//0.8f;
float gfLOut = 0.0f;
float gfHOut = 1.0f;
float gfG = 2.5f;//0.5f;//2.5f;
float time1, accumulate;
//pass the found Speed Limit Number to the callee (GUI)
int giSLCurFrm = -1; //SL found in the current frame (-1 means no SL)
int giSLResult = -1; //SL found as a result of temporal integration (-1 means no SL)
int giShowClaheGUI = 0; //allow ssd_fft_GUI to turn on/off CLAHE showing (to capture the CLAHE effect in DAGM video) if -1 show, if 0 do not.
char gacClipName[11];
////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
//Round a / b to nearest higher integer value
int iDivUp(int iX, int iY) {
	return (iX % iY != 0) ? (iX / iY + 1) : (iX / iY);
}

//Align a to nearest higher multiple of b
int iAlignUp(int iX, int iY) {
	return (iX % iY != 0) ? (iX - iX % iY + iY) : iX;
}

//convert 1D Index to 2D Coordinates
void Indx2Coord(int iImgW, int iIndx, int* iRow, int* iCol)
{
	//cuda is row major and zero-based
	*iCol = iIndx%iImgW;
	//regular division of integer returns floor
	*iRow = iIndx / iImgW;
}

//convert 2D Coordinates to 2D Index
void Coord2Indx(int iImgW, int iRow, int iCol, int* iIndx)
{
	//cuda is row major and zero-based
	*iIndx = (iImgW*iRow) + iCol;
}

//assign values to rectangle specified by coord
void assignVal(int iImgW, float* afImg, int4 aiCoord, float fVal)
{
	int iIndx;
	for (int iRow = aiCoord.x; iRow <= aiCoord.y; iRow++)
	{
		for (int iCol = aiCoord.z; iCol <= aiCoord.w; iCol++)
		{
			Coord2Indx(iImgW, iRow, iCol, &iIndx);
			afImg[iIndx] = fVal;
		}
	}
}

//sum elements
float sum(float* afImg, int iSz)
{
	float fTotal = 0;
	for (int i = 0; i<iSz; i++)
	{
		fTotal += afImg[i];
	}
	return fTotal;
}

//get surrounding coordinates of the areas centered around a point 
int4 getSurrCoord(int iRow, int iCol, int iSurrH, int iNumCols, int iNumRows)
{
	//TODO: maybe I should shift area if it is at border, to produce lower PSR?
	int iHalfSurrH = iSurrH / 2;
	int iSurrRowBeg = iRow - iHalfSurrH + 1;
	if (iSurrRowBeg < 0) iSurrRowBeg = 0;
	int iSurrRowEnd = iRow + iHalfSurrH;
	if (iSurrRowEnd >= iNumRows) iSurrRowEnd = iNumRows - 1;
	int iSurrColBeg = iCol - iHalfSurrH + 1;
	if (iSurrColBeg < 0) iSurrColBeg = 0;
	int iSurrColEnd = iCol + iHalfSurrH;
	if (iSurrColEnd >= iNumCols) iSurrColEnd = iNumCols - 1;
	int4 aiAreaCoord = { iSurrRowBeg, iSurrRowEnd, iSurrColBeg, iSurrColEnd };

	return aiAreaCoord;
}

//make FFT size power of two 
int getPOTSz(int iSz) {
	//Highest non-zero bit position of iSz
	int iHiBit;
	//Neares lower and higher powers of two numbers for iSz
	unsigned int uiLowPOT, uiHiPOT;

	//Find highest non-zero bit (1U is unsigned one)
	for (iHiBit = 31; iHiBit >= 0; iHiBit--)
		if (iSz & (1U << iHiBit)) break;

	//No need to align, if already power of two
	uiLowPOT = 1U << iHiBit;
	if (uiLowPOT == iSz) return iSz;

	//Align to a nearest higher power of two, if the size is small enough,
	//else align only to a nearest higher multiple of 512,
	//in order to save computation and memory bandwidth
	uiHiPOT = 1U << (iHiBit + 1);
	if (uiHiPOT <= 1024)
		return uiHiPOT;
	else
		return iAlignUp(iSz, 512);
}


//Get the full path name
char* getFullPathOfFile(char* pcFileName)
{
	strcpy(g_sPath, g_sPathBegin);
	strcat(g_sPath, pcFileName);
	return g_sPath;
}


CompFlt_struct_t readCompFlt()
{
	CompFlt_struct_t gstCompFlt;
	FILE *fCompFlts = fopen(getFullPathOfFile("CompFlts.bin"), "rb");
	fread(&gstCompFlt.iH, sizeof(int), 1, fCompFlts);
	fread(&gstCompFlt.iW, sizeof(int), 1, fCompFlts);
	fread(&gstCompFlt.iNumIPRot, sizeof(int), 1, fCompFlts);
	fread(&gstCompFlt.iNumSz, sizeof(int), 1, fCompFlts);
	fread(&gstCompFlt.iNumOrigFlt, sizeof(int), 1, fCompFlts);
	fread(&gstCompFlt.iNumMulCompFlt, sizeof(int), 1, fCompFlts);
	int iNumTpl = gstCompFlt.iNumOrigFlt - gstCompFlt.iNumMulCompFlt;
	int iNumIPRotMemSz = gstCompFlt.iNumIPRot * sizeof(int);
	int iNumSzMemSz = gstCompFlt.iNumSz * sizeof(int);
	int iNumTplMemSz = iNumTpl * sizeof(int);
	int iNumAccResMemSz = iNumTpl * sizeof(AccRes_struct_t);
	gstCompFlt.iDataSz = gstCompFlt.iH * gstCompFlt.iW * gstCompFlt.iNumIPRot * gstCompFlt.iNumSz * gstCompFlt.iNumOrigFlt;
	gstCompFlt.iDataMemSz = gstCompFlt.iDataSz * sizeof(float);
#ifdef PINNED_MEM
	CUDA_SAFE_CALL(cudaMallocHost((void**)&gstCompFlt.aiIPAngs, iNumIPRotMemSz));
	CUDA_SAFE_CALL(cudaMallocHost((void**)&gstCompFlt.aiTplCols, iNumSzMemSz));
	CUDA_SAFE_CALL(cudaMallocHost((void**)&gstCompFlt.aiTpl_no, iNumTplMemSz));
	CUDA_SAFE_CALL(cudaMallocHost((void**)&gstCompFlt.h_afData, gstCompFlt.iDataMemSz));
	CUDA_SAFE_CALL(cudaMallocHost((void**)&gastAccRes, iNumAccResMemSz));

#else
	gstCompFlt.aiIPAngs = (int *)malloc(iNumIPRotMemSz);
	gstCompFlt.aiTplCols = (int *)malloc(iNumSzMemSz);
	gstCompFlt.aiTpl_no = (int *)malloc(iNumTplMemSz);
	gstCompFlt.h_afData = (float *)_aligned_malloc(gstCompFlt.iDataMemSz, AOCL_ALIGNMENT);
	gastAccRes = (AccRes_struct_t *)malloc(iNumAccResMemSz);
#endif
	fread(gstCompFlt.aiIPAngs, sizeof(int), gstCompFlt.iNumIPRot, fCompFlts);
	fread(gstCompFlt.aiTplCols, sizeof(int), gstCompFlt.iNumSz, fCompFlts);
	fread(gstCompFlt.aiTpl_no, sizeof(int), iNumTpl, fCompFlts);

	fread(gstCompFlt.h_afData, sizeof(float), gstCompFlt.iDataSz, fCompFlts);
	fclose(fCompFlts);
	//initialized the accpsr to zero
	memset(gastAccRes, '\0', iNumAccResMemSz);
	return gstCompFlt;
}
void getKernelDims(int iBlockDimX, int iSz, dim3* dThreads, dim3* dBlocks)
{
	(*dThreads).x = iBlockDimX;
	int iGDx = (iSz) % (iBlockDimX) > 0 ? ((iSz) / (iBlockDimX)) + 1 : (iSz) / (iBlockDimX);
	(*dBlocks).x = iGDx;
	return;
}


inline void InitKerTim(int iSz)
{
#ifdef KERTIM
	if (iSz == 5)
	{
		CUDA_SAFE_CALL(cudaThreadSynchronize());
		sdkResetTimer(&guiKerTim);
		sdkStartTimer(&guiKerTim);
	}
#endif
}

inline void WrapKerTim(char* sKerName, int iSz)
{
#ifdef KERTIM
	if (iSz == 5) //1(copyscn convert fix), 2(1stPassInit), 3(2ndPassInit), giScnSz (1stLoop), giTplSz(2ndLoop)
	{
		CUDA_SAFE_CALL(cudaThreadSynchronize());
		sdkStopTimer(&guiKerTim);
		double dTime = sdkGetTimerValue(&guiKerTim);
		printf("%s time: %f msecs.\n", sKerName, dTime);
		g_dTotalKerTime += dTime;
	}
#endif
}


#define LOGNScn 9
#define LOGNTpl 6
#define NS (1 << LOGNScn) //  full sceen size
#define NT (1 << (LOGNTpl-1)) // template size

void transpose(cl_mem cl_input) // for FPGA FFT output to match with cuFFT output
{
	cufftComplex* h_tmp = (cufftComplex*)_aligned_malloc(giScnMemSzCmplxPad, AOCL_ALIGNMENT);
	cufftComplex* h_tmp2 = (cufftComplex*)_aligned_malloc(giScnMemSzCmplxPad, AOCL_ALIGNMENT);
	clEnqueueReadBuffer(queue, cl_input, CL_TRUE, 0, giScnMemSzCmplxPad, h_tmp, 0, NULL, NULL);

	for (int row = 0; row < 513; row++)
		for (int col = 0; col<512; col++)
		{
			h_tmp2[row + col * 513] = h_tmp[col + row * 512];
		}
	for (int i = 512 * 513; i < 512 * 1024; i++)
	{
		h_tmp2[i].x = 0;
		h_tmp2[i].y = 0;
	}
	clEnqueueWriteBuffer(queue, cl_input, CL_TRUE, 0, giScnMemSzCmplxPad, h_tmp2, 0, NULL, NULL);
}
void transpose2(cl_mem cl_input)
{
	cufftComplex* h_tmp = (cufftComplex*)_aligned_malloc(giScnMemSzCmplxPad, AOCL_ALIGNMENT);
	cufftComplex* h_tmp2 = (cufftComplex*)_aligned_malloc(giScnMemSzCmplxPad, AOCL_ALIGNMENT);
	clEnqueueReadBuffer(queue, cl_input, CL_TRUE, 0, giScnMemSzCmplxPad, h_tmp, 0, NULL, NULL);

	for (int row = 0; row < 512; row++)
		for (int col = 0; col<513; col++)
		{
			h_tmp2[row + col * 512] = h_tmp[col + row * 513];
		}
	for (int i = 512 * 513; i < 512 * 1024; i++)
	{
		h_tmp2[i].x = 0;
		h_tmp2[i].y = 0;
	}
	clEnqueueWriteBuffer(queue, cl_input, CL_TRUE, 0, giScnMemSzCmplxPad, h_tmp2, 0, NULL, NULL);
}

void transposeT(cl_mem cl_input)// template size
{
	cufftComplex* h_tmp = (cufftComplex*)_aligned_malloc(giTplMemSzCmplxPad, AOCL_ALIGNMENT);
	cufftComplex* h_tmp2 = (cufftComplex*)_aligned_malloc(giTplMemSzCmplxPad, AOCL_ALIGNMENT);
	clEnqueueReadBuffer(queue, cl_input, CL_TRUE, 0, giTplMemSzCmplxPad, h_tmp, 0, NULL, NULL);

	for (int row = 0; row < 33; row++)
		for (int col = 0; col<64; col++)
		{
			h_tmp2[row + col * 33] = h_tmp[col + row * 64];
		}
	for (int i = 64 * 33; i < 64 * 64; i++)
	{
		h_tmp2[i].x = 0;
		h_tmp2[i].y = 0;
	}
	clEnqueueWriteBuffer(queue, cl_input, CL_TRUE, 0, giTplMemSzCmplxPad, h_tmp2, 0, NULL, NULL);
}
void transposeT2(cl_mem cl_input)
{
	cufftComplex* h_tmp = (cufftComplex*)_aligned_malloc(giTplMemSzCmplxPad, AOCL_ALIGNMENT);
	cufftComplex* h_tmp2 = (cufftComplex*)_aligned_malloc(giTplMemSzCmplxPad, AOCL_ALIGNMENT);
	clEnqueueReadBuffer(queue, cl_input, CL_TRUE, 0, giTplMemSzCmplxPad, h_tmp, 0, NULL, NULL);

	for (int row = 0; row < 64; row++)
		for (int col = 0; col<33; col++)
		{
			h_tmp2[row + col * 64] = h_tmp[col + row * 33];
		}
	for (int i = 64 * 33; i < 64 * 64; i++)
	{
		h_tmp2[i].x = 0;
		h_tmp2[i].y = 0;
	}
	clEnqueueWriteBuffer(queue, cl_input, CL_TRUE, 0, giTplMemSzCmplxPad, h_tmp2, 0, NULL, NULL);
}


void fftScn(bool inverse, cl_mem* idata, cl_mem* odata)
{
	// Can't pass bool to device, so convert it to int
	int inverse_int = inverse;
	// Can't pass bool to device, so convert it to int
	int mangle_int = 0;

	printf("Kernel initialization is complete.\n");

	// Get the iterationstamp to evaluate performance
	double time = getCurrentTimestamp();

	// Loop twice over the kernels

	// Set the kernel arguments
	// Loop twice over the kernels
	for (int i = 0; i < 2; i++) {
		// Set the kernel arguments
		status = clSetKernelArg(fetch_kernel0, 0, sizeof(cl_mem), i == 0 ? (void *)idata : (void *)&d_tmp0);
		checkError(status, "Failed to set kernel arg 0");
		status = clSetKernelArg(fetch_kernel0, 1, sizeof(cl_mem), (void*)&d_tmp02);
		checkError(status, "Failed to set kernel arg 1");
		status = clSetKernelArg(fetch_kernel0, 2, sizeof(cl_int), (void*)&mangle_int);
		checkError(status, "Failed to set kernel arg 2");
		status = clSetKernelArg(fetch_kernel0, 3, sizeof(cl_int), (void*)&inverse_int);
		checkError(status, "Failed to set kernel arg 3");
		status = clSetKernelArg(fetch_kernel0, 4, sizeof(cl_int), (void*)&i);
		checkError(status, "Failed to set kernel arg 4");
		size_t lws_fetch[] = { NS };
		if (inverse_int^i == 0) {
			size_t gws_fetch[] = { NS * NS / 8 };;
			status = clEnqueueNDRangeKernel(queue, fetch_kernel0, 1, 0, gws_fetch, lws_fetch, 0, NULL, NULL);
			checkError(status, "Failed to launch kernel");
		}
		else {
			size_t gws_fetch[] = { NS * (NS + 8) / 8 };;
			status = clEnqueueNDRangeKernel(queue, fetch_kernel0, 1, 0, gws_fetch, lws_fetch, 0, NULL, NULL);
			checkError(status, "Failed to launch kernel");
		}



		// Launch the fft kernel - we launch a single work item hence enqueue a task
		status = clSetKernelArg(fft_kernel0, 0, sizeof(cl_int), (void*)&inverse_int);
		checkError(status, "Failed to set kernel arg 0");
		int ii = inverse_int^i;
		status = clSetKernelArg(fft_kernel0, 1, sizeof(cl_int), (void*)&ii);
		checkError(status, "Failed to set kernel arg 1");
		status = clEnqueueTask(queue2, fft_kernel0, 0, NULL, NULL);
		checkError(status, "Failed to launch kernel");
		// Set the kernel arguments
		status = clSetKernelArg(transpose_kernel0, 0, sizeof(cl_mem), i == 0 ? (void *)&d_tmp0 : (void *)odata);
		checkError(status, "Failed to set kernel arg 0");
		status = clSetKernelArg(transpose_kernel0, 1, sizeof(cl_int), (void*)&mangle_int);
		checkError(status, "Failed to set kernel arg 1");
		status = clSetKernelArg(transpose_kernel0, 2, sizeof(cl_int), (void*)&inverse_int);
		checkError(status, "Failed to set kernel arg 2");
		status = clSetKernelArg(transpose_kernel0, 3, sizeof(cl_int), (void*)&i);
		checkError(status, "Failed to set kernel arg 3");
		size_t lws_transpose[] = { NS };
		if (inverse_int^i == 0) {
			size_t gws_transpose[] = { NS * NS / 8 };;
			status = clEnqueueNDRangeKernel(queue3, transpose_kernel0, 1, 0, gws_transpose, lws_transpose, 0, NULL, NULL);
			checkError(status, "Failed to launch kernel");
		}
		else {
			size_t gws_transpose[] = { NS * (NS + 8) / 8 };;
			status = clEnqueueNDRangeKernel(queue3, transpose_kernel0, 1, 0, gws_transpose, lws_transpose, 0, NULL, NULL);
			checkError(status, "Failed to launch kernel");
		}
		if (!i) {
			status = clSetKernelArg(extraCol_kernel0, 0, sizeof(cl_mem), (void *)&d_tmp02);
			checkError(status, "Failed to set kernel arg 0");
			status = clSetKernelArg(extraCol_kernel0, 1, sizeof(cl_int), (void *)&inverse_int);
			checkError(status, "Failed to set kernel arg 1");
			size_t lws_extraCol[] = { 8 };
			size_t gws_extraCol[] = { NS };;
			status = clEnqueueNDRangeKernel(queue4, extraCol_kernel0, 1, 0, gws_extraCol, lws_extraCol, 0, NULL, NULL);
			checkError(status, "Failed to launch kernel");
		}
		// Wait for all command queues to complete pending events
		status = clFinish(queue);
		checkError(status, "failed to finish");
		status = clFinish(queue2);
		checkError(status, "failed to finish");
		status = clFinish(queue3);
		checkError(status, "failed to finish");
		status = clFinish(queue4);
		checkError(status, "failed to finish");
		if (!inverse_int && !i) {
			clEnqueueCopyBuffer(queue, d_tmp02, d_tmp0, 0, NS*NS * sizeof(float2), NS * sizeof(float2), 0, NULL, NULL);
			// clEnqueueReadBuffer(queue, d_tmp, CL_TRUE, 0, sizeof(float2) * N *( N+1), h_outData, 0, NULL, NULL);
			//for (int i = 0; i < N+1; i++) {
			/* for (int j = 0; j < N; j++)
			printf("%d= (%.1f,%.1f) ", j + i*N, h_outData[j + i*N].x, h_outData[j + i*N].y);
			printf("\n");*/

			//}
		}
	}
	// Record execution time
	time = getCurrentTimestamp() - time;

	printf("\tProcessing time = %.4fms\n", (float)(time * 1E3));
	double gpoints_per_sec = ((double)NS * NS / time) * 1E-9;
	double gflops = 2 * 5 * NS * NS  * (log((float)NS) / log((float)2)) / (time * 1E9);
	printf("\tThroughput = %.4f Gpoints / sec (%.4f Gflops)\n", gpoints_per_sec, gflops);
}
void fftTpl(bool inverse, cl_mem* idata, cl_mem* odata)
{
	// Can't pass bool to device, so convert it to int
	int inverse_int = inverse;
	// Can't pass bool to device, so convert it to int
	int mangle_int = 0;

	printf("Kernel initialization is complete.\n");

	// Get the iterationstamp to evaluate performance
	double time = getCurrentTimestamp();

	// Loop twice over the kernels

	// Set the kernel arguments
	for (int i = 0; i < 2; i++) {

		// Set the kernel arguments
		status = clSetKernelArg(fetch_kernel1, 0, sizeof(cl_mem), i == 0 ? (void *)idata : (void *)&d_tmp1);
		checkError(status, "Failed to set kernel arg 0");
		status = clSetKernelArg(fetch_kernel1, 1, sizeof(cl_mem), (void*)&d_tmp12);
		checkError(status, "Failed to set kernel arg 1");
		status = clSetKernelArg(fetch_kernel1, 2, sizeof(cl_int), (void*)&mangle_int);
		checkError(status, "Failed to set kernel arg 2");
		status = clSetKernelArg(fetch_kernel1, 3, sizeof(cl_int), (void*)&inverse_int);
		checkError(status, "Failed to set kernel arg 3");
		status = clSetKernelArg(fetch_kernel1, 4, sizeof(cl_int), (void*)&i);
		checkError(status, "Failed to set kernel arg 4");
		size_t lws_fetch[] = { NT };
		if (inverse_int^i == 0) {
			size_t gws_fetch[] = { NT * NT / 4 };;
			status = clEnqueueNDRangeKernel(queue, fetch_kernel1, 1, 0, gws_fetch, lws_fetch, 0, NULL, NULL);
			checkError(status, "Failed to launch kernel");
		}
		else {
			size_t gws_fetch[] = { NT * (NT + 4) / 4 };;
			status = clEnqueueNDRangeKernel(queue, fetch_kernel1, 1, 0, gws_fetch, lws_fetch, 0, NULL, NULL);
			checkError(status, "Failed to launch kernel");
		}




		// Launch the fft kernel - we launch a single work item hence enqueue a task
		status = clSetKernelArg(fft_kernel1, 0, sizeof(cl_int), (void*)&inverse_int);
		checkError(status, "Failed to set kernel arg 0");
		int ii = inverse_int^i;
		status = clSetKernelArg(fft_kernel1, 1, sizeof(cl_int), (void*)&ii);
		checkError(status, "Failed to set kernel arg 1");
		status = clEnqueueTask(queue2, fft_kernel1, 0, NULL, NULL);
		checkError(status, "Failed to launch kernel");
		// Set the kernel arguments
		status = clSetKernelArg(transpose_kernel1, 0, sizeof(cl_mem), i == 0 ? (void *)&d_tmp1 : (void *)odata);
		checkError(status, "Failed to set kernel arg 0");
		status = clSetKernelArg(transpose_kernel1, 1, sizeof(cl_int), (void*)&mangle_int);
		checkError(status, "Failed to set kernel arg 1");
		status = clSetKernelArg(transpose_kernel1, 2, sizeof(cl_int), (void*)&inverse_int);
		checkError(status, "Failed to set kernel arg 2");
		status = clSetKernelArg(transpose_kernel1, 3, sizeof(cl_int), (void*)&i);
		checkError(status, "Failed to set kernel arg 3");
		size_t lws_transpose[] = { NT };
		if (inverse_int^i == 0) {
			size_t gws_transpose[] = { NT * NT / 4 };;
			status = clEnqueueNDRangeKernel(queue3, transpose_kernel1, 1, 0, gws_transpose, lws_transpose, 0, NULL, NULL);
			checkError(status, "Failed to launch kernel");
		}
		else {
			size_t gws_transpose[] = { NT * (NT + 4) / 4 };;
			status = clEnqueueNDRangeKernel(queue3, transpose_kernel1, 1, 0, gws_transpose, lws_transpose, 0, NULL, NULL);
			checkError(status, "Failed to launch kernel");
		}
		if (!i) {
			status = clSetKernelArg(extraCol_kernel1, 0, sizeof(cl_mem), (void *)&d_tmp12);
			checkError(status, "Failed to set kernel arg 0");
			status = clSetKernelArg(extraCol_kernel1, 1, sizeof(cl_int), (void *)&inverse_int);
			checkError(status, "Failed to set kernel arg 1");
			size_t lws_extraCol[] = { 8 };
			if (!inverse_int) {
				size_t gws_extraCol[] = { 2 * NT };;
				status = clEnqueueNDRangeKernel(queue4, extraCol_kernel1, 1, 0, gws_extraCol, lws_extraCol, 0, NULL, NULL);
			}
			else {
				size_t gws_extraCol[] = { NT };;
				status = clEnqueueNDRangeKernel(queue4, extraCol_kernel1, 1, 0, gws_extraCol, lws_extraCol, 0, NULL, NULL);
			}
			checkError(status, "Failed to launch kernel");
		}

		// Wait for all command queues to complete pending events
		status = clFinish(queue);
		checkError(status, "failed to finish");
		status = clFinish(queue2);
		checkError(status, "failed to finish");
		status = clFinish(queue3);
		checkError(status, "failed to finish");
		status = clFinish(queue4);
		checkError(status, "failed to finish");

		if (!i&&!inverse_int) {
			clEnqueueCopyBuffer(queue, d_tmp12, d_tmp1, 0, 2 * NT*NT * sizeof(float2), 2 * NT * sizeof(float2), 0, NULL, NULL);
			//clEnqueueReadBuffer(queue, d_tmp1, CL_TRUE, 0, sizeof(float2) * 2 * NT *(NT + 1), h_outData, 0, NULL, NULL);
		}
	}
#ifdef cltime
	time = getCurrentTimestamp() - time;
	printf("\tProcessing time = %.4fms\n", (float)(time * 1E3));
	double gpoints_per_sec = ((double)NT * NT / 2 / time) * 1E-9;
	double gflops = 2 * 5 * NT * NT / 2 * (log((float)NT) / log((float)2)) / (time * 1E9);
	printf("\tThroughput = %.4f Gpoints / sec (%.4f Gflops)\n", gpoints_per_sec, gflops);
#endif
}
void MaxIdx(cufftReal* d_afData, int iSz, int** d_piMaxIdx)
{
	int iGDx;
	if (iSz == giScnSz) {
		iGDx = giWholeMaxGDx; // = (640*480/512*16) = (307200/8192) = 38 - will need two passes 
		dim3 thread(BLOCKDIMX_MAX, 1, 1);
		dim3 grid(iGDx, 1);

		//calculate block maxs
		InitKerTim(5);
		max_k << < grid, thread >> >(d_afData, NULL, iSz, gd_afBlockMaxs, gd_aiBlockMaxIdxs);
		WrapKerTim("max_k", 5);
		//WrapKerTim("Max1stPass", iSz);
		CUT_CHECK_ERROR("Kernel execution failed");
	}
	else {
		iGDx = giPartMaxGDx;; // if TplSz = 60, (60*60/512*8)+1 = (3600/4096)+1 = 1 - will only need one pass 
							  //if TplSz is larger it is possible that we need two passes.
							  //gd_afBlockMaxs have enough storage for finding max in whole scene.
							  //so it is definitely enough for finding max in part scene

							  //max will do 2 passes. In the first pass there will be several blocks. 
							  //In the second	there will be only one block.

							  //now do the first pass: each thread will read EACHTHREADREADS pixels. 
							  //Each block reads BLOCKDIMX_MAX*EACHTHREADREADS = 512*16 = 8192 pixels

		dim3 thread(BLOCKDIMX_MAX, 1, 1);
		dim3 grid(iGDx, 1);
		max_k << < grid, thread >> >(d_afData, NULL, iSz, gd_afBlockMaxs, gd_aiBlockMaxIdxs);
		*d_piMaxIdx = gd_aiBlockMaxIdxs;
	}

	if (iGDx == 1)
	{

	}
	else
	{
		//now do the second pass: each thread will read EACHTHREADREADS blockmaxs. 
		//We have only one block and this block reads iGDx blockmaxs.
		//note that (iGDx/EACHTHREADREADS) <= BLOCKDIMX_MAX
		dim3 thread2(BLOCKDIMX_MAX, 1, 1);
		dim3 grid2(1, 1);

		// execute the kernel
		//calculate maxs of block maxs
		//InitKerTim(5);
		max_k << < grid2, thread2 >> >(gd_afBlockMaxs, gd_aiBlockMaxIdxs, iGDx, gd_pfMax, gd_piMaxIdx);
		//WrapKerTim("Max2ndPass", 5);
		*d_piMaxIdx = gd_piMaxIdx;
		CUT_CHECK_ERROR("Kernel execution failed");
	}
}

void clMaxIdx(cufftReal* d_afData, int iSz, int* piMaxIdx)
{
#ifdef FPGA
	////////////////////////////////////////////////////////////////////////////////////////////////////////

	cufftReal* h_afData;
	int* h_aiBlockMaxIdxs;
	h_afData = (cufftReal*)_aligned_malloc(giScnMemSzReal, AOCL_ALIGNMENT);
	h_aiBlockMaxIdxs = (int*)_aligned_malloc(sizeof(int)*giWholeMaxGDx, AOCL_ALIGNMENT);
	CUDA_SAFE_CALL(cudaMemcpy(h_afData, d_afData, giScnMemSzReal, cudaMemcpyDeviceToHost));// copy memory to host
	cl_mem cl_d_afData = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, giScnMemSzReal, h_afData, NULL);
	cl_mem cl_gd_afBlockMaxs = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cufftReal)*giWholeMaxGDx, NULL, NULL);
	cl_mem cl_gd_aiBlockMaxIdxs = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*giWholeMaxGDx, NULL, NULL);

	// Set the kernel arguments 
	g_time = getCurrentTimestamp();
	status = clSetKernelArg(max_k_kernel, 0, sizeof(cl_mem), (void*)&cl_d_afData);
	checkError(status, "Failed to set kernel arg 0");
	status = clSetKernelArg(max_k_kernel, 1, sizeof(cl_mem), NULL);
	checkError(status, "Failed to set kernel arg 1");
	status = clSetKernelArg(max_k_kernel, 2, sizeof(int), (void*)&iSz);
	checkError(status, "Failed to set kernel arg 2");
	status = clSetKernelArg(max_k_kernel, 3, sizeof(cl_mem), (void*)&cl_gd_afBlockMaxs);
	checkError(status, "Failed to set kernel arg 3");
	status = clSetKernelArg(max_k_kernel, 4, sizeof(cl_mem), (void*)&cl_gd_aiBlockMaxIdxs);
	checkError(status, "Failed to set kernel arg 4");



	// Configure work set over which the kernel will execute
	//size_t wgSize[3] = { 1, 1, 1 };
	//size_t gSize[3] = { 1, 1, 1 };
	//printf("iGDx=%d\n", iGDx);
	// Launch the kernel
	status = clEnqueueNDRangeKernel(queue, max_k_kernel, 1, NULL, gSize, wgSize, 0, NULL, NULL);

	checkError(status, "Failed to launch kernel");
#ifdef cltime
	status = clFinish(queue);
	g_time = getCurrentTimestamp() - g_time;
	printf("\tmax_k_kernel Processing time = %.4fms\n", (float)(g_time * 1E3));
#endif

	//clReleaseEvent(*write_event);

	//Read back data 

	status = clEnqueueReadBuffer(queue, cl_gd_aiBlockMaxIdxs, CL_TRUE, 0, sizeof(int)*giWholeMaxGDx, h_aiBlockMaxIdxs, 0, NULL, NULL);
	checkError(status, "Failed to read buffer cl_gd_aiBlockMaxIdxs");

	//Free CL buffer

	status = clReleaseMemObject(cl_d_afData);
	checkError(status, "Failed to release buffer");
	status = clReleaseMemObject(cl_gd_afBlockMaxs);
	checkError(status, "Failed to release buffer");
	status = clReleaseMemObject(cl_gd_aiBlockMaxIdxs);
	checkError(status, "Failed to release buffer");
	// Wait for command queue to complete pending events
	status = clFinish(queue);
	checkError(status, "Failed to finish");

	*piMaxIdx = *h_aiBlockMaxIdxs;
	// Free the resources allocated
	//AOCLcleanup();
	_aligned_free(h_afData);
	_aligned_free(h_aiBlockMaxIdxs);
	////////////////////////////////////////////////////////////////////////////////////////////////////////
#endif
}

void clPointWiseMult(float2* gd_afMul, cl_mem* cl_gd_Tpl, cl_mem* cl_gd_afPadScnOut, int offset, int tpl, int iSz)
{


	int sz = tpl ? giTplMemSzCmplx : giScnMemSzCmplxPad;
	int dataN = iSz;
	cl_int status;
	cufftComplex* h_afMul, *h_afScnOut, *h_afTplOut;
	h_afMul = (cufftComplex *)_aligned_malloc(sz, AOCL_ALIGNMENT);
	//h_afScnOut = (cufftComplex *)_aligned_malloc(sz, AOCL_ALIGNMENT);
	//h_afTplOut = (cufftComplex *)_aligned_malloc(sz, AOCL_ALIGNMENT);
	CUDA_SAFE_CALL(cudaMemcpy(h_afMul, gd_afMul, sz, cudaMemcpyDeviceToHost));// copy memory to host
																			  //CUDA_SAFE_CALL(cudaMemcpy(h_afScnOut, d_afScnOut, sz, cudaMemcpyDeviceToHost));
																			  //CUDA_SAFE_CALL(cudaMemcpy(h_afTplOut, d_afTplOut, sz, cudaMemcpyDeviceToHost));
	cl_mem cl_gd_afMul = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, sz, h_afMul, NULL);
	//cl_mem cl_d_afScnOut = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, sz , h_afScnOut , NULL);
	cl_mem cl_d_afTplOut = clCreateBuffer(context, CL_MEM_READ_WRITE, sz, NULL, NULL);
	clEnqueueCopyBuffer(queue, *cl_gd_Tpl, cl_d_afTplOut, offset, 0, sz, 0, NULL, NULL);
	//cl_mem cl_gd_afMul2 = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, sz / 2, h_afMul + dataN, NULL);
	//cl_mem cl_d_afScnOut2 = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, sz / 2, h_afScnOut + dataN, NULL);
	//cl_mem cl_d_afTplOut2 = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, sz / 2, h_afTplOut + dataN, NULL);


	// Set the kernel arguments 
	status = clSetKernelArg(pointWiseMul_kernel, 0, sizeof(cl_mem), (void*)&cl_gd_afMul);
	checkError(status, "Failed to set kernel arg 0");
	status = clSetKernelArg(pointWiseMul_kernel, 1, sizeof(cl_mem), (void*)cl_gd_afPadScnOut);
	checkError(status, "Failed to set kernel arg 1");
	status = clSetKernelArg(pointWiseMul_kernel, 2, sizeof(cl_mem), (void*)&cl_d_afTplOut);
	checkError(status, "Failed to set kernel arg 2");
	status = clSetKernelArg(pointWiseMul_kernel, 3, sizeof(int), (void*)&dataN);
	checkError(status, "Failed to set kernel arg 3");
	float arg4 = (1.0f / (float)iSz);
	status = clSetKernelArg(pointWiseMul_kernel, 4, sizeof(float), (void*)&arg4);
	checkError(status, "Failed to set kernel arg 4");
	// Set the kernel arguments 
	//status = clSetKernelArg(pointWiseMul_kernel2, 0, sizeof(cl_mem), (void*)&cl_gd_afMul2);
	//checkError(status, "Failed to set kernel arg 0");
	//status = clSetKernelArg(pointWiseMul_kernel2, 1, sizeof(cl_mem), (void*)&cl_d_afScnOut2);
	//checkError(status, "Failed to set kernel arg 1");
	//status = clSetKernelArg(pointWiseMul_kernel2, 2, sizeof(cl_mem), (void*)&cl_d_afTplOut2);
	//checkError(status, "Failed to set kernel arg 2");
	//status = clSetKernelArg(pointWiseMul_kernel2, 3, sizeof(int), (void*)&dataN);
	//checkError(status, "Failed to set kernel arg 3");
	//status = clSetKernelArg(pointWiseMul_kernel2, 4, sizeof(float), (void*)&arg4);
	//checkError(status, "Failed to set kernel arg 4");

	// Configure work set over which the kernel will execute
	size_t wgSize[3] = { 1, 1, 1 };
	size_t gSize[3] = { 4, 1, 1 };
	g_time = getCurrentTimestamp();
	// Launch the kernel
	status = clEnqueueNDRangeKernel(queue, pointWiseMul_kernel, 1, 0, gSize, wgSize, 0, NULL, NULL);
	checkError(status, "Failed to launch kernel");
	//status = clEnqueueNDRangeKernel(queue2, pointWiseMul_kernel2, 1, NULL, gSize, wgSize, 0, NULL, NULL);
	//checkError(status, "Failed to launch kernel");
#ifdef cltime
	status = clFinish(queue);
	g_time = getCurrentTimestamp() - g_time;
	printf("\tpointWiseMul_kernel processing time = %.4fms\n", (float)(g_time * 1E3));
#endif
	//status = clFinish(queue2);
	//time = getCurrentTimestamp() - time;
	//printf("\tProcessing time = %.4fms\n", (float)(time * 1E3));

	//Read back data 
	status = clEnqueueReadBuffer(queue, cl_gd_afMul, CL_TRUE, 0, sz, h_afMul, 0, NULL, NULL);
	checkError(status, "Failed to read buffer cl_gd_afMul");
	//status = clEnqueueReadBuffer(queue2, cl_gd_afMul2, CL_TRUE, 0, sz / 2, h_afMul + dataN, 0, NULL, NULL);
	checkError(status, "Failed to read buffer cl_gd_afMul");

	//Free CL buffer
	//status = clReleaseMemObject(cl_gd_afMul);
	//checkError(status, "Failed to release buffer");
	//status = clReleaseMemObject(cl_d_afScnOut);
	//checkError(status, "Failed to release buffer");
	status = clReleaseMemObject(cl_d_afTplOut);
	checkError(status, "Failed to release buffer");
	//status = clReleaseMemObject(cl_gd_afMul2);
	//checkError(status, "Failed to release buffer");
	//status = clReleaseMemObject(cl_d_afScnOut2);
	//checkError(status, "Failed to release buffer");
	//status = clReleaseMemObject(cl_d_afTplOut2);
	//checkError(status, "Failed to release buffer");
	// Wait for command queue to complete pending events
	status = clFinish(queue);
	//status = clFinish(queue2);
	checkError(status, "Failed to finish");

	// Free the resources allocated
	//AOCLcleanup();
	CUDA_SAFE_CALL(cudaMemcpy(gd_afMul, h_afMul, sz, cudaMemcpyHostToDevice));
	//CUDA_SAFE_CALL(cudaMemcpy(gd_afMul+dataN, h_afMul + dataN, sz / 2, cudaMemcpyHostToDevice));
	_aligned_free(h_afMul);
	//_aligned_free(h_afScnOut);
	//_aligned_free(h_afTplOut);
}

void clKthLaw(float2* gd_afPadScnOutPad, int giScnSzPad)
{
	cufftComplex* h_afPadScnOut;
	h_afPadScnOut = (cufftComplex *)_aligned_malloc(giScnMemSzCmplxPad, AOCL_ALIGNMENT);

	CUDA_SAFE_CALL(cudaMemcpy(h_afPadScnOut, gd_afPadScnOutPad, giScnMemSzCmplxPad, cudaMemcpyDeviceToHost));// copy memory to host

																											 //cl_mem cl_d_afPadScnOut = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, giScnMemSzCmplx, h_afPadScnOut, NULL);
																											 //cl_gd_afPadScnOut = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, giScnMemSzCmplx, h_afPadScnOut, NULL);
	clEnqueueWriteBuffer(queue, cl_gd_afPadScnOut, CL_FALSE, 0, giScnMemSzCmplxPad, h_afPadScnOut, 0, NULL, NULL);
	//cl_event* write_event = (cl_event *)malloc(sizeof(cl_event));
	//status = clEnqueueWriteBuffer(queue, cl_d_afPadScnOut, CL_TRUE, 0, giScnMemSzCmplx, h_afPadScnOut, 0, NULL, NULL);// write into CL buffer
	//checkError(status, "Failed to write buffer cl_gd_afPadScnOut");

	// Set the kernel arguments 
	status = clSetKernelArg(kthLaw_kernel, 0, sizeof(cl_mem), (void*)&cl_gd_afPadScnOut);
	checkError(status, "Failed to set kernel arg 0");
	status = clSetKernelArg(kthLaw_kernel, 1, sizeof(cl_int), (void*)&giScnSzPad);
	checkError(status, "Failed to set kernel arg 1");


	// Configure work set over which the kernel will execute
	//size_t wgSize[3] = { 1, 1, 1 };
	//size_t gSize[3] = { 1, 1, 1 };
	g_time = getCurrentTimestamp();

	// Launch the kernel
	status = clEnqueueNDRangeKernel(queue, kthLaw_kernel, 1, NULL, gSize, wgSize, 0, NULL, NULL);
	checkError(status, "Failed to launch kernel");

#ifdef cltime
	status = clFinish(queue);
	g_time = getCurrentTimestamp() - g_time;
	printf("\tkthLaw_kernel processing time = %.4fms\n", (float)(g_time * 1E3));
#endif

	//clReleaseEvent(*write_event);

	//Read back data 
	status = clEnqueueReadBuffer(queue, cl_gd_afPadScnOut, CL_TRUE, 0, giScnMemSzCmplxPad, h_afPadScnOut, 0, NULL, NULL);
	checkError(status, "Failed to read buffer cl_gd_afPadScnOut");

	//Free CL buffer
	//status = clReleaseMemObject(cl_d_afPadScnOut);
	//checkError(status, "Failed to release buffer");
	// Wait for command queue to complete pending events

	// Free the resources allocated
	//AOCLcleanup();
	CUDA_SAFE_CALL(cudaMemcpy(gd_afPadScnOutPad, h_afPadScnOut, giScnMemSzCmplxPad, cudaMemcpyHostToDevice));
	_aligned_free(h_afPadScnOut);

}


//compute PSR value
float getPSR(cufftReal* gd_afCorr, cufftReal* gh_afArea, int* iPeakIndx, int iSz, int iW, int iH)
{

	int iI;
	int *d_piMaxIdx = NULL;
#ifdef FPGA
	clMaxIdx(gd_afCorr, iSz, iPeakIndx);
#else
	MaxIdx(gd_afCorr, iSz, &d_piMaxIdx);
#endif
										//InitKerTim(iSz);
										//CUDA_SAFE_CALL(cudaMemcpy(iPeakIndx, (int*)d_piMaxIdx, sizeof(int), cudaMemcpyDeviceToHost));
										//WrapKerTim("MemcpyD2HPeak", iSz);
										//find PSR on the cpu, because we are dealing with at most giAreaH x giAreaH elements
	int iMaxRow, iMaxCol;
	Indx2Coord(iW, *iPeakIndx, &iMaxRow, &iMaxCol);
	//The int4 type is a CUDA built-in type with four fields: x(RowBeg),y(RowEnd),z(ColBeg),w(ColEnd)
	int4 aiAreaCoord = getSurrCoord(iMaxRow, iMaxCol, giAreaH, iW, iH);
	int iStart = (aiAreaCoord.x*iW) + aiAreaCoord.z;
	//area is not always giAreaH x giAreaH, it might be cut if the peak is close to boundary
	int iNewAreaH = aiAreaCoord.y - aiAreaCoord.x + 1;
	int iNewAreaW = aiAreaCoord.w - aiAreaCoord.z + 1;
	int iNewAreaSz = iNewAreaW*iNewAreaH;
	//transfer the area
	//InitKerTim(iSz);
	CUDA_SAFE_CALL(cudaMemcpy2D(gh_afArea, iNewAreaW * sizeof(cufftReal), gd_afCorr + iStart, iW * sizeof(cufftReal), iNewAreaW * sizeof(cufftReal), iNewAreaH, cudaMemcpyDeviceToHost));
	//WrapKerTim("MemcpyD2HArea", iSz);
	//find the new index of the max value in the area cut from corr plane
	float fMax = gh_afArea[0];
	int iNewMaxIndx = 0;
	for (iI = 0; iI<iNewAreaSz; iI++)
	{
		if (gh_afArea[iI] > fMax)
		{
			fMax = gh_afArea[iI];
			iNewMaxIndx = iI;
		}
	}
	int iNewMaxRow, iNewMaxCol;
	Indx2Coord(iNewAreaW, iNewMaxIndx, &iNewMaxRow, &iNewMaxCol);
	int4 aiMaskCoord = getSurrCoord(iNewMaxRow, iNewMaxCol, giMaskH, iNewAreaW, iNewAreaH);
	//mask is not always giMaskH x giMaskH, it might be cut if the peak is close to boundary
	int iNewMaskH = aiMaskCoord.y - aiMaskCoord.x + 1;
	int iNewMaskW = aiMaskCoord.w - aiMaskCoord.z + 1;
	//assign mask values to zero
	assignVal(iNewAreaW, gh_afArea, aiMaskCoord, 0);
	//calculate mean by not counting the mask
	int iFrameNumElem = (iNewAreaH*iNewAreaW) - (iNewMaskH*iNewMaskW);
	float fMean = sum(gh_afArea, iNewAreaSz) / iFrameNumElem;
	//mask values = mean
	assignVal(iNewAreaW, gh_afArea, aiMaskCoord, fMean);
	//calculate standard deviation by not counting the mask
	//calculate sum of sqr_dif
	float fTotal = 0;
	float fVal;
	for (iI = 0; iI < iNewAreaSz; iI++)
	{
		fVal = gh_afArea[iI] - fMean;
		fTotal += fVal*fVal;
	}
	float afStdVar = sqrt(fTotal / (iFrameNumElem - 1));
	float fMeasure;
	if (afStdVar != 0)
		fMeasure = (fMax - fMean) / afStdVar;
	else
		//if we are out of bound while copying part scene, this might happen since part scene will have lots of zeros
		fMeasure = 0;
	return fMeasure;
}



void Corr(cl_mem* cl_gd_Tpl, int offset, dim3 dBlocks, dim3 dThreads, cl_mem* cl_gd_afPadScnOut/*cufftComplex* d_afScnOut*/, int iSz, cufftComplex* gd_afMul, cufftHandle hFFTplanInv, cufftReal* gd_afCorr, cufftReal* gh_afArea, int* piPeakIndx, float* pfPSR, int iW, int iH, bool tpl)
{
	//take conjugate of template fft and point wise multiply with scene and scale it with image size
#ifdef FPGA
	clPointWiseMult(gd_afMul, cl_gd_Tpl, cl_gd_afPadScnOut, offset, tpl, iSz);
#else
	InitKerTim(5);
	pointWiseMul << <dBlocks, dThreads >> >(gd_afMul, d_afScnOut, d_afTplOut, iSz, 1.0f / (float)iSz);
	WrapKerTim("Mul", 5);
#endif
	CUT_CHECK_ERROR("pointWiseMul() execution failed\n");
	//take inverse FFT of multiplication
	//InitKerTim(iSz);
	if (tpl) {
	//CUFFT_SAFE_CALL(cufftExecC2R(hFFTplanInv, (cufftComplex *)gd_afMul, (cufftReal *)gd_afCorr));
	cl_mem cl_afCorr = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, giTplMemSzCmplxPad + 2, NULL, NULL);
	transposeT2(cl_gd_afMul);
	fftTpl(true, &cl_gd_afMul, &cl_afCorr);
	cufftReal* h_gd_afMul = (cufftReal*)_aligned_malloc(giTplMemSzRealPad, AOCL_ALIGNMENT);
	clEnqueueReadBuffer(queue4, cl_afCorr, CL_TRUE, 0, giTplMemSzRealPad, h_gd_afMul, 0, NULL, NULL);
	cudaMemcpy(gd_afCorr, h_gd_afMul, giTplMemSzRealPad, cudaMemcpyHostToDevice);
	_aligned_free(h_gd_afMul);
	clReleaseMemObject(cl_afCorr);
	}
	else {
	cl_mem cl_afCorr = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, giScnMemSzRealPad +2, NULL, NULL);
	transpose2(cl_gd_afMul);
	fftScn(true, &cl_gd_afMul, &cl_afCorr);
	cufftReal* h_gd_afMul = (cufftReal*)_aligned_malloc(giScnMemSzRealPad, AOCL_ALIGNMENT);
	clEnqueueReadBuffer(queue4, cl_afCorr, CL_TRUE, 0, giScnMemSzRealPad, h_gd_afMul, 0, NULL, NULL);
	cudaMemcpy(gd_afCorr, h_gd_afMul, giScnMemSzRealPad, cudaMemcpyHostToDevice);
	_aligned_free(h_gd_afMul);
	clReleaseMemObject(cl_afCorr);
	}
	status = clReleaseMemObject(cl_gd_afMul);
	checkError(status, "Failed to release buffer");
	//WrapKerTim("FFTinv", iSz);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	//find the PSR
	*pfPSR = getPSR(gd_afCorr, gh_afArea, piPeakIndx, iSz, iW, iH);
	//printf("PSR=%f, piPeakIndx=%d\n", *pfPSR, *piPeakIndx);
	return;
}

//cuda
void Corr2(cufftComplex* d_afTplOut, dim3 dBlocks, dim3 dThreads, cufftComplex* d_afScnOut, int iSz, cufftComplex* gd_afMul, cufftHandle hFFTplanInv, cufftReal* gd_afCorr, cufftReal* gh_afArea, int* piPeakIndx, float* pfPSR, int iW, int iH, bool szi)
{
	pointWiseMul << <dBlocks, dThreads >> >(gd_afMul, d_afScnOut, d_afTplOut, iSz, 1.0f / (float)iSz);

	CUT_CHECK_ERROR("pointWiseMul() execution failed\n");
	//take inverse FFT of multiplication
	//InitKerTim(iSz);
	CUFFT_SAFE_CALL(cufftExecC2R(hFFTplanInv, (cufftComplex *)gd_afMul, (cufftReal *)gd_afCorr));
	//WrapKerTim("FFTinv", iSz);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	//find the PSR
	*pfPSR = getPSR(gd_afCorr, gh_afArea, piPeakIndx, iSz, iW, iH);
	return;
}
inline void InitTim()
{
#ifdef PARTIM
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	CUT_SAFE_CALL(sdkResetTimer(&guiParTim));
	CUT_SAFE_CALL(sdkStartTimer(&guiParTim));
#endif
}

inline void WrapTim(char* sParName)
{
#ifdef PARTIM
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	CUT_SAFE_CALL(sdkStopTimer(&guiParTim));
	double dTime = sdkGetTimerValue(&guiParTim);
	printf("%s time: %f msecs.\n", sParName, dTime);
	g_dRunsOnGPUTotalTime += dTime;
#endif
}

void PrepTplFFT(cl_mem cl_gd_afCompFlt, cufftReal* gd_afCompFlt, cufftReal** d_pafPadTplIn, cufftComplex** d_pafPadTplOut, cufftComplex** d_pafWholeTplFFT, cufftComplex** d_pafPartTplFFT, cufftHandle ghFFTplanWholeFwd, cufftHandle ghFFTplanPartFwdC)
{

#ifdef SAVEFFT
	int iSzIndx, iIPIndx, iFltIndx, iFltAbsIndx, iIPAbsIndx;
	cufftReal
		//	*d_afTpl,
		*d_afPadTplIn,
		*gpu_d_afTpl;
	cl_mem cl_d_afTpl = clCreateBuffer(context, CL_MEM_READ_WRITE, giTplMemSzReal, NULL, NULL);
	cl_mem cl_d_afPadTpl = clCreateBuffer(context, CL_MEM_READ_WRITE, giScnMemSzRealPad, NULL, NULL);
	cl_mem cl_afWholeTplFFT = clCreateBuffer(context, CL_MEM_READ_WRITE, giScnMemSzCmplx, NULL, NULL);
	//first allocate mem
	//WholeTpls are the MulCompFlts (last flts in the compflt list). They are used in 1st pass. Their size is as big as scn
	//PartTpls are all other comp flt excluding MulCompFlts. They are used in 2nd pass. Their size is as big as tpl (is not blowed up to scn size)
	int iWholeMemSz = giPadScnH * giPadScnW * giNumIPInFirst * giNumSz * gstCompFlt.iNumMulCompFlt * sizeof(cufftComplex);
	CUDA_SAFE_CALL(cudaMalloc((void **)&*d_pafWholeTplFFT, iWholeMemSz));
	cl_gd_afWholeTplFFT = clCreateBuffer(context, CL_MEM_READ_WRITE, iWholeMemSz, NULL, NULL);
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_afPadTplIn, giScnMemSzRealPad));
	CUDA_SAFE_CALL(cudaMalloc((void **)&gpu_d_afTpl, giTplMemSzReal));
	int iPartMemSz = giTplH * giTplW * giNumIPRot * giNumSz * giNumSngCompFlt * sizeof(cufftComplex);
	CUDA_SAFE_CALL(cudaMalloc((void **)&*d_pafPartTplFFT, iPartMemSz));
	cl_gd_afPartTplFFT = clCreateBuffer(context, CL_MEM_READ_WRITE, iPartMemSz, NULL, NULL);
	//take FFT of WholeTpls
	for (iFltIndx = giNumSngCompFlt; iFltIndx < giNumOrigFlt; iFltIndx++)
	{
		for (iSzIndx = 0; iSzIndx < giNumSz; iSzIndx++)
		{
			for (iIPIndx = giBegIdxIPInFirst; iIPIndx < giEndIdxIPInFirst; iIPIndx++)
			{
				CUDA_SAFE_CALL(cudaMemset(d_afPadTplIn, 0, giScnMemSzRealPad));
				clEnqueueCopyBuffer(queue, cl_gd_afCompFlt, cl_d_afTpl, cl_gd_afCompFlt(iIPIndx, iSzIndx, iFltIndx), 0, giTplMemSzReal, 0, NULL, NULL);
				gpu_d_afTpl = gd_afCompFlt(iIPIndx, iSzIndx, iFltIndx);
				cufftReal* h_afTpl = (cufftReal*)_aligned_malloc(giTplMemSzReal, AOCL_ALIGNMENT);
				cufftReal* h_afPadTplIn = (cufftReal*)_aligned_malloc(giScnMemSzRealPad, AOCL_ALIGNMENT);
				cufftComplex*  h_afWholeTplFFT = (cufftComplex*)_aligned_malloc(giScnMemSzCmplxPad, AOCL_ALIGNMENT);
				//////cl_mem cl_gd_afWholeTplFFTtmp = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, giScnMemSzCmplx, NULL, NULL);
				clEnqueueReadBuffer(queue, cl_d_afTpl, CL_TRUE, 0, giTplMemSzReal, h_afTpl, 0, NULL, NULL);
				//cudaMemcpy(h_afTpl2, gpu_d_afTpl, giTplMemSzReal, cudaMemcpyDeviceToHost);
				//pad template
				CUDA_SAFE_CALL(cudaMemcpy2D(d_afPadTplIn, (giPadScnW * sizeof(cufftReal)), h_afTpl, giTplWMemSz, giTplWMemSz, giTplH, cudaMemcpyHostToDevice));
				cudaMemcpy(h_afPadTplIn, d_afPadTplIn, giScnMemSzRealPad, cudaMemcpyDeviceToHost);
				//CUDA_SAFE_CALL(cudaMemcpy2D(d_afPadTplIn, (giScnW * sizeof(cufftReal)), gpu_d_afTpl, giTplWMemSz, giTplWMemSz, giTplH, cudaMemcpyDeviceToDevice));
				//take the fft and save it to WholeTplFFT
				iFltAbsIndx = iFltIndx - giNumSngCompFlt;
				iIPAbsIndx = iIPIndx - giBegIdxIPInFirst;
				//printf("iIPIndx=%d iSzIndx=%d iFltIndx=%d d_afPadTplIn= %d\n", iIPIndx, iSzIndx, iFltIndx, d_afPadTplIn);
				CUFFT_SAFE_CALL(cufftExecR2C(ghFFTplanWholeFwd, (cufftReal *)d_afPadTplIn, (cufftComplex *)*d_pafWholeTplFFT(iIPAbsIndx, iSzIndx, iFltAbsIndx)));
				cudaMemcpy(h_afWholeTplFFT, *d_pafWholeTplFFT(iIPAbsIndx, iSzIndx, iFltAbsIndx), giScnMemSzCmplxPad, cudaMemcpyDeviceToHost);
				//clEnqueueWriteBuffer(queue, cl_gd_afWholeTplFFT, CL_TRUE, cl_gd_afWholeTplFFT(iIPAbsIndx, iSzIndx, iFltAbsIndx), giScnMemSzCmplxPad, h_afWholeTplFFT, 0, NULL, NULL);
				//fpga fft:
				//cufftReal* h_test = (cufftReal*)_aligned_malloc(giTplMemSzReal, AOCL_ALIGNMENT);
				//clEnqueueReadBuffer(queue, d_afTpl, CL_TRUE, 0, giTplMemSzReal, h_test, 0, NULL, NULL);
				//for (int i = 0; i<giTplW; i++)
				//	printf("test0[%d]=%f", i, h_afTpl[i]);

				/*float zero=0;
				clEnqueueFillBuffer(queue, cl_d_afPadTpl, &zero, sizeof(float), 0, giScnMemSzRealPad+8, 0, NULL, NULL);//why need giScnMemSzRealPad+1?
				const size_t sorigin[] = { 0, 0, 0 };
				const size_t dorigin[] = { 0, 0, 0 };
				const size_t regin[] = { giTplWMemSz, giTplH, 1 };
				clEnqueueCopyBufferRect(queue, d_afTpl, cl_d_afPadTpl, sorigin, dorigin, regin, 0, 0, (giPadScnW * sizeof(cufftReal)), giScnMemSzRealPad, 0, NULL, NULL);*/

				//Pad zero in host manually, clEnqueueFillBuffer has bug when buffer is large
				cufftReal* h_padTpl = (cufftReal*)_aligned_malloc(giScnMemSzRealPad, AOCL_ALIGNMENT);

				for (int col = 0; col<giPadScnW; col++)
					for (int row = 0; row < giPadScnH; row++)
					{
						if (col < giTplW && row < giTplH)
							h_padTpl[col + row*giPadScnW] = h_afTpl[col + row*giTplW];
						else
							h_padTpl[col + row*giPadScnW] = 0;

					}

				for (int i = 0; i<giScnMemSzRealPad / sizeof(float); i++)
					if (h_padTpl[i] != h_afPadTplIn[i])
						printf("dif[%d]=%f", i, h_padTpl[i]);

				clEnqueueWriteBuffer(queue, cl_d_afPadTpl, CL_TRUE, 0, giScnMemSzRealPad, h_padTpl, 0, NULL, NULL);
				cl_mem cl_afWholeTplFFT = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, giScnMemSzCmplxPad, NULL, NULL);
				fftScn(false, &cl_d_afPadTpl, &cl_afWholeTplFFT);

				transpose(cl_afWholeTplFFT);
				clEnqueueCopyBuffer(queue4, cl_afWholeTplFFT, cl_gd_afWholeTplFFT, cl_gd_afWholeTplFFT(iIPAbsIndx, iSzIndx, iFltAbsIndx), 0, giScnMemSzCmplx, 0, NULL, NULL);
				cufftComplex* h_TplFFTtest = (cufftComplex*)_aligned_malloc(giScnMemSzCmplxPad, AOCL_ALIGNMENT);
				clEnqueueReadBuffer(queue, cl_afWholeTplFFT, CL_TRUE, 0, giScnMemSzCmplxPad, h_TplFFTtest, 0, NULL, NULL);
				clFinish(queue);					
			
			}
		}
	}
	CUDA_SAFE_CALL(cudaFree(d_afPadTplIn));
	//take FFT of PartTpls
	for (iFltIndx = 0; iFltIndx < giNumSngCompFlt; iFltIndx++)
	{
		for (iSzIndx = 0; iSzIndx < giNumSz; iSzIndx++)
		{
			for (iIPIndx = 0; iIPIndx < giNumIPRot; iIPIndx++)
			{
				//gd_afCompFlt(iIPIndx, iSzIndx, iFltIndx, d_afTpl);
				gpu_d_afTpl = gd_afCompFlt(iIPIndx, iSzIndx, iFltIndx);
				clEnqueueCopyBuffer(queue, cl_gd_afCompFlt, cl_d_afTpl, cl_gd_afCompFlt(iIPIndx, iSzIndx, iFltIndx), 0, giTplMemSzReal, 0, NULL, NULL);

				//cufftReal* h_afTpl = (cufftReal*)_aligned_malloc(giTplMemSzReal, AOCL_ALIGNMENT);
				//status = clEnqueueReadBuffer(queue, d_afTpl, CL_TRUE, 0, giTplMemSzReal, h_afTpl, 0, NULL, NULL);
				cufftComplex*  h_afPartTplFFT = (cufftComplex*)_aligned_malloc(giTplMemSzCmplx, AOCL_ALIGNMENT);
				//CUDA_SAFE_CALL(cudaMemcpy(gpu_d_afTpl, h_afTpl, giTplMemSzReal, cudaMemcpyHostToDevice));
				//printf("iIPIndx=%d iSzIndx=%d iFltIndx=%d d_afTpl= %d\n", iIPIndx, iSzIndx, iFltIndx, d_afTpl);
				//take the fft and save it to PartTplFFT
				CUFFT_SAFE_CALL(cufftExecR2C(ghFFTplanPartFwd, (cufftReal *)gpu_d_afTpl, (cufftComplex *)*d_pafPartTplFFT(iIPIndx, iSzIndx, iFltIndx)));

				cl_mem cl_pafPartTplFFT = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, giTplMemSzCmplxPad, NULL, NULL);
				fftTpl(false, &cl_d_afTpl, &cl_pafPartTplFFT);
				transposeT(cl_pafPartTplFFT);
				clEnqueueCopyBuffer(queue4, cl_pafPartTplFFT, cl_gd_afPartTplFFT, cl_gd_afPartTplFFT(iIPIndx, iSzIndx, iFltIndx), 0, giTplMemSzCmplx, 0, NULL, NULL);
				
				//cudaMemcpy(h_afPartTplFFT, *d_pafPartTplFFT(iIPIndx, iSzIndx, iFltIndx), giTplMemSzCmplx, cudaMemcpyDeviceToHost);
				//clEnqueueWriteBuffer(queue, cl_gd_afPartTplFFT, CL_TRUE, cl_gd_afPartTplFFT(iIPIndx, iSzIndx, iFltIndx), giTplMemSzCmplx, h_afPartTplFFT, 0, NULL, NULL);
				
				
			}
		}
	}
#else
	//allocate gd_afPadTplIn and gd_afPadTplOut 
	CUDA_SAFE_CALL(cudaMalloc((void **)&*d_pafPadTplIn, giScnMemSzReal));
	CUDA_SAFE_CALL(cudaMemset(*d_pafPadTplIn, 0, giScnMemSzReal));
	CUDA_SAFE_CALL(cudaMalloc((void **)&*d_pafPadTplOut, giScnMemSzCmplx));
#endif
}

void DestroyTplFFT(cufftComplex* gd_afWholeTplFFT, cufftComplex* gd_afPartTplFFT, cufftReal* gd_afPadTplIn, cufftComplex* gd_afPadTplOut)
{
#ifdef SAVEFFT
	CUDA_SAFE_CALL(cudaFree(gd_afWholeTplFFT));
	CUDA_SAFE_CALL(cudaFree(gd_afPartTplFFT));
	clReleaseMemObject(cl_gd_afWholeTplFFT);
	clReleaseMemObject(cl_gd_afPartTplFFT);
#else
	CUDA_SAFE_CALL(cudaFree(gd_afPadTplIn));
	CUDA_SAFE_CALL(cudaFree(gd_afPadTplOut));
#endif
}

void getWholeTplFFT(cufftReal* gd_afCompFlt, int iIPIndx, int iSzIndx, int iFltIndx, cufftReal* gd_afPadTplIn, cufftComplex** d_pafPadTplOut, cufftHandle ghFFTplanWholeFwd)
{
#ifdef SAVEFFT
	int iFltAbsIndx = iFltIndx - giNumSngCompFlt;// from 0 to giNumOrigFlt-giNumSngCompFlt
	int iIPAbsIndx = iIPIndx - giBegIdxIPInFirst;
	*d_pafPadTplOut = gd_afWholeTplFFT(iIPAbsIndx, iSzIndx, iFltAbsIndx);
	//cufftComplex*  h_afWholeTplFFT = (cufftComplex*)_aligned_malloc(giScnMemSzCmplx, AOCL_ALIGNMENT);
	//clEnqueueReadBuffer(queue, cl_gd_afWholeTplFFT, CL_TRUE, cl_gd_afWholeTplFFT(iIPAbsIndx, iSzIndx, iFltAbsIndx), giScnMemSzCmplx, h_afWholeTplFFT, 0, NULL, NULL);
	//(*d_pafPadTplOut, h_afWholeTplFFT, giScnMemSzCmplx, cudaMemcpyHostToDevice);
#else
	//find the starting index of template
	cufftReal* d_afTpl = gd_afCompFlt(iIPIndx, iSzIndx, iFltIndx);
	//pad template
	CUDA_SAFE_CALL(cudaMemcpy2D(gd_afPadTplIn, (giScnW * sizeof(cufftReal)), d_afTpl, giTplWMemSz, giTplWMemSz, giTplH, cudaMemcpyDeviceToDevice));
	//take the FFT of the template
	CUFFT_SAFE_CALL(cufftExecR2C(ghFFTplanWholeFwd, (cufftReal *)gd_afPadTplIn, (cufftComplex *)*d_pafPadTplOut));
#endif
}

void getPartTplFFT(cufftReal* gd_afCompFlt, int iIPIndx, int iSzIndx, int iFltIndx, cufftComplex** d_pafPadTplOut, cufftHandle ghFFTplanPartFwd, cufftComplex* gd_afPartTplFFT)
{
#ifdef SAVEFFT
	*d_pafPadTplOut = gd_afPartTplFFT(iIPIndx, iSzIndx, iFltIndx);
#else
	//get the pointer to the tpl
	cufftReal* d_afTpl = gd_afCompFlt(iIPIndx, iSzIndx, iFltIndx);
	//take the FFT of the template
	CUFFT_SAFE_CALL(cufftExecR2C(ghFFTplanPartFwd, (cufftReal *)d_afTpl, (cufftComplex *)*d_pafPadTplOut));
#endif
}

//If the MaxPeakIndex is close to the boundry of the image, when we try to copy the part of the scene
//we can be out bound! check for this condition, and if so get part of the scene until boundry.
//since we initalize the part image to zero, it would have an effect such that part image is padded with zero
void getCopyWidthHeight(int iMaxPeakIndx, int* piPartW, int* piPartH)
{
	int iMaxPeakRow, iMaxPeakCol;
	//make sure we are not out of bounds
	Indx2Coord(giPadScnW, iMaxPeakIndx, &iMaxPeakRow, &iMaxPeakCol);
	*piPartW = giTplW;
	int iEndCol = iMaxPeakCol + *piPartW - 1;
	if (iEndCol >= giScnW && iMaxPeakCol <= giScnW)
		*piPartW = *piPartW - (iEndCol + 1 - giScnW);
	*piPartH = giTplH;
	int iEndRow = iMaxPeakRow + *piPartH - 1;
	if (iEndRow >= giScnH && iMaxPeakRow <= giScnH)
		*piPartH = *piPartH - (iEndRow + 1 - giScnH);
}

/*B = GRAYTO8(A) converts the double array A to unisgned char by scaling A by 255
* and then rounding.  NaN's in A are converted to 0.  Values in A greater
* than 1.0 are converted to 255; values less than 0.0 are converted to 0.
*/
void ConvertFromDouble(float *pr, unsigned char *qr, int numElements)
{
	int k;
	float val;

	for (k = 0; k < numElements; k++)
	{
		val = *pr++;
		if (val == NULL) {
			*qr++ = 0;
		}
		else {
			val = val * 255.0f + 0.5f;
			if (val > 255.0) val = 255.0;
			if (val < 0.0)   val = 0.0;
			*qr++ = (unsigned char)val;
		}
	}
}

//this function immitates Matlab imadjust function's LookUp Table creation.
void genLUT()
{
	float fN = LUTSIZE;
	float fD1 = 0;
	float fD2 = 1;
	for (int i = 0; i < fN - 1; i++)
	{
		gfLUT[i] = fD1 + i*((fD2 - fD1) / (fN - 1));
	}
	gfLUT[int(fN - 1)] = fD2;

	//make sure lut is in the range [gfLIn;gfHIn]
	for (int i = 0; i < fN; i++)
	{
		if (gfLUT[i] < gfLIn) gfLUT[i] = gfLIn;
		if (gfLUT[i] > gfHIn) gfLUT[i] = gfHIn;
	}

	//out = ( (img - lIn(d,:)) ./ (hIn(d,:) - lIn(d,:)) ) .^ (g(d,:));
	for (int i = 0; i < fN; i++)
	{
		gfLUT[i] = pow((gfLUT[i] - gfLIn) / (gfHIn - gfLIn), gfG);
	}
	//out(:) = out .* (hOut(d,:) - lOut(d,:)) + lOut(d,:);
	for (int i = 0; i < fN; i++)
	{
		gfLUT[i] = gfLUT[i] * (gfHOut - gfLOut) + gfLOut;
	}
	ConvertFromDouble(gfLUT, gacLUT, LUTSIZE);
}

void CpyScnToDevAndPreProcess(unsigned char* acScn, float* d_afPadScnIn, bool bConGam, bool bFixDead)
{
	//I can do the adjusting before fixing the dead pixel. Adjusted dead pixel will be overwritten as an overage of adjusted neighbors. Adjusting is done to each pixel independently.
	//copy scene to device
	InitTim();
	InitKerTim(1);
	CUDA_SAFE_CALL(cudaMemcpy(gd_ac4Scn, acScn + giScnOffset, giScnMemSzUChar, cudaMemcpyHostToDevice));
	WrapKerTim("CopyFrameToGPUMem", 1);
	WrapTim("CopyFrameToGPUMem");

	clEnqueueWriteBuffer(queue, cl_gd_ac4Scn, CL_TRUE, 0, giScnMemSzUChar, acScn + giScnOffset, 0, NULL, NULL);

	InitTim();

#ifdef FPGA
	////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Set the kernel arguments 
	status = clSetKernelArg(convertChar4ToFloatDoConGam_kernel, 0, sizeof(cl_mem), (void*)&cl_gd_ac4Scn);
	checkError(status, "Failed to set kernel arg 0");
	status = clSetKernelArg(convertChar4ToFloatDoConGam_kernel, 1, sizeof(cl_mem), (void*)&cl_gd_afPadScnIn);
	checkError(status, "Failed to set kernel arg 1");
	int sc = (giScnSz / 4);
	status = clSetKernelArg(convertChar4ToFloatDoConGam_kernel, 2, sizeof(int), (void*)&sc);
	checkError(status, "Failed to set kernel arg 2");
	int bc = bConGam; // bool to int
	status = clSetKernelArg(convertChar4ToFloatDoConGam_kernel, 3, sizeof(int), (void*)&bc);
	checkError(status, "Failed to set kernel arg 3");

	// Configure work set over which the kernel will execute
	//size_t wgSize[3] = { 1, 1, 1 };
	//size_t gSize[3] = { 1, 1, 1 };
	g_time = getCurrentTimestamp();

	// Launch the kernel
	status = clEnqueueNDRangeKernel(queue, convertChar4ToFloatDoConGam_kernel, 1, NULL, gSize, wgSize, 0, NULL, NULL);
	checkError(status, "Failed to launch kernel");
	//clReleaseEvent(*write_event);
#ifdef cltime
	status = clFinish(queue);
	g_time = getCurrentTimestamp() - g_time;
	printf("\tconvertChar4ToFloatDoConGam_kernel processing time = %.4fms\n", (float)(g_time * 1E3));
#endif

	status = clFinish(queue);
	checkError(status, "Failed to finish");

	float4* h_afPadScnIn;
	h_afPadScnIn = (float4*)_aligned_malloc(giScnMemSzReal, AOCL_ALIGNMENT);
	status = clEnqueueReadBuffer(queue, cl_gd_afPadScnIn, CL_TRUE, 0, giScnMemSzReal, h_afPadScnIn, 0, NULL, NULL);
	CUDA_SAFE_CALL(cudaMemcpy((float4*)d_afPadScnIn, h_afPadScnIn, giScnMemSzReal, cudaMemcpyHostToDevice));

	_aligned_free(h_afPadScnIn);
	////////////////////////////////////////////////////////////////////////////////////////////////////////

#else
	InitKerTim(5);
	convertChar4ToFloatDoConGam << <gdBlocksConv, gdThreadsConv >> > (gd_ac4Scn, (float4*)d_afPadScnIn, (giScnSz / 4), bConGam);
	WrapKerTim("convertChar4ToFloatDoConGam", 5);
#endif
	WrapTim("convertChar4ToFloatDoConGam");

	if (bFixDead)
	{
		InitTim();
		//InitKerTim(5);
		fixDeadPixels << <gdBlocksDead, gdThreadsDead >> > ((cufftReal*)d_afPadScnIn, giScnSz, giScnW, giScnH);
		//WrapKerTim("fixDeadPixels", 5);
		WrapTim("fixDeadPixel");
	}

#ifdef COPYBACKAFTERDEADFIX
	//only for visualization purposes. no need to optimize below code with kernels.
	/*cufftReal* h_afScnOut = (cufftReal*)malloc(giScnMemSzReal);
	CUDA_SAFE_CALL(cudaMemcpy(h_afScnOut, d_afPadScnIn, giScnMemSzReal, cudaMemcpyDeviceToHost));
	for (int i = 0; i < giScnSz; i++)
	acScn[i + giScnOffset] = (unsigned char)h_afScnOut[i];
	free(h_afScnOut);//????*/
#endif

#ifdef SAVEFIXEDSCN
	cufftReal* h_afScn = (cufftReal*)malloc(giScnMemSzReal);
	CUDA_SAFE_CALL(cudaMemcpy(h_afScn, d_afPadScnIn, giScnMemSzReal, cudaMemcpyDeviceToHost));
	FILE *fFixedScn = fopen(getFullPathOfFile("fixedScn.bin"), "wb");
	fwrite(h_afScn, sizeof(cufftReal), giScnSz, fFixedScn);
	fclose(fFixedScn);
	free(h_afScn);
#endif

#ifdef RUNCLAHE
#ifdef PARTIM
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	CUT_SAFE_CALL(sdkResetTimer(&guiParTim));
	CUT_SAFE_CALL(sdkStartTimer(&guiParTim));
#endif
	//IMPLEMENT THIS SECTION ON GPU: only for testing CLAHE it is running on the CPU
	cufftReal* h_afScnClahe = (cufftReal*)_aligned_malloc(giScnMemSzReal, AOCL_ALIGNMENT);
	CUDA_SAFE_CALL(cudaMemcpy(h_afScnClahe, d_afPadScnIn, giScnMemSzReal, cudaMemcpyDeviceToHost));
	//status = clEnqueueReadBuffer(queue, cl_gd_afPadScnIn, CL_FALSE, 0, giScnMemSzReal, h_afScnClahe, 0, NULL, NULL);
	checkError(status, "Failed to read buffer cl_gd_afPadScnIn");
	unsigned char* acScnClahe = (unsigned char*)_aligned_malloc(giScnMemSzUChar, AOCL_ALIGNMENT);
	for (int i = 0; i < giScnSz; i++)
	{
		acScnClahe[i] = (unsigned char)h_afScnClahe[i];
	}
	//convert to unsigned int
	CLAHE(acScnClahe, giScnW, giScnH, 0, 255, giScnW / 8, giScnH / 8, 256, 0.3f); //80 60, 80 30
																				  //copy scene to device
#ifdef FPGA
	////////////////////////////////////////////////////////////////////////////////////////////////////////
	clEnqueueWriteBuffer(queue, cl_gd_ac4Scn, CL_TRUE, 0, giScnMemSzUChar, acScnClahe, 0, NULL, NULL);
	//uchar4* h_ac4Scn;
	//float4* h_afPadScnIn;
	//h_ac4Scn = (uchar4*)_aligned_malloc(giScnMemSzUChar, AOCL_ALIGNMENT);
	h_afPadScnIn = (float4*)_aligned_malloc(giScnMemSzReal, AOCL_ALIGNMENT);
	//CUDA_SAFE_CALL(cudaMemcpy(h_ac4Scn, gd_ac4Scn, giScnMemSzUChar, cudaMemcpyDeviceToHost));// copy memory to host
	//CUDA_SAFE_CALL(cudaMemcpy(h_afPadScnIn, (float4*)d_afPadScnIn, giScnMemSzReal, cudaMemcpyDeviceToHost));// copy memory to host

	// Set the kernel arguments 
	status = clSetKernelArg(convertChar4ToFloatDoConGam_kernel, 0, sizeof(cl_mem), (void*)&cl_gd_ac4Scn);
	checkError(status, "Failed to set kernel arg 0");
	status = clSetKernelArg(convertChar4ToFloatDoConGam_kernel, 1, sizeof(cl_mem), (void*)&cl_gd_afPadScnIn);
	checkError(status, "Failed to set kernel arg 1");
	status = clSetKernelArg(convertChar4ToFloatDoConGam_kernel, 2, sizeof(int), (void*)&sc);
	checkError(status, "Failed to set kernel arg 2");
	status = clSetKernelArg(convertChar4ToFloatDoConGam_kernel, 3, sizeof(int), (void*)&bc);
	checkError(status, "Failed to set kernel arg 3");

	// Configure work set over which the kernel will execute
	//size_t wgSize[3] = { 1, 1, 1 };
	//size_t gSize[3] = { 1, 1, 1 };
	g_time = getCurrentTimestamp();

	// Launch the kernel
	status = clEnqueueNDRangeKernel(queue, convertChar4ToFloatDoConGam_kernel, 1, NULL, gSize, wgSize, 0, NULL, NULL);
	checkError(status, "Failed to launch kernel");
	//clReleaseEvent(*write_event);

	//Read back data 
	//status = clEnqueueReadBuffer(queue, cl_gd_ac4Scn, CL_TRUE, 0, giScnMemSzUChar, h_ac4Scn, 0, NULL, NULL);
	//checkError(status, "Failed to read buffer cl_gd_ac4Scn");
	status = clEnqueueReadBuffer(queue, cl_gd_afPadScnIn, CL_TRUE, 0, giScnMemSzReal, h_afPadScnIn, 0, NULL, NULL);
	checkError(status, "Failed to read buffer cl_d_afPadScnIn");

	//Free CL buffer
	//status = clReleaseMemObject(cl_gd_ac4Scn);
	//checkError(status, "Failed to release buffer");
	//status = clReleaseMemObject(cl_d_afPadScnIn);
	//checkError(status, "Failed to release buffer");
	// Wait for command queue to complete pending events
	status = clFinish(queue);
	checkError(status, "Failed to finish");


	// Free the resources allocated
	//AOCLcleanup();
	//CUDA_SAFE_CALL(cudaMemcpy(gd_ac4Scn, h_ac4Scn, giScnMemSzUChar, cudaMemcpyHostToDevice));
	//_aligned_free(h_ac4Scn);
	CUDA_SAFE_CALL(cudaMemcpy((float4*)d_afPadScnIn, h_afPadScnIn, giScnMemSzReal, cudaMemcpyHostToDevice));
	_aligned_free(h_afPadScnIn);
	////////////////////////////////////////////////////////////////////////////////////////////////////////

#else
	CUDA_SAFE_CALL(cudaMemcpy(gd_ac4Scn, acScnClahe, giScnMemSzUChar, cudaMemcpyHostToDevice));
	convertChar4ToFloatDoConGam << <gdBlocksConv, gdThreadsConv >> >(gd_ac4Scn, (float4*)d_afPadScnIn, (giScnSz / 4), bConGam);
#endif
	_aligned_free(h_afScnClahe);
	//free(h_afScnOut);
	_aligned_free(acScnClahe);
#ifdef PARTIM
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	CUT_SAFE_CALL(sdkStopTimer(&guiParTim));
	g_dClaheTime = sdkGetTimerValue(&guiParTim);
	printf("Clahe (runs on the CPU!) time: %f msecs.\n", g_dClaheTime);
#endif
	////////////////////
#endif

#ifdef COPYBACK
	//only for visualization purposes. no need to optimize below code with kernels.
	cufftReal* h_afScnOut = (cufftReal*)malloc(giScnMemSzReal);
	CUDA_SAFE_CALL(cudaMemcpy(h_afScnOut, d_afPadScnIn, giScnMemSzReal, cudaMemcpyDeviceToHost));
	for (int i = 0; i<giScnSz; i++)
		acScn[i + giScnOffset] = (unsigned char)h_afScnOut[i];
	free(h_afScnOut);
#endif
	//added to show CLAHE effect in the ssd_fft_gpu_GUI
	if (giShowClaheGUI == -1)
	{
		//only for visualization purposes. no need to optimize below code with kernels.
		cufftReal* h_afScnOutGUI = (cufftReal*)malloc(giScnMemSzReal);
		CUDA_SAFE_CALL(cudaMemcpy(h_afScnOutGUI, d_afPadScnIn, giScnMemSzReal, cudaMemcpyDeviceToHost));
		for (int i = 0; i<giScnSz; i++)
			acScn[i + giScnOffset] = (unsigned char)h_afScnOutGUI[i];
		free(h_afScnOutGUI);
	}
}


void DisplayResults(float fPSR, int iTplIndx, int iIPIndx, int iSzIndx, int iStatsFrameCur)
{
	giSLCurFrm = -1;
	giSLResult = -1;
#ifdef DISP_FRM_RECOG
	if (fPSR > gfPSRTrashold)
	{
		//printf("Max PSR value: %f (TplNo = %d, IPAng = %d, Sz = %d)\n", fPSR, gstCompFlt.aiTpl_no[iTplIndx], gstCompFlt.aiIPAngs[iIPIndx], gstCompFlt.aiTplCols[iSzIndx]);
		printf("Frame votes for %3d %s (PSR: %5.2f, in-plane rotation: %3d\xf8, size: %2d)\n", gstCompFlt.aiTpl_no[iTplIndx], acMeasure, fPSR, gstCompFlt.aiIPAngs[iIPIndx], gstCompFlt.aiTplCols[iSzIndx]);
		giSLCurFrm = gstCompFlt.aiTpl_no[iTplIndx];
	}
	//	else
	//		printf("\n");
#endif

#ifdef MAJVOT
	int iNumTpl = gstCompFlt.iNumOrigFlt - gstCompFlt.iNumMulCompFlt;
	float fAddConfIP, fAddConfSz;

	//update the AccRes
	if (giFrameNo == 0)
	{
		if (fPSR > gfPSRTrashold)
		{
			//start the tracking at the first seen sign
			giFrameNo++;
			gastAccRes[iTplIndx].fAccConf = gastAccRes[iTplIndx].fAccConf + fPSR;
			giNumFramesInAcc++;
			gastAccRes[iTplIndx].iPrevIP = iIPIndx;
			gastAccRes[iTplIndx].iPrevSz = iSzIndx;
		}
	}
	else
	{
		//increase the tracked frameNum regardless of the PSR value if we already started the tracking
		giFrameNo++;
		if (fPSR > gfPSRTrashold)
		{
			fAddConfIP = 0;
			fAddConfSz = 0;
			if (gastAccRes[iTplIndx].fAccConf > 0)
			{
				//there has been a previous recognition of this tpl (iPrevIP and iPrevSz has valid values)
				//increase confidence if IP is the same as previous and/or Sz is getting larger.
				if ((iIPIndx - gastAccRes[iTplIndx].iPrevIP) == 0)
					fAddConfIP = gfAddConfIPFac*fPSR;
				if ((iSzIndx - gastAccRes[iTplIndx].iPrevSz) == 0)
					fAddConfSz = gfAddConfEqSzFac*fPSR;
				else if ((iSzIndx - gastAccRes[iTplIndx].iPrevSz) > 0)
					fAddConfSz = gfAddConfGrSzFac*fPSR;
			}
			gastAccRes[iTplIndx].fAccConf = gastAccRes[iTplIndx].fAccConf + fPSR + fAddConfIP + fAddConfSz;
			giNumFramesInAcc++;
			gastAccRes[iTplIndx].iPrevIP = iIPIndx;
			gastAccRes[iTplIndx].iPrevSz = iSzIndx;
		}
	}

	int iMaxTplIndx = -1;
	if (giFrameNo == giTrackingLen)
	{
		//find the bestTpl
		float fMaxAccConf = gastAccRes[0].fAccConf;
		iMaxTplIndx = 0;
		for (int i = 1; i<iNumTpl; i++)
		{
			if (gastAccRes[i].fAccConf > fMaxAccConf)
			{
				iMaxTplIndx = i;
				fMaxAccConf = gastAccRes[i].fAccConf;
			}
		}
		//printf("\n           Tpl = %d (Max AccConf = %f)\n\n", gstCompFlt.aiTpl_no[iMaxTplIndx], fMaxAccConf);
#ifdef REALTIME
		if (fMaxAccConf > gfAccPSRTrasholdSpecialReal && giNumFramesInAcc == 1 && gstCompFlt.aiTpl_no[iMaxTplIndx] != 2)
			printf("\n           Best Tpl = %d (Max AccConf = %f)\n(special rule for realtime emulation=> result is based on only ONE frame with VERY high confidence)\n\n", gstCompFlt.aiTpl_no[iMaxTplIndx], fMaxAccConf);
		else if (fMaxAccConf > gfAccPSRTrashold && gstCompFlt.aiTpl_no[iMaxTplIndx] != 2) //2 = 00t
			printf("\n           Best Tpl = %d (Max AccConf = %f)\n\n", gstCompFlt.aiTpl_no[iMaxTplIndx], fMaxAccConf);
		else
			iMaxTplIndx = -1;
#else
		if (fMaxAccConf > gfAccPSRTrashold && gstCompFlt.aiTpl_no[iMaxTplIndx] != 2) //2 = 00t
		{
			//printf("\n           Best Tpl = %d (Max AccConf = %f)\n\n", gstCompFlt.aiTpl_no[iMaxTplIndx], fMaxAccConf);
			printf("\n\n\n\n     System concludes that speed limit is %3d %s! (Total votes: %6.2f)\n\n\n\n\n", gstCompFlt.aiTpl_no[iMaxTplIndx], acMeasure, fMaxAccConf);
			giSLResult = gstCompFlt.aiTpl_no[iMaxTplIndx];
		}
		else
			iMaxTplIndx = -1;
#endif
		giFrameNo = 0;
		giNumFramesInAcc = 0;
		//initialize the accpsr to zero
		memset(gastAccRes, '\0', (iNumTpl * sizeof(AccRes_struct_t)));
	}

#ifdef STATS
	// Print the best sign found in the sequence of frames to the stats file
	if (iMaxTplIndx != -1)
		fprintf(g_fStatsFile, "%d\t%d\n", iStatsFrameCur, gstCompFlt.aiTpl_no[iMaxTplIndx]);
	else
		fprintf(g_fStatsFile, "%d\t0\n", iStatsFrameCur);
#endif
#endif
}

#ifndef US_SIGNS 
#ifdef STATS
void IncFPSCount(unsigned long ulTimeStamp, int iFrameCur)
{
	int iDiff = ulTimeStamp - g_ulLastTimeStamp;
	if (iDiff >= 61 && iDiff <= 64) //mostly 62, 63
		gi16fps++;
	else if (iDiff >= 123 && iDiff <= 126) //mostly 124, 125
		gi8fps++;
	else if (iDiff >= 185 && iDiff <= 188) //mostly 186, 187
		gi5fps++;
	else if (iDiff >= 247 && iDiff <= 250) //mostly 248, 249
		gi4fps++;
	else if (iDiff == 0)
		gi0fps++;
	else
		fprintf(g_fStatsFile, "%d\tTimeNotKnown\t%d\n", iFrameCur, iDiff);

}
#endif
#endif
////////////////////////////////////////////////////////////////////////////////
// Member Functions
////////////////////////////////////////////////////////////////////////////////
void ssd_fft_gpu_init()
{
	///////////////////////////////////////////////////////////////
	accumulate = 0;
	cl_int status;

	if (!init()) {
		return;
	}
	/*
	// Set the kernel argument (argument 0)
	status = clSetKernelArg(kernel, 0, sizeof(cl_int), (void*)&thread_id_to_output);
	checkError(status, "Failed to set kernel arg 0");

	printf("\nKernel initialization is complete.\n");
	printf("Launching the kernel...\n\n");

	// Configure work set over which the kernel will execute
	size_t wgSize[3] = { work_group_size, 1, 1 };
	size_t gSize[3] = { work_group_size, 1, 1 };

	// Launch the kernel
	status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, gSize, wgSize, 0, NULL, NULL);
	checkError(status, "Failed to launch kernel");

	// Wait for command queue to complete pending events
	status = clFinish(queue);
	checkError(status, "Failed to finish");

	printf("\nKernel execution is complete.\n");

	// Free the resources allocated
	AOCLcleanup();*/
	////////////////////////////////////////////////////////////////

#ifdef CMPTEST
	cmpTest();
#endif

#ifdef DISP_DEV_INIT
	//display the device info
	int iDeviceCount;
	cudaGetDeviceCount(&iDeviceCount);
	printf("Device Count: %d\n", iDeviceCount);
	cudaSetDevice(0); //when using animas since there is only 1 GPU (G80) ID must be 0 (otherwise it fails on fist fft execution), on burn we can set it to any ID we want
	int iDev;
	struct cudaDeviceProp prop;
	cudaGetDevice(&iDev);
	cudaGetDeviceProperties(&prop, iDev);
	printf("The Properties of the Device with ID %d are\n", iDev);
	printf("\tDevice Name : %s\n", prop.name);
	printf("\tDevice Total Global Memory Size (MBs) : %u\n", prop.totalGlobalMem / 1048576); //1 Megabyte = 1048576 Bytes
	printf("\tDevice Total Constant Memory Size (KBs) : %u\n", prop.totalConstMem / 1024); //1 KB = 1024 Bytes
	printf("\tDevice # of MultiProcessors : %d\n", prop.multiProcessorCount); //1SM(streaming processor has 8 SM (streaming processors = cores)
#endif
#ifdef DISP_DEV_INIT
	printf("Read composite filters...\n");
#endif
	//read comp filters
	gstCompFlt = readCompFlt();//can't read
	giTplH = gstCompFlt.iH;//
	giTplW = gstCompFlt.iW;//
						   //printf("giTplW=%d\n", giTplW);
						   //printf("giTplH=%d\n", giTplH);
	giTplSz = giTplH * giTplW;
	giTplWMemSz = giTplW * sizeof(cufftReal);
	giTplMemSzReal = giTplH * giTplW * sizeof(cufftReal);
	giTplMemSzCmplx = giTplH * giTplW * sizeof(cufftComplex);
	giTplSzPad = giPadTplH * giPadTplW;
	giTplWMemSzPad = giPadTplW * sizeof(cufftReal);
	giTplMemSzRealPad = giPadTplH * giPadTplW * sizeof(cufftReal);
	giTplMemSzCmplxPad = giPadTplH * giPadTplW * sizeof(cufftComplex);
	giNumIPRot = gstCompFlt.iNumIPRot;
	giNumSz = gstCompFlt.iNumSz;
	giNumOrigFlt = gstCompFlt.iNumOrigFlt;
	giNumSngCompFlt = giNumOrigFlt - gstCompFlt.iNumMulCompFlt;


	//do some check
	giPartMaxGDx = (giTplSz) % (BLOCKDIMX_MAX*EACHTHREADREADS) > 0 ? ((giTplSz) / (BLOCKDIMX_MAX*EACHTHREADREADS)) + 1 : (giTplSz) / (BLOCKDIMX_MAX*EACHTHREADREADS);
	if (giPartMaxGDx > 1)
	{
		printf("Warning: Max of part scn can not be found in one pass!\n");
	}
	giWholeMaxGDx = (giScnSzPad) % (BLOCKDIMX_MAX*EACHTHREADREADS) > 0 ? ((giScnSzPad) / (BLOCKDIMX_MAX*EACHTHREADREADS)) + 1 : (giScnSzPad) / (BLOCKDIMX_MAX*EACHTHREADREADS);
	if ((giWholeMaxGDx / EACHTHREADREADS) > BLOCKDIMX_MAX)
	{
		//in the second pass each thread will read EACHTHREADREADS blockmaxs. There is giWholeMaxGDx blocks at most.
		//if giWholeMaxGDx/EACHTHREADREADS > BLOCKDIMX_MAX this means that second pass should have more than one block.
		//but it should have only one!
		printf("Error: Each thread in max kernel should read more than %d elements!\n", EACHTHREADREADS);
		exit(0);
	}
#ifdef DISP_DEV_INIT
	printf("Allocating memory...\n");
#endif
#ifdef PINNED_MEM
	CUDA_SAFE_CALL(cudaMallocHost((void**)&gh_acScn, giOrigScnMemSzUChar));
	CUDA_SAFE_CALL(cudaMallocHost((void**)&gh_afArea, giAreaMemSzReal));
#else
	gh_acScn = (unsigned char *)malloc(giOrigScnMemSzUChar);
	gh_afArea = (cufftReal *)malloc(giAreaMemSzReal);
#endif
	cl_afPadScnInPad = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, giScnMemSzRealPad, NULL, NULL);
	cl_afPadScnOutPad = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, giScnMemSzCmplxPad, NULL, NULL);
	//cl_gd_afMulFit = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, giScnMemSzRealPad, NULL, NULL);
	//cl_afCorr = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, giScnMemSzRealPad, NULL, NULL);
	d_tmp0 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, sizeof(float2) * NS * (NS + 1), NULL, &status);
	d_tmp1 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, sizeof(float2) * NT * NT / 2, NULL, &status);
	d_tmp02 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, sizeof(float2) * NS, NULL, &status);
	d_tmp12 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, sizeof(float2) * NT, NULL, &status);
	CUDA_SAFE_CALL(cudaMalloc((void **)&gd_ac4Scn, giScnMemSzUChar));
	cl_gd_ac4Scn = clCreateBuffer(context, CL_MEM_READ_WRITE, giScnMemSzUChar, NULL, NULL);//
	CUDA_SAFE_CALL(cudaMalloc((void **)&gd_afCompFlt, gstCompFlt.iDataMemSz));
	cl_gd_afCompFlt = clCreateBuffer(context, CL_MEM_READ_WRITE, gstCompFlt.iDataMemSz, NULL, NULL);
	CUDA_SAFE_CALL(cudaMalloc((void **)&gd_afPadScnIn, giScnMemSzReal));
	CUDA_SAFE_CALL(cudaMalloc((void **)&gd_afPadScnInPad, giScnMemSzRealPad));
	CUDA_SAFE_CALL(cudaMalloc((void **)&gd_afPadScnInPadC, giScnMemSzCmplxPad));
	cl_gd_afPadScnIn = clCreateBuffer(context, CL_MEM_READ_WRITE, giScnMemSzReal, NULL, NULL);
	CUDA_SAFE_CALL(cudaMalloc((void **)&gd_afCorr, giScnMemSzRealPad));
	cl_gd_afCorr = clCreateBuffer(context, CL_MEM_READ_WRITE, giScnMemSzReal, NULL, NULL);
	CUDA_SAFE_CALL(cudaMalloc((void **)&gd_afPadScnOut, giScnMemSzCmplx));
	CUDA_SAFE_CALL(cudaMalloc((void **)&gd_afPadScnOutPad, giScnMemSzCmplxPad));
	cl_gd_afPadScnOut = clCreateBuffer(context, CL_MEM_READ_WRITE, giScnMemSzCmplxPad, NULL, NULL);
	CUDA_SAFE_CALL(cudaMalloc((void **)&gd_afMul, giScnMemSzCmplxPad));
	cl_gd_afMul = clCreateBuffer(context, CL_MEM_READ_WRITE, giScnMemSzCmplxPad, NULL, NULL);
	CUDA_SAFE_CALL(cudaMalloc((void **)&gd_afScnPartIn, giTplMemSzReal));
	CUDA_SAFE_CALL(cudaMalloc((void **)&gd_afScnPartInC, giTplMemSzCmplx));
	cl_gd_afScnPartIn = clCreateBuffer(context, CL_MEM_READ_WRITE, giTplMemSzReal, NULL, NULL);
	CUDA_SAFE_CALL(cudaMalloc((void **)&gd_afScnPartOut, giTplMemSzCmplx));
	cl_gd_afScnPartOut = clCreateBuffer(context, CL_MEM_READ_WRITE, giTplMemSzCmplx, NULL, NULL);
	CUDA_SAFE_CALL(cudaMalloc((void**)&gd_pfMax, sizeof(cufftReal)));
	cl_gd_pfMax = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cufftReal), NULL, NULL);
	CUDA_SAFE_CALL(cudaMalloc((void**)&gd_afBlockMaxs, sizeof(cufftReal)*giWholeMaxGDx));//no need
	CUDA_SAFE_CALL(cudaMalloc((void**)&gd_piMaxIdx, sizeof(int)));
	cl_gd_piMaxIdx = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);
	CUDA_SAFE_CALL(cudaMalloc((void**)&gd_aiBlockMaxIdxs, sizeof(int)*giWholeMaxGDx));//no need
	//cl_afCorr = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, giTplMemSzRealPad + 2, NULL, NULL);

	gd_afWholeTplFFT = NULL;
	gd_afPartTplFFT = NULL;
	gd_afPadTplIn = NULL;
	gd_afPadTplOut = NULL;

#ifdef DISP_DEV_INIT
	printf("Scene size      : %i x %i\n", giScnW, giScnH);
	printf("Template size	: %i x %i\n", giTplW, giTplH);
#endif

	//calculate the block and grid size (both will be 1D) to be used by kernels
	getKernelDims(BLOCKDIMX, giScnSz / 4, &gdThreadsConv, &gdBlocksConv);
	getKernelDims(BLOCKDIMX, giScnSz / 2, &gdThreadsDead, &gdBlocksDead);
	gdThreadsDead.x = gdThreadsDead.x + (HALFWARP + 1);
	getKernelDims(BLOCKDIMX, giScnSzPad, &gdThreadsWhole, &gdBlocksWhole);
	getKernelDims(BLOCKDIMX, giTplSz, &gdThreadsPart, &gdBlocksPart);

	//Creating FFT plan for whole scene
	CUFFT_SAFE_CALL(cufftPlan2d(&ghFFTplanWholeFwd, giPadScnH, giPadScnW, CUFFT_R2C));
	CUFFT_SAFE_CALL(cufftPlan2d(&ghFFTplanWholeInv, giPadScnH, giPadScnW, CUFFT_C2R));
	//CUFFT_SAFE_CALL(cufftPlan2d(&ghFFTplanWholeFwdC, giPadScnH, giPadScnW, CUFFT_C2C));
	//CUFFT_SAFE_CALL(cufftPlan2d(&ghFFTplanWholeInvC, giPadScnH, giPadScnW, CUFFT_C2C));
	//Creating FFT plan for part of the scene
	CUFFT_SAFE_CALL(cufftPlan2d(&ghFFTplanPartFwd, giTplH, giTplW, CUFFT_R2C));
	CUFFT_SAFE_CALL(cufftPlan2d(&ghFFTplanPartInv, giTplH, giTplW, CUFFT_C2R));
	//CUFFT_SAFE_CALL(cufftPlan2d(&ghFFTplanPartFwdC, giTplH, giTplW, CUFFT_C2C));
	//CUFFT_SAFE_CALL(cufftPlan2d(&ghFFTplanPartInvC, giTplH, giTplW, CUFFT_C2C));

	//CUT_SAFE_CALL( sdkCreateTimer(&guiParTim) );
	sdkCreateTimer(&guiParTim);
	//CUT_SAFE_CALL( sdkCreateTimer(&guiKerTim) );
	sdkCreateTimer(&guiKerTim);

	InitTim();
	//copy all Composite Filters to device memory (copying device to device would take less time)
	CUDA_SAFE_CALL(cudaMemcpy(gd_afCompFlt, gstCompFlt.h_afData, gstCompFlt.iDataMemSz, cudaMemcpyHostToDevice));
	clEnqueueWriteBuffer(queue, cl_gd_afCompFlt, CL_FALSE, 0, gstCompFlt.iDataMemSz, gstCompFlt.h_afData, 0, NULL, NULL);
	//figure out params regarding IPRot
#ifdef DoIPInSecond
	giBegIdxIPInFirst = giNumIPRot / 2; //middle is the not-IProtated compFlt
	giEndIdxIPInFirst = giBegIdxIPInFirst + 1;
	giNumIPInFirst = 1;
	giBegIdxIPInSecond = 0;
	giEndIdxIPInSecond = giNumIPRot;
#else
	giBegIdxIPInFirst = 0;
	giEndIdxIPInFirst = giNumIPRot;
	giNumIPInFirst = giNumIPRot;
	//assign second pass params on-line
#endif
	PrepTplFFT(cl_gd_afCompFlt, gd_afCompFlt, &gd_afPadTplIn, &gd_afPadTplOut, &gd_afWholeTplFFT, &gd_afPartTplFFT, ghFFTplanWholeFwd, ghFFTplanPartFwd);
	WrapTim("PrepTplFFT");
	if (gbConGam)
	{
		genLUT();
		CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_acLUT, gacLUT, sizeof(unsigned char)*LUTSIZE));
	}

#ifndef REALTIME
	gfAccPSRTrashold = 31.5f;//32;//before: gfAccPSRFac*gfPSRTrashold if clahelimit = 0.6f fac = 4, if clahelimit= 0.3f (less noise) fac = 5 to avoid FP, TN
#else
	gfAccPSRTrashold = 25.95f;
#endif
}

void BestTpl(unsigned char* acScn, int* piMaxPeakIndx, int* piPartW, int* piPartH, int* file_info, unsigned long ulTimeStamp)
{

	int iIPIndx, iSzIndx, iFltIndx, iPeakIndx;

	float fPSR, fMaxPSR;

	int iMaxIPIndx, iMaxSzIndx, iMaxFltIndx;

	int iPartWMemSz;

	giShowClaheGUI = file_info[2];
	//save the scn in bin file (to transfer the videos from Realis to GUI)
#ifdef SAVESCNBIN
	// If the first frameID
	if (file_info[0] == file_info[1])
	{
		if (g_fScnBin != NULL)
			fclose(g_fScnBin);
		char acScnName[] = "00000.txt";
		itoa(file_info[1], acScnName, 10);
		strcpy(g_sScnBinPath, g_sScnBinPathBegin);
		strcat(g_sScnBinPath, acScnName);
		strcat(g_sScnBinPath, ".bin\0");
		g_fScnBin = fopen(g_sScnBinPath, "wb");
	}
	fwrite(acScn, sizeof(unsigned char), giOrigScnSz, g_fScnBin);
#endif

#ifdef STATS
	// current Frame ID 
	int iFrameCur = file_info[0];
	// First Frame ID
	int iFrameBeg = file_info[1];
	// If the first frameID
	if (iFrameCur == iFrameBeg)
	{
		//close prev stats file 
		if (g_fStatsFile != NULL)
		{
			//close the stats file for prev video
			fclose(g_fStatsFile);
			//add this videos time to all video time
			g_iNumVideos++;
			g_fAllVideoTime = g_fAllVideoTime + (float)(((double)g_ulLastTimeStamp - (double)g_ulFirstTimeStamp) / 1000 / 60);
		}
		//start the time to calculate current video time
		g_ulFirstTimeStamp = ulTimeStamp;
		//open a stats file for current video
		strcpy(g_sStatsPath, g_sStatsPathBegin);
#ifndef US_SIGNS
		char acFName[] = "00000.txt";
		itoa(iFrameBeg, acFName, 10);
		strcat(g_sStatsPath, acFName);
#else
		strcat(g_sStatsPath, gacClipName);
#endif
		strcat(g_sStatsPath, ".txt\0");
		g_fStatsFile = fopen(g_sStatsPath, "wb");
		if (g_fStatsFile == NULL)
			printf("Error openning stats file!");
	}
	else //if not the first frame in the video, increment the appropriate FPS(frames per second) counter
	{
#ifndef US_SIGNS
		IncFPSCount(ulTimeStamp, iFrameCur);
#endif
	}
	g_ulLastTimeStamp = ulTimeStamp;
#endif

	*piMaxPeakIndx = -1;
	*piPartW = -1;
	*piPartH = -1;
#ifdef REALTIME
	int iTimeDiff;
	if (file_info[0] > file_info[1]) //if it is not the first frame 
	{
		iTimeDiff = ulTimeStamp - g_ulPrevTimeStamp;
		if (iTimeDiff < g_iRuntime) //do not process this frame
		{
#ifdef STATS
			fprintf(g_fStatsFile, "%d\t-1\n", iFrameCur); //enter -1 as speed sign found
#endif
			return;
		}
		else
			g_ulPrevTimeStamp = ulTimeStamp;
	}
	else
		g_ulPrevTimeStamp = ulTimeStamp;
#endif

	bool bLoadScn = false;
	//Read scene...
	if (acScn == NULL)
	{
		//no video input, process the scn from file
		FILE *fScn = fopen(getFullPathOfFile("scn.bin"), "rb");
		fread(gh_acScn, sizeof(unsigned char), giOrigScnSz, fScn);
		fclose(fScn);
		acScn = gh_acScn;
		bLoadScn = true;
	}
	/*	else
	{
	FILE *fScnIn = fopen(getFullPathOfFile("scnV.bin"), "wb");
	fwrite(acScn, sizeof(unsigned char), giOrigScnSz, fScnIn);
	fclose(fScnIn);
	FILE *fScn = fopen(getFullPathOfFile("scnV.bin"), "rb");
	fread(gh_acScn, sizeof(unsigned char), giOrigScnSz, fScn);
	fclose(fScn);
	acScn = gh_acScn;
	}
	*/
	bool bFixDead = gbFixDead;
	if (bLoadScn) bFixDead = 0;

	////////FIRST PASS///////////
#ifdef ALLTIM
	unsigned int uiAllTim;
	CUT_SAFE_CALL(cutCreateTimer(&uiAllTim));
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	CUT_SAFE_CALL(cutResetTimer(uiAllTim));
	CUT_SAFE_CALL(cutStartTimer(uiAllTim));
#endif
#ifdef PARTIM
	g_dRunsOnGPUTotalTime = 0;
#endif
#ifdef KERTIM
	g_dTotalKerTime = 0;
#endif

	CpyScnToDevAndPreProcess(acScn, gd_afPadScnIn, gbConGam, bFixDead);
	CUDA_SAFE_CALL(cudaMemset(gd_afPadScnInPad, 0, giScnMemSzRealPad));
	CUDA_SAFE_CALL(cudaMemcpy2D(gd_afPadScnInPad, (giPadScnW * sizeof(cufftReal)), gd_afPadScnIn, giScnW * sizeof(cufftReal), giScnW * sizeof(cufftReal), giScnH, cudaMemcpyDeviceToDevice));

	//CUDA_SAFE_CALL(cudaMemset(gd_afPadScnInPadC, 0, giScnMemSzCmplxPad));
	//CUDA_SAFE_CALL(cudaMemcpy2D(gd_afPadScnInPadC, 2 * sizeof(float), gd_afPadScnInPad, sizeof(float), sizeof(float), giPadScnW * giPadScnH, cudaMemcpyDeviceToDevice));
	//Running the correlation...
	InitTim();
	//take the FFT of the scene
	//InitKerTim(5);
	cl_afPadScnInPad = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, giScnMemSzRealPad, NULL, NULL);
	cl_afPadScnOutPad = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, giScnMemSzCmplxPad, NULL, NULL);
	cufftReal* h_afPadScnInPad = (cufftReal*)_aligned_malloc(giScnMemSzRealPad, AOCL_ALIGNMENT);
	CUDA_SAFE_CALL(cudaMemcpy(h_afPadScnInPad, gd_afPadScnInPad, giScnMemSzRealPad, cudaMemcpyDeviceToHost));
	clEnqueueWriteBuffer(queue, cl_afPadScnInPad, CL_FALSE, 0, giScnMemSzRealPad, h_afPadScnInPad, 0, NULL, NULL);
	fftScn(false, &cl_afPadScnInPad, &cl_afPadScnOutPad);
	transpose(cl_afPadScnOutPad);
	cufftComplex* h_afPadScnOutPad = (cufftComplex*)_aligned_malloc(giScnMemSzCmplxPad, AOCL_ALIGNMENT);
	status = clEnqueueReadBuffer(queue4, cl_afPadScnOutPad, CL_TRUE, 0, giScnMemSzCmplxPad, h_afPadScnOutPad, 0, NULL, NULL);
	CUDA_SAFE_CALL(cudaMemcpy(gd_afPadScnOutPad, h_afPadScnOutPad, giScnMemSzCmplxPad, cudaMemcpyHostToDevice));
	clReleaseMemObject(cl_afPadScnInPad);
	clReleaseMemObject(cl_afPadScnOutPad);
	//CUFFT_SAFE_CALL(cufftExecR2C(ghFFTplanWholeFwd, (cufftReal *)gd_afPadScnInPad, (cufftComplex *)gd_afPadScnOutPad));
	//WrapKerTim("wholeFFT", 5);
	//apply kth law to scene




//clKthLaw(float2* gd_afPadScnOutPad, int giScnSzPad);
	
#ifdef FPGA
	clKthLaw(gd_afPadScnOutPad, giScnSzPad);
#else
	InitKerTim(5);
	kthLaw << <gdBlocksWhole, gdThreadsWhole >> >(gd_afPadScnOut, giScnSz);
	WrapKerTim("kthLaw", 5);
#endif
	WrapKerTim("wholeKth", 2);
	//initialize max PSR value
	fMaxPSR = INT_MIN;
	//First find the peak with MulCompFlts
	WrapTim("FirstPassInit");
	InitTim();
	for (iFltIndx = giNumSngCompFlt; iFltIndx < giNumOrigFlt; iFltIndx++)
	{
		for (iSzIndx = 0; iSzIndx < giNumSz; iSzIndx++)
		{
			for (iIPIndx = giBegIdxIPInFirst; iIPIndx < giEndIdxIPInFirst; iIPIndx++)
			{
				//I am not initializing gh_afArea. make sure you reach right coords.
				//getWholeTplFFT(gd_afCompFlt, iIPIndx, iSzIndx, iFltIndx, gd_afPadTplIn, &gd_afPadTplOut, ghFFTplanWholeFwd);
				int iFltAbsIndx = iFltIndx - giNumSngCompFlt;// from 0 to giNumOrigFlt-giNumSngCompFlt
				int iIPAbsIndx = iIPIndx - giBegIdxIPInFirst;
				//perform correlation

				Corr(&cl_gd_afWholeTplFFT, cl_gd_afWholeTplFFT(iIPAbsIndx, iSzIndx, iFltAbsIndx), gdBlocksWhole, gdThreadsWhole, &cl_gd_afPadScnOut, giScnSzPad, gd_afMul, ghFFTplanWholeInv, gd_afCorr, gh_afArea, &iPeakIndx, &fPSR, giPadScnW, giPadScnH, false);
				//printf("PSR value for MulCompFlt: %f (iFltIndx = %d IPAng = %d, Sz = %d)\n", fPSR, iFltIndx, gstCompFlt.aiIPAngs[iIPIndx], gstCompFlt.aiTplCols[iSzIndx]);

				if (fPSR > fMaxPSR)
				{
					fMaxPSR = fPSR;
					iMaxIPIndx = iIPIndx;
					iMaxSzIndx = iSzIndx;
					*piMaxPeakIndx = iPeakIndx;
				}
			}
		}
	}
#ifndef DoIPInSecond
	giBegIdxIPInSecond = iMaxIPIndx;
	giEndIdxIPInSecond = giBegIdxIPInSecond + 1;
#endif
	WrapTim("FirstPassLoop");
#ifdef CHECKRES
#ifndef DoIPInSecond
	if (bLoadScn) //if processing a scn from file(no video input), and trying IPRots in first pass
	{
		//make sure this is the last tpl
		if (iFltIndx == giNumOrigFlt && iSzIndx == giNumSz && iIPIndx == giNumIPRot)
		{
			cmpCPU(gd_afCorr, "resMulFFTInv.bin", 0, giScnSz, 0, (float)1e-6);
			cmpCPU(&fPSR, "PSR.bin", 0, 1, 1, (float)1e-6);
		}
	}
#endif
#endif
	////////SECOND PASS///////////
	InitTim();
	//we know the max IP and Sz. Now try different templates
	//copy template-size portion of the scene starting at peak point
	//	CUDA_SAFE_CALL( cudaMemcpy2D( gd_afScnPartIn, giTplWMemSz, gd_afPadScnIn+iMaxPeakIndx, giScnW*sizeof(cufftReal), giTplWMemSz, giTplH , cudaMemcpyDeviceToDevice ));
	getCopyWidthHeight(*piMaxPeakIndx, piPartW, piPartH);
	printf("piMaxPeakIndx=%d\n", *piMaxPeakIndx);
	int ConvertPeakIndx = *piMaxPeakIndx % giPadScnW + (*piMaxPeakIndx / giPadScnW) * 640;
	printf("ConvertPeakIndx=%d\n", ConvertPeakIndx);
	iPartWMemSz = *piPartW * sizeof(cufftReal);
	//make sure you initialize gd_afScnPartIn with zeros before processing each frame (if we are out of bounds, we will have a part image padded with zeros)
	InitKerTim(3);
	CUDA_SAFE_CALL(cudaMemset(gd_afScnPartIn, 0, giTplMemSzReal));
	CUDA_SAFE_CALL(cudaMemcpy2D(gd_afScnPartIn, giTplWMemSz, gd_afPadScnIn + ConvertPeakIndx, giScnW * sizeof(cufftReal), iPartWMemSz, *piPartH, cudaMemcpyDeviceToDevice));

	WrapKerTim("MemcpyD2DPart", 3);
	//take the FFT of the scene
	InitKerTim(3);
	CUFFT_SAFE_CALL(cufftExecR2C(ghFFTplanPartFwd, (cufftReal *)gd_afScnPartIn, (cufftComplex *)gd_afScnPartOut));
	WrapKerTim("partFFT", 3);

	
	//apply kth law to scene
#ifdef FPGA
	clKthLaw(gd_afScnPartOut, giTplSz);
#else
	InitKerTim(5);
	kthLaw << <gdBlocksPart, gdThreadsPart >> >(gd_afScnPartOut, giTplSz);
	WrapKerTim("partKth", 5);
#endif
	fMaxPSR = INT_MIN;
	WrapTim("SecondPassInit");
	InitTim();
	for (iFltIndx = 0; iFltIndx < giNumSngCompFlt; iFltIndx++)
	{
		for (iIPIndx = giBegIdxIPInSecond; iIPIndx < giEndIdxIPInSecond; iIPIndx++)
		{
			//getPartTplFFT(gd_afCompFlt, iIPIndx, iMaxSzIndx, iFltIndx, &gd_afPadTplOut, ghFFTplanPartFwd, gd_afPartTplFFT);
			Corr(&cl_gd_afPartTplFFT, cl_gd_afPartTplFFT(iIPIndx, iMaxSzIndx, iFltIndx), gdBlocksPart, gdThreadsPart, &cl_d_afScnPartOut, giTplSz, gd_afMul, ghFFTplanPartInv, gd_afCorr, gh_afArea, &iPeakIndx, &fPSR, giTplW, giTplH, true);
			//Corr2(gd_afPadTplOut, gdBlocksPart, gdThreadsPart, gd_afScnPartOut, giTplSz, gd_afMul, ghFFTplanPartInv, gd_afCorr, gh_afArea, &iPeakIndx, &fPSR, giTplW, giTplH, true);
			if (fPSR > fMaxPSR)
			{
				fMaxPSR = fPSR;
				iMaxFltIndx = iFltIndx;
				iMaxIPIndx = iIPIndx;
			}
		}
	}
	WrapTim("SecondPassLoop");
	status = clReleaseMemObject(cl_d_afScnPartOut);
	checkError(status, "Failed to release buffer");
#ifdef CHECKRES
#ifndef DoIPInSecond
	if (bLoadScn) //if processing a scn from file(no video input), and trying IPRots in first pass
	{
		//make sure this is the first tpl (the one before MulCompFlts)
		if (iFltIndx == giNumSngCompFlt)
		{
			cmpCPU(&fPSR, "PSRPart.bin", 0, 1, 1, (float)1e-4);
		}
	}
#endif
#endif

#ifdef KERTIM
	printf("Kernel time: %f msecs.\n", g_dTotalKerTime);
#endif
#ifdef PARTIM
	printf("GPU time: %f msecs.\n", g_dRunsOnGPUTotalTime);
	printf("\nRuntime(GPU time + Clahe): %f msecs.\n\n", g_dRunsOnGPUTotalTime + g_dClaheTime);
#endif
#ifdef ALLTIM
	CUT_SAFE_CALL(cutStopTimer(uiAllTim));
	double gpuTime = sdkGetTimerValue(uiAllTim);
	//#ifndef PARTIM
	printf("Runtime(GPU time + Clahe): %f msecs.\n", gpuTime);
	//#endif
#endif

	DisplayResults(fMaxPSR, iMaxFltIndx, iMaxIPIndx, iMaxSzIndx, file_info[0]);
	//in realis show the peak in correct position (add offset of the window in the frame)
#ifdef SHOWBOX_WHENRECOG
	if (fMaxPSR <= gfPSRTrashold) //hide the box upper left corner, if the PSR is below trashold
		*piMaxPeakIndx = 0 - ((giTplH*giScnW * 2) + giScnOffset);
#endif
	*piMaxPeakIndx = *piMaxPeakIndx + giScnOffset;
	//printf("MaxPeakIndx: %d, FrameID: %d\n", *piMaxPeakIndx, file_info[0]);
}


void ssd_fft_gpu_findBestTpl(unsigned char* acScn, int* piMaxPeakIndx, int* piPartW, int* piPartH, int* file_info, unsigned long ulTimeStamp)
{
	BestTpl(acScn, piMaxPeakIndx, piPartW, piPartH, file_info, ulTimeStamp);
}


void ssd_fft_gpu_returnBestTpl(unsigned char* acScn, int* piMaxPeakIndx, int* piPartW, int* piPartH, int* file_info, unsigned long ulTimeStamp, int* iSLCurFrm, int* iSLResult, char* acClipName)
{
	strcpy(gacClipName, acClipName);
	BestTpl(acScn, piMaxPeakIndx, piPartW, piPartH, file_info, ulTimeStamp);
	*iSLCurFrm = giSLCurFrm;
	*iSLResult = giSLResult;
	//time1 = getCurrentTimestamp() - time1;
	//accumulate = accumulate + time1;
	//printf("\tProcessing time = %.4fms\n", (float)(accumulate * 1E3));
	//time1 = getCurrentTimestamp();
}


void ssd_fft_gpu_exit() {
#ifndef US_SIGNS
#ifdef STATS
	//add the last video time	
	g_iNumVideos++;
	g_fAllVideoTime = g_fAllVideoTime + (float)(((double)g_ulLastTimeStamp - (double)g_ulFirstTimeStamp) / 1000 / 60);
	//write all video time to the file
	strcpy(g_sStatsPath, g_sStatsPathBegin);
	strcat(g_sStatsPath, "AllVideoTime.txt\0");
	g_fStatsFile = fopen(g_sStatsPath, "wb");
	if (g_fStatsFile == NULL)
		printf("Error openning stats file for measuring all video time!");
	fprintf(g_fStatsFile, "%d\t%f\t%d\t%d\t%d\t%d\t%d\n", g_iNumVideos, g_fAllVideoTime, gi16fps, gi8fps, gi5fps, gi4fps, gi0fps);
	fclose(g_fStatsFile);
#endif
#endif

	printf("Shutting down...\n");

	CUFFT_SAFE_CALL(cufftDestroy(ghFFTplanWholeFwd));
	CUFFT_SAFE_CALL(cufftDestroy(ghFFTplanWholeInv));
	CUFFT_SAFE_CALL(cufftDestroy(ghFFTplanPartFwd));
	CUFFT_SAFE_CALL(cufftDestroy(ghFFTplanPartInv));
	//CUFFT_SAFE_CALL(cufftDestroy(ghFFTplanWholeFwdC));
	//CUFFT_SAFE_CALL(cufftDestroy(ghFFTplanWholeInvC));
	//CUFFT_SAFE_CALL(cufftDestroy(ghFFTplanPartFwdC));
	//CUFFT_SAFE_CALL(cufftDestroy(ghFFTplanPartInvC));
	CUDA_SAFE_CALL(cudaFree(gd_ac4Scn));
	//clReleaseMemObject(cl_gd_afMulFit);
	//clReleaseMemObject(cl_afCorr);
	clReleaseMemObject(d_tmp0);
	clReleaseMemObject(d_tmp1);
	clReleaseMemObject(d_tmp02);
	clReleaseMemObject(d_tmp12);
	clReleaseMemObject(cl_gd_ac4Scn);
	CUDA_SAFE_CALL(cudaFree(gd_afPadScnIn));
	clReleaseMemObject(cl_gd_afPadScnIn);
	CUDA_SAFE_CALL(cudaFree(gd_afScnPartIn));
	clReleaseMemObject(cl_gd_afScnPartIn);
	CUDA_SAFE_CALL(cudaFree(gd_afScnPartOut));
	clReleaseMemObject(cl_gd_afScnPartOut);
	CUDA_SAFE_CALL(cudaFree(gd_afCompFlt));
	clReleaseMemObject(cl_gd_afCompFlt);
	CUDA_SAFE_CALL(cudaFree(gd_afPadScnOut));
	clReleaseMemObject(cl_gd_afPadScnOut);
	CUDA_SAFE_CALL(cudaFree(gd_afCorr));
	clReleaseMemObject(cl_gd_afCorr);
	CUDA_SAFE_CALL(cudaFree(gd_afMul));
	clReleaseMemObject(cl_gd_afMul);
	CUDA_SAFE_CALL(cudaFree(gd_pfMax));
	clReleaseMemObject(cl_gd_pfMax);
	CUDA_SAFE_CALL(cudaFree(gd_afBlockMaxs));//no need
	CUDA_SAFE_CALL(cudaFree(gd_piMaxIdx));
	clReleaseMemObject(cl_gd_piMaxIdx);
	CUDA_SAFE_CALL(cudaFree(gd_aiBlockMaxIdxs));// no need i guess
	//clReleaseMemObject(cl_afCorr);
	DestroyTplFFT(gd_afWholeTplFFT, gd_afPartTplFFT, gd_afPadTplIn, gd_afPadTplOut);
#ifdef PINNED_MEM
	cudaFreeHost(gh_acScn);
	cudaFreeHost(gh_afArea);
	cudaFreeHost(gstCompFlt.aiIPAngs);
	cudaFreeHost(gstCompFlt.aiTplCols);
	cudaFreeHost(gstCompFlt.aiTpl_no);
	cudaFreeHost(gstCompFlt.h_afData);
	cudaFreeHost(gastAccRes);
#else
	free(gh_acScn);
	free(gh_afArea);
	free(gstCompFlt.aiIPAngs);
	free(gstCompFlt.aiTplCols);
	free(gstCompFlt.aiTpl_no);
	_aligned_free(gstCompFlt.h_afData);
	free(gastAccRes);
#endif
	//CUT_EXIT(argc, argv);
}

bool init() {
	cl_int status;

	if (!setCwdToExeDir()) {
		return false;
	}

	// Get the OpenCL platform.
	platform = findPlatform("Intel");
	if (platform == NULL) {
		printf("ERROR: Unable to find Intel FPGA OpenCL platform.\n");
		return false;
	}

	// User-visible output - Platform information
	{
		char char_buffer[STRING_BUFFER_LEN];
		printf("Querying platform for info:\n");
		printf("==========================\n");
		clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
		printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
		clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
		printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
		clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
		printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);
	}

	// Query the available OpenCL devices.
	scoped_array<cl_device_id> devices;
	cl_uint num_devices;

	devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));

	// We'll just use the first device.
	device = devices[0];

	// Display some device information.
	display_device_info(device);

	// Create the context.
	context = clCreateContext(NULL, 1, &device, &oclContextCallback, NULL, &status);
	checkError(status, "Failed to create context");

	// Create the command queue.
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create command queue");
	queue2 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create command queue2");
	queue3 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create command queue3");
	queue4 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create command queue4");

	// Create the program.
	std::string binary_file = getBoardBinaryFile("ssd_fft_fpga_kernel", device);
	printf("Using AOCX: %s\n", binary_file.c_str());
	program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

	// Build the program that was just created.
	status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
	checkError(status, "Failed to build program");

	// Create the kernel - name passed in here must match kernel name in the
	// original CL file, that was compiled into an AOCX file using the AOC tool
	const char *kernel_name = "fetch0";
	fetch_kernel0 = clCreateKernel(program, kernel_name, &status);
	checkError(status, "Failed to create kernel fetch0");

	kernel_name = "fetch1";
	fetch_kernel1 = clCreateKernel(program, kernel_name, &status);
	checkError(status, "Failed to create kernel fetch1");

	kernel_name = "fft2d0";
	fft_kernel0 = clCreateKernel(program, kernel_name, &status);
	checkError(status, "Failed to create kernel fft2d0");

	kernel_name = "fft2d1";
	fft_kernel1 = clCreateKernel(program, kernel_name, &status);
	checkError(status, "Failed to create kernel fft2d1");

	kernel_name = "transpose0";
	transpose_kernel0 = clCreateKernel(program, kernel_name, &status);
	checkError(status, "Failed to create kernel transpose0");

	kernel_name = "transpose1";
	transpose_kernel1 = clCreateKernel(program, kernel_name, &status);
	checkError(status, "Failed to create kernel transpose1");

	kernel_name = "extraCol0";
	extraCol_kernel0 = clCreateKernel(program, kernel_name, &status);
	checkError(status, "Failed to create kernel extraCol0");

	kernel_name = "extraCol1";
	extraCol_kernel1 = clCreateKernel(program, kernel_name, &status);
	checkError(status, "Failed to create kernel extraCol1");

	kernel_name = "kthLaw";  // Kernel name, as defined in the CL file
	kthLaw_kernel = clCreateKernel(program, kernel_name, &status);
	checkError(status, "Failed to create kernel kthLaw");

	kernel_name = "pointWiseMul";  // Kernel name, as defined in the CL file
	pointWiseMul_kernel = clCreateKernel(program, kernel_name, &status);
	checkError(status, "Failed to create kernel pointWiseMul");

	kernel_name = "pointWiseMul2";  // Kernel name, as defined in the CL file
									//pointWiseMul_kernel2 = clCreateKernel(program, kernel_name, &status);
									//checkError(status, "Failed to create kernel pointWiseMul");

	kernel_name = "ComplexScale";  // Kernel name, as defined in the CL file
	ComplexScale_kernel = clCreateKernel(program, kernel_name, &status);
	checkError(status, "Failed to create kernel ComplexScale");

	kernel_name = "convertChar4ToFloatDoConGam";  // Kernel name, as defined in the CL file
	convertChar4ToFloatDoConGam_kernel = clCreateKernel(program, kernel_name, &status);
	checkError(status, "Failed to create kernel convertChar4ToFloatDoConGam");

	kernel_name = "fixDeadPixels";  // Kernel name, as defined in the CL file
									//fixDeadPixels_kernel = clCreateKernel(program, kernel_name, &status);
									//checkError(status, "Failed to create kernel fixDeadPixels");

	kernel_name = "max_k";  // Kernel name, as defined in the CL file
	max_k_kernel = clCreateKernel(program, kernel_name, &status);
	checkError(status, "Failed to create kernel max_k");
	return true;
}

// Free the resources allocated during initialization
void AOCLcleanup() { //cleanup CL
	if (fetch_kernel0) {
		clReleaseKernel(fetch_kernel0);
	}
	if (fetch_kernel1) {
		clReleaseKernel(fetch_kernel1);
	}
	if (fft_kernel0) {
		clReleaseKernel(fft_kernel0);
	}
	if (fft_kernel1) {
		clReleaseKernel(fft_kernel1);
	}
	if (transpose_kernel0) {
		clReleaseKernel(transpose_kernel0);
	}
	if (transpose_kernel1) {
		clReleaseKernel(transpose_kernel1);
	}
	if (extraCol_kernel0) {
		clReleaseKernel(extraCol_kernel0);
	}
	if (extraCol_kernel1) {
		clReleaseKernel(extraCol_kernel1);
	}
	if (kthLaw_kernel) {
		clReleaseKernel(kthLaw_kernel);
	}
	if (pointWiseMul_kernel) {
		clReleaseKernel(pointWiseMul_kernel);
	}
	if (pointWiseMul_kernel2) {
		clReleaseKernel(pointWiseMul_kernel2);
	}
	if (ComplexScale_kernel) {
		clReleaseKernel(ComplexScale_kernel);
	}
	if (convertChar4ToFloatDoConGam_kernel) {
		clReleaseKernel(convertChar4ToFloatDoConGam_kernel);
	}
	if (fixDeadPixels_kernel) {
		clReleaseKernel(fixDeadPixels_kernel);
	}
	if (max_k_kernel) {
		clReleaseKernel(fixDeadPixels_kernel);
	}
	if (program) {
		clReleaseProgram(program);
	}
	if (queue) {
		clReleaseCommandQueue(queue);
	}
	if (queue2) {
		clReleaseCommandQueue(queue2);
	}
	if (queue3) {
		clReleaseCommandQueue(queue3);
	}
	if (queue4) {
		clReleaseCommandQueue(queue4);
	}
	if (context) {
		clReleaseContext(context);
	}
}

// Helper functions to display parameters returned by OpenCL queries
static void device_info_ulong(cl_device_id device, cl_device_info param, const char* name) {
	cl_ulong a;
	clGetDeviceInfo(device, param, sizeof(cl_ulong), &a, NULL);
	printf("%-40s = %lu\n", name, a);
}
static void device_info_uint(cl_device_id device, cl_device_info param, const char* name) {
	cl_uint a;
	clGetDeviceInfo(device, param, sizeof(cl_uint), &a, NULL);
	printf("%-40s = %u\n", name, a);
}
static void device_info_bool(cl_device_id device, cl_device_info param, const char* name) {
	cl_bool a;
	clGetDeviceInfo(device, param, sizeof(cl_bool), &a, NULL);
	printf("%-40s = %s\n", name, (a ? "true" : "false"));
}
static void device_info_string(cl_device_id device, cl_device_info param, const char* name) {
	char a[STRING_BUFFER_LEN];
	clGetDeviceInfo(device, param, STRING_BUFFER_LEN, &a, NULL);
	printf("%-40s = %s\n", name, a);
}

// Query and display OpenCL information on device and runtime environment
static void display_device_info(cl_device_id device) {

	printf("Querying device for info:\n");
	printf("========================\n");
	device_info_string(device, CL_DEVICE_NAME, "CL_DEVICE_NAME");
	device_info_string(device, CL_DEVICE_VENDOR, "CL_DEVICE_VENDOR");
	device_info_uint(device, CL_DEVICE_VENDOR_ID, "CL_DEVICE_VENDOR_ID");
	device_info_string(device, CL_DEVICE_VERSION, "CL_DEVICE_VERSION");
	device_info_string(device, CL_DRIVER_VERSION, "CL_DRIVER_VERSION");
	device_info_uint(device, CL_DEVICE_ADDRESS_BITS, "CL_DEVICE_ADDRESS_BITS");
	device_info_bool(device, CL_DEVICE_AVAILABLE, "CL_DEVICE_AVAILABLE");
	device_info_bool(device, CL_DEVICE_ENDIAN_LITTLE, "CL_DEVICE_ENDIAN_LITTLE");
	device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE");
	device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE");
	device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_SIZE, "CL_DEVICE_GLOBAL_MEM_SIZE");
	device_info_bool(device, CL_DEVICE_IMAGE_SUPPORT, "CL_DEVICE_IMAGE_SUPPORT");
	device_info_ulong(device, CL_DEVICE_LOCAL_MEM_SIZE, "CL_DEVICE_LOCAL_MEM_SIZE");
	device_info_ulong(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, "CL_DEVICE_MAX_CLOCK_FREQUENCY");
	device_info_ulong(device, CL_DEVICE_MAX_COMPUTE_UNITS, "CL_DEVICE_MAX_COMPUTE_UNITS");
	device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_ARGS, "CL_DEVICE_MAX_CONSTANT_ARGS");
	device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE");
	device_info_uint(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
	device_info_uint(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, "CL_DEVICE_MEM_BASE_ADDR_ALIGN");
	device_info_uint(device, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE");
	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR");
	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT");
	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT");
	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG");
	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT");
	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE");

	{
		cl_command_queue_properties ccp;
		clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &ccp, NULL);
		printf("%-40s = %s\n", "Command queue out of order? ", ((ccp & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) ? "true" : "false"));
		printf("%-40s = %s\n", "Command queue profiling enabled? ", ((ccp & CL_QUEUE_PROFILING_ENABLE) ? "true" : "false"));
	}
}