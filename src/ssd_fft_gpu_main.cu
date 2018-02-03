
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
#define d_pafWholeTplFFT(iIPAbsIndx, iSzIndx, iFltAbsIndx) d_pafWholeTplFFT + ((giScnH * giScnW * giNumIPInFirst * giNumSz) * iFltAbsIndx) + ((giScnH * giScnW * giNumIPInFirst) * iSzIndx) + ((giScnH * giScnW) * iIPAbsIndx)
#define gd_afWholeTplFFT(iIPAbsIndx, iSzIndx, iFltAbsIndx) gd_afWholeTplFFT + ((giScnH * giScnW * giNumIPInFirst * giNumSz) * iFltAbsIndx) + ((giScnH * giScnW * giNumIPInFirst) * iSzIndx) + ((giScnH * giScnW) * iIPAbsIndx)
#define d_pafPartTplFFT(iIPIndx, iSzIndx, iFltIndx) d_pafPartTplFFT + ((giTplH * giTplW * giNumIPRot * giNumSz) * iFltIndx) + ((giTplH * giTplW * giNumIPRot) * iSzIndx) + ((giTplH * giTplW) * iIPIndx)
#define gd_afPartTplFFT(iIPIndx, iSzIndx, iFltIndx) gd_afPartTplFFT + ((giTplH * giTplW * giNumIPRot * giNumSz) * iFltIndx) + ((giTplH * giTplW * giNumIPRot) * iSzIndx) + ((giTplH * giTplW) * iIPIndx)
////////////////////////////////////////////////////////////////////////////////
// Global vars
////////////////////////////////////////////////////////////////////////////////
size_t wgSize[3] = { 1, 1, 1 };
size_t gSize[3] = { 1, 1, 1 };
//trashold for the PSR (might be different for day and night)
//const float gfPSRTrashold = 8.0f;
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
const int	giScnMemSzReal = giScnSz * sizeof(cufftReal);
const int   giScnMemSzCmplx = giScnSz * sizeof(cufftComplex);
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





				//unsigned int guiParTim;
StopWatchInterface *guiParTim;
//unsigned int guiKerTim;
StopWatchInterface *guiKerTim;
double g_dRunsOnGPUTotalTime;
double g_dTotalKerTime;
double g_dClaheTime;

int giTplH, giTplW, giTplSz, giTplWMemSz, giTplMemSzReal, giTplMemSzCmplx;
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
*gd_afCorr;

//typedef float cufftComplex[2];
cufftComplex
*gd_afScnPartOut,
*gd_afPadTplOut,
*gd_afPadScnOut,
*gd_afWholeTplFFT,
*gd_afPartTplFFT,
*gd_afMul;

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

//compare GPU results to CPU results
void cmpCPU(void* afVals, char* pcFileName, bool bComplex, int iSz, bool bHost, float fEpsilon)
{
	int iMemSzReal = iSz * sizeof(cufftReal);
	FILE* fRef = fopen(getFullPathOfFile(pcFileName), "rb");

	if (bComplex)
	{
		//file has both real and imaginary values
		int iMemSzCmplx = iSz * sizeof(cufftComplex);
		cufftComplex* h_afCmplx;
		if (!bHost)
		{
			h_afCmplx = (cufftComplex *)malloc(iMemSzCmplx);
			// copy result from device to host
			CUDA_SAFE_CALL(cudaMemcpy(h_afCmplx, (cufftComplex*)afVals, iMemSzCmplx, cudaMemcpyDeviceToHost));
		}
		else
		{
			h_afCmplx = (cufftComplex *)afVals;
		}
		//extract real and imaginary parts
		float* h_afReal = (float*)malloc(iMemSzReal);
		float* h_afImag = (float*)malloc(iMemSzReal);
		for (int iI = 0; iI < iSz; iI++)
		{
			h_afReal[iI] = h_afCmplx[iI].x;
			h_afImag[iI] = h_afCmplx[iI].y;
		}

		// allocate mem to hold CPU results 
		float* afRealRef = (float*)malloc(iMemSzReal);
		float* afImagRef = (float*)malloc(iMemSzReal);
		fread(afRealRef, sizeof(float), iSz, fRef);
		fread(afImagRef, sizeof(float), iSz, fRef);

		CUTBoolean cutbResReal = cutComparefe(afRealRef, h_afReal, iSz, fEpsilon);
		CUTBoolean cutbResImag = cutComparefe(afImagRef, h_afImag, iSz, fEpsilon);
		printf("Checking %s result: %s\n", pcFileName, (1 == (cutbResReal && cutbResImag)) ? "PASSED" : "FAILED");


		//clean up memory
		if (!bHost) { free(h_afCmplx); }
		free(h_afReal);
		free(h_afImag);

		free(afRealRef);
		free(afImagRef);
	}
	else
	{
		//file has only real values
		int iMemSzReal = iSz * sizeof(cufftReal);
		cufftReal* h_afReal;
		if (!bHost)
		{
			h_afReal = (cufftReal *)malloc(iMemSzReal);
			// copy result from device to host
			CUDA_SAFE_CALL(cudaMemcpy(h_afReal, (cufftReal*)afVals, iMemSzReal, cudaMemcpyDeviceToHost));
		}
		else
		{
			h_afReal = (cufftReal *)afVals;
		}

		// allocate mem to hold CPU results 
		float* afRealRef = (float*)malloc(iMemSzReal);
		fread(afRealRef, sizeof(float), iSz, fRef);

		CUTBoolean cutbResReal = cutComparefe(afRealRef, h_afReal, iSz, fEpsilon);
		printf("Checking %s result: %s\n", pcFileName, (cutbResReal) ? "PASSED" : "FAILED");


		//clean up memory
		if (!bHost) { free(h_afReal); }
		free(afRealRef);
	}
	fclose(fRef);

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
	gstCompFlt.h_afData = (float *)malloc(gstCompFlt.iDataMemSz);
	gastAccRes = (float *)malloc(iNumAccResMemSz);
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
	if (iSz == giTplSz)
	{
		CUDA_SAFE_CALL(cudaThreadSynchronize());
		CUT_SAFE_CALL(cutResetTimer(guiKerTim));
		CUT_SAFE_CALL(cutStartTimer(guiKerTim));
	}
#endif
}

inline void WrapKerTim(char* sKerName, int iSz)
{
#ifdef KERTIM
	if (iSz == giTplSz) //1(copyscn convert fix), 2(1stPassInit), 3(2ndPassInit), giScnSz (1stLoop), giTplSz(2ndLoop)
	{
		CUDA_SAFE_CALL(cudaThreadSynchronize());
		CUT_SAFE_CALL(cutStopTimer(guiKerTim));
		double dTime = sdkGetTimerValue(guiKerTim);
		printf("%s time: %f msecs.\n", sKerName, dTime);
		g_dTotalKerTime += dTime;
	}
#endif
}

void MaxIdx(cufftReal* d_afData, int iSz, int** d_piMaxIdx)
{
	int iGDx;
	if (iSz == giScnSz)
		iGDx = giWholeMaxGDx; // = (640*480/512*8) = (307200/4096) = 75 - will need two passes 
	else
		iGDx = giPartMaxGDx;; // if TplSz = 60, (60*60/512*8)+1 = (3600/4096)+1 = 1 - will only need one pass 
							  //if TplSz is larger it is possible that we need two passes.
							  //gd_afBlockMaxs have enough storage for finding max in whole scene.
							  //so it is definitely enough for finding max in part scene

							  //max will do 2 passes. In the first pass there will be several blocks. 
							  //In the second	there will be only one block.

							  //now do the first pass: each thread will read EACHTHREADREADS pixels. 
							  //Each block reads BLOCKDIMX_MAX*EACHTHREADREADS = 512*8 = 4096 pixels

	dim3 thread(BLOCKDIMX_MAX, 1, 1);
	dim3 grid(iGDx, 1);
	
	//calculate block maxs
	InitKerTim(iSz);
	max_k << < grid, thread >> >(d_afData, NULL, iSz, gd_afBlockMaxs, gd_aiBlockMaxIdxs);
	WrapKerTim("Max1stPass", iSz);
	CUT_CHECK_ERROR("Kernel execution failed");

	if (iGDx == 1)
	{
		*d_piMaxIdx = gd_aiBlockMaxIdxs;
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
		InitKerTim(iSz);
		max_k << < grid2, thread2 >> >(gd_afBlockMaxs, gd_aiBlockMaxIdxs, iGDx, gd_pfMax, gd_piMaxIdx);
		WrapKerTim("Max2ndPass", iSz);
		*d_piMaxIdx = gd_piMaxIdx;
		CUT_CHECK_ERROR("Kernel execution failed");
	}
}

//compute PSR value
float getPSR(cufftReal* gd_afCorr, cufftReal* gh_afArea, int* iPeakIndx, int iSz, int iW, int iH)
{
	int iI;
	int *d_piMaxIdx = NULL;
	MaxIdx(gd_afCorr, iSz, &d_piMaxIdx);
	InitKerTim(iSz);
	CUDA_SAFE_CALL(cudaMemcpy(iPeakIndx, (int*)d_piMaxIdx, sizeof(int), cudaMemcpyDeviceToHost));
	WrapKerTim("MemcpyD2HPeak", iSz);
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
	InitKerTim(iSz);
	CUDA_SAFE_CALL(cudaMemcpy2D(gh_afArea, iNewAreaW * sizeof(cufftReal), gd_afCorr + iStart, iW * sizeof(cufftReal), iNewAreaW * sizeof(cufftReal), iNewAreaH, cudaMemcpyDeviceToHost));
	WrapKerTim("MemcpyD2HArea", iSz);
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

void Corr(cufftComplex* d_afTplOut, dim3 dBlocks, dim3 dThreads, cufftComplex* d_afScnOut, int iSz, cufftComplex* gd_afMul, cufftHandle hFFTplanInv, cufftReal* gd_afCorr, cufftReal* gh_afArea, int* piPeakIndx, float* pfPSR, int iW, int iH, bool szi)
{   
	//take conjugate of template fft and point wise multiply with scene and scale it with image size
	InitKerTim(iSz);
	pointWiseMul << <dBlocks, dThreads >> >(gd_afMul, d_afScnOut, d_afTplOut, iSz, 1.0f / (float)iSz);
	WrapKerTim("Mul", iSz);
	CUT_CHECK_ERROR("pointWiseMul() execution failed\n");
	//take inverse FFT of multiplication
	InitKerTim(iSz);
	CUFFT_SAFE_CALL(cufftExecC2R(hFFTplanInv, (cufftComplex *)gd_afMul, (cufftReal *)gd_afCorr));
	WrapKerTim("FFTinv", iSz);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	//find the PSR
	*pfPSR = getPSR(gd_afCorr, gh_afArea, piPeakIndx, iSz, iW, iH);
	return;
}
inline void InitTim()
{
#ifdef PARTIM
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	CUT_SAFE_CALL(sdkResetTimer(guiParTim));
	CUT_SAFE_CALL(sdkStartTimer(guiParTim));
#endif
}

inline void WrapTim(char* sParName)
{
#ifdef PARTIM
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	CUT_SAFE_CALL(sdkStopTimer(guiParTim));
	double dTime = sdkGetTimerValue(guiParTim);
	printf("%s time: %f msecs.\n", sParName, dTime);
	g_dRunsOnGPUTotalTime += dTime;
#endif
}

void PrepTplFFT(cufftReal* gd_afCompFlt, cufftReal** d_pafPadTplIn, cufftComplex** d_pafPadTplOut, cufftComplex** d_pafWholeTplFFT, cufftComplex** d_pafPartTplFFT, cufftHandle ghFFTplanWholeFwd, cufftHandle ghFFTplanPartFwd)
{
#ifdef SAVEFFT
	int iSzIndx, iIPIndx, iFltIndx, iFltAbsIndx, iIPAbsIndx;
	cufftReal
		*d_afTpl,
		*d_afPadTplIn;
	//first allocate mem
	//WholeTpls are the MulCompFlts (last flts in the compflt list). They are used in 1st pass. Their size is as big as scn
	//PartTpls are all other comp flt excluding MulCompFlts. They are used in 2nd pass. Their size is as big as tpl (is not blowed up to scn size)
	int iWholeMemSz = giScnH * giScnW * giNumIPInFirst * giNumSz * gstCompFlt.iNumMulCompFlt * sizeof(cufftComplex);
	CUDA_SAFE_CALL(cudaMalloc((void **)&*d_pafWholeTplFFT, iWholeMemSz));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_afPadTplIn, giScnMemSzReal));
	int iPartMemSz = giTplH * giTplW * giNumIPRot * giNumSz * giNumSngCompFlt * sizeof(cufftComplex);
	CUDA_SAFE_CALL(cudaMalloc((void **)&*d_pafPartTplFFT, iPartMemSz));
	//take FFT of WholeTpls
	for (iFltIndx = giNumSngCompFlt; iFltIndx < giNumOrigFlt; iFltIndx++)
	{
		for (iSzIndx = 0; iSzIndx < giNumSz; iSzIndx++)
		{
			for (iIPIndx = giBegIdxIPInFirst; iIPIndx < giEndIdxIPInFirst; iIPIndx++)
			{
				CUDA_SAFE_CALL(cudaMemset(d_afPadTplIn, 0, giScnMemSzReal));
				d_afTpl = gd_afCompFlt(iIPIndx, iSzIndx, iFltIndx);
				//pad template
				CUDA_SAFE_CALL(cudaMemcpy2D(d_afPadTplIn, (giScnW * sizeof(cufftReal)), d_afTpl, giTplWMemSz, giTplWMemSz, giTplH, cudaMemcpyDeviceToDevice));
				//take the fft and save it to WholeTplFFT
				iFltAbsIndx = iFltIndx - giNumSngCompFlt;
				iIPAbsIndx = iIPIndx - giBegIdxIPInFirst;
				//printf("iIPIndx=%d iSzIndx=%d iFltIndx=%d d_afPadTplIn= %d\n", iIPIndx, iSzIndx, iFltIndx, d_afPadTplIn);
				CUFFT_SAFE_CALL(cufftExecR2C(ghFFTplanWholeFwd, (cufftReal *)d_afPadTplIn, (cufftComplex *)*d_pafWholeTplFFT(iIPAbsIndx, iSzIndx, iFltAbsIndx)));

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
				d_afTpl = gd_afCompFlt(iIPIndx, iSzIndx, iFltIndx);
				//printf("iIPIndx=%d iSzIndx=%d iFltIndx=%d d_afTpl= %d\n", iIPIndx, iSzIndx, iFltIndx, d_afTpl);
				//take the fft and save it to PartTplFFT
				CUFFT_SAFE_CALL(cufftExecR2C(ghFFTplanPartFwd, (cufftReal *)d_afTpl, (cufftComplex *)*d_pafPartTplFFT(iIPIndx, iSzIndx, iFltIndx)));
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
#else
	CUDA_SAFE_CALL(cudaFree(gd_afPadTplIn));
	CUDA_SAFE_CALL(cudaFree(gd_afPadTplOut));
#endif
}

void getWholeTplFFT(cufftReal* gd_afCompFlt, int iIPIndx, int iSzIndx, int iFltIndx, cufftReal* gd_afPadTplIn, cufftComplex** d_pafPadTplOut, cufftHandle ghFFTplanWholeFwd, cufftComplex* gd_afWholeTplFFT)
{
#ifdef SAVEFFT
	int iFltAbsIndx = iFltIndx - giNumSngCompFlt;
	int iIPAbsIndx = iIPIndx - giBegIdxIPInFirst;
	*d_pafPadTplOut = gd_afWholeTplFFT(iIPAbsIndx, iSzIndx, iFltAbsIndx);
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
	Indx2Coord(giScnW, iMaxPeakIndx, &iMaxPeakRow, &iMaxPeakCol);
	*piPartW = giTplW;
	int iEndCol = iMaxPeakCol + *piPartW - 1;
	if (iEndCol >= giScnW)
		*piPartW = *piPartW - (iEndCol + 1 - giScnW);
	*piPartH = giTplH;
	int iEndRow = iMaxPeakRow + *piPartH - 1;
	if (iEndRow >= giScnH)
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
	WrapKerTim("MemcpyH2DScn", 1);
	WrapTim("CopyFrameToGPUMem");

	InitTim();
	InitKerTim(1);
	convertChar4ToFloatDoConGam << <gdBlocksConv, gdThreadsConv >> > (gd_ac4Scn, (float4*)d_afPadScnIn, (giScnSz / 4), bConGam);
	WrapKerTim("ConvertScn", 1);
	WrapTim("convertChar4ToFloatDoConGam");

	if (bFixDead)
	{
		InitTim(); 
		InitKerTim(1);
		fixDeadPixels << <gdBlocksDead, gdThreadsDead >> > ((cufftReal*)d_afPadScnIn, giScnSz, giScnW, giScnH);
		WrapKerTim("FixScn", 1);
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



#ifdef RUNCLAHE

	//IMPLEMENT THIS SECTION ON GPU: only for testing CLAHE it is running on the CPU
	cufftReal* h_afScnClahe = (cufftReal*)malloc(giScnMemSzReal);
	CUDA_SAFE_CALL(cudaMemcpy(h_afScnClahe, d_afPadScnIn, giScnMemSzReal, cudaMemcpyDeviceToHost));
	unsigned char* acScnClahe = (unsigned char*)malloc(giScnMemSzUChar);
	for (int i = 0; i < giScnSz; i++)
		{acScnClahe[i] = (unsigned char)h_afScnClahe[i];}
	//convert to unsigned int
	CLAHE(acScnClahe, giScnW, giScnH, 0, 255, giScnW / 8, giScnH / 8, 256, 0.3f); //80 60, 80 30
																				  //copy scene to device
	CUDA_SAFE_CALL(cudaMemcpy(gd_ac4Scn, acScnClahe, giScnMemSzUChar, cudaMemcpyHostToDevice));
	convertChar4ToFloatDoConGam << <gdBlocksConv, gdThreadsConv >> >(gd_ac4Scn, (float4*)d_afPadScnIn, (giScnSz / 4), bConGam);
	free(h_afScnClahe);
	//free(h_afScnOut);
	free(acScnClahe);

	////////////////////
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
			printf("\n      System concludes that speed limit is %3d %s! (Total votes: %6.2f)\n\n", gstCompFlt.aiTpl_no[iMaxTplIndx], acMeasure, fMaxAccConf);
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
#endif
}


////////////////////////////////////////////////////////////////////////////////
// Member Functions
////////////////////////////////////////////////////////////////////////////////
void ssd_fft_gpu_init()
{

	//read comp filters
	gstCompFlt = readCompFlt();//can't read
	giTplH = gstCompFlt.iH;//
	giTplW = gstCompFlt.iW;//
	giTplSz = giTplH * giTplW;
	giTplWMemSz = giTplW * sizeof(cufftReal);
	giTplMemSzReal = giTplH * giTplW * sizeof(cufftReal);
	giTplMemSzCmplx = giTplH * giTplW * sizeof(cufftComplex);
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
	giWholeMaxGDx = (giScnSz) % (BLOCKDIMX_MAX*EACHTHREADREADS) > 0 ? ((giScnSz) / (BLOCKDIMX_MAX*EACHTHREADREADS)) + 1 : (giScnSz) / (BLOCKDIMX_MAX*EACHTHREADREADS);
	if ((giWholeMaxGDx / EACHTHREADREADS) > BLOCKDIMX_MAX)
	{
		//in the second pass each thread will read EACHTHREADREADS blockmaxs. There is giWholeMaxGDx blocks at most.
		//if giWholeMaxGDx/EACHTHREADREADS > BLOCKDIMX_MAX this means that second pass should have more than one block.
		//but it should have only one!
		printf("Error: Each thread in max kernel should read more than %d elements!\n", EACHTHREADREADS);
		exit(0);
	}

#ifdef PINNED_MEM
	CUDA_SAFE_CALL(cudaMallocHost((void**)&gh_acScn, giOrigScnMemSzUChar));
	CUDA_SAFE_CALL(cudaMallocHost((void**)&gh_afArea, giAreaMemSzReal));
#else
	gh_acScn = (unsigned char *)malloc(giOrigScnMemSzUChar);
	gh_afArea = (cufftReal *)malloc(giAreaMemSzReal);
#endif
	CUDA_SAFE_CALL(cudaMalloc((void **)&gd_ac4Scn, giScnMemSzUChar));
	CUDA_SAFE_CALL(cudaMalloc((void **)&gd_afCompFlt, gstCompFlt.iDataMemSz));
	CUDA_SAFE_CALL(cudaMalloc((void **)&gd_afPadScnIn, giScnMemSzReal));
	CUDA_SAFE_CALL(cudaMalloc((void **)&gd_afCorr, giScnMemSzReal));
	CUDA_SAFE_CALL(cudaMalloc((void **)&gd_afPadScnOut, giScnMemSzCmplx));
	CUDA_SAFE_CALL(cudaMalloc((void **)&gd_afMul, giScnMemSzCmplx));
	CUDA_SAFE_CALL(cudaMalloc((void **)&gd_afScnPartIn, giTplMemSzReal));
	CUDA_SAFE_CALL(cudaMalloc((void **)&gd_afScnPartOut, giTplMemSzCmplx));
	CUDA_SAFE_CALL(cudaMalloc((void**)&gd_pfMax, sizeof(cufftReal)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&gd_afBlockMaxs, sizeof(cufftReal)*giWholeMaxGDx));
	CUDA_SAFE_CALL(cudaMalloc((void**)&gd_piMaxIdx, sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&gd_aiBlockMaxIdxs, sizeof(int)*giWholeMaxGDx));

	gd_afWholeTplFFT = NULL;
	gd_afPartTplFFT = NULL;
	gd_afPadTplIn = NULL;
	gd_afPadTplOut = NULL;


	//calculate the block and grid size (both will be 1D) to be used by kernels
	getKernelDims(BLOCKDIMX, giScnSz / 4, &gdThreadsConv, &gdBlocksConv);
	getKernelDims(BLOCKDIMX, giScnSz / 2, &gdThreadsDead, &gdBlocksDead);
	gdThreadsDead.x = gdThreadsDead.x + (HALFWARP + 1);
	getKernelDims(BLOCKDIMX, giScnSz, &gdThreadsWhole, &gdBlocksWhole);
	getKernelDims(BLOCKDIMX, giTplSz, &gdThreadsPart, &gdBlocksPart);

	//Creating FFT plan for whole scene
	CUFFT_SAFE_CALL(cufftPlan2d(&ghFFTplanWholeFwd, giScnH, giScnW, CUFFT_R2C));
	CUFFT_SAFE_CALL(cufftPlan2d(&ghFFTplanWholeInv, giScnH, giScnW, CUFFT_C2R));
	//Creating FFT plan for part of the scene
	CUFFT_SAFE_CALL(cufftPlan2d(&ghFFTplanPartFwd, giTplH, giTplW, CUFFT_R2C));
	CUFFT_SAFE_CALL(cufftPlan2d(&ghFFTplanPartInv, giTplH, giTplW, CUFFT_C2R));

	sdkCreateTimer(&guiParTim);
	sdkCreateTimer(&guiKerTim);

	InitTim();
	//copy all Composite Filters to device memory (copying device to device would take less time)
	CUDA_SAFE_CALL(cudaMemcpy(gd_afCompFlt, gstCompFlt.h_afData, gstCompFlt.iDataMemSz, cudaMemcpyHostToDevice));
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
	PrepTplFFT(gd_afCompFlt, &gd_afPadTplIn, &gd_afPadTplOut, &gd_afWholeTplFFT, &gd_afPartTplFFT, ghFFTplanWholeFwd, ghFFTplanPartFwd);
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

	*piMaxPeakIndx = -1;
	*piPartW = -1;
	*piPartH = -1;


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
	bool bFixDead = gbFixDead;
	if (bLoadScn) bFixDead = 0;

	////////FIRST PASS///////////

	CpyScnToDevAndPreProcess(acScn, gd_afPadScnIn, gbConGam, bFixDead);

	//Running the correlation...
	InitTim();
	//take the FFT of the scene
	InitKerTim(2);
	CUFFT_SAFE_CALL(cufftExecR2C(ghFFTplanWholeFwd, (cufftReal *)gd_afPadScnIn, (cufftComplex *)gd_afPadScnOut));
	WrapKerTim("wholeFFT", 2);
	//apply kth law to scene
	InitKerTim(2);
	kthLaw << <gdBlocksWhole, gdThreadsWhole >> >(gd_afPadScnOut, giScnSz);
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
				getWholeTplFFT(gd_afCompFlt, iIPIndx, iSzIndx, iFltIndx, gd_afPadTplIn, &gd_afPadTplOut, ghFFTplanWholeFwd, gd_afWholeTplFFT);
				//perform correlation
				Corr(gd_afPadTplOut, gdBlocksWhole, gdThreadsWhole, gd_afPadScnOut, giScnSz, gd_afMul, ghFFTplanWholeInv, gd_afCorr, gh_afArea, &iPeakIndx, &fPSR, giScnW, giScnH, false);
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

	////////SECOND PASS///////////
	InitTim();
	//we know the max IP and Sz. Now try different templates
	//copy template-size portion of the scene starting at peak point
	//	CUDA_SAFE_CALL( cudaMemcpy2D( gd_afScnPartIn, giTplWMemSz, gd_afPadScnIn+iMaxPeakIndx, giScnW*sizeof(cufftReal), giTplWMemSz, giTplH , cudaMemcpyDeviceToDevice ));
	getCopyWidthHeight(*piMaxPeakIndx, piPartW, piPartH);
	iPartWMemSz = *piPartW * sizeof(cufftReal);
	//make sure you initialize gd_afScnPartIn with zeros before processing each frame (if we are out of bounds, we will have a part image padded with zeros)
	InitKerTim(3);
	CUDA_SAFE_CALL(cudaMemset(gd_afScnPartIn, 0, giTplMemSzReal));
	CUDA_SAFE_CALL(cudaMemcpy2D(gd_afScnPartIn, giTplWMemSz, gd_afPadScnIn + *piMaxPeakIndx, giScnW * sizeof(cufftReal), iPartWMemSz, *piPartH, cudaMemcpyDeviceToDevice));
	WrapKerTim("MemcpyD2DPart", 3);
	//take the FFT of the scene
	InitKerTim(3);
	CUFFT_SAFE_CALL(cufftExecR2C(ghFFTplanPartFwd, (cufftReal *)gd_afScnPartIn, (cufftComplex *)gd_afScnPartOut));
	WrapKerTim("partFFT", 3);
	//apply kth law to scene
	InitKerTim(3);
	kthLaw << <gdBlocksPart, gdThreadsPart >> >(gd_afScnPartOut, giTplSz);
	WrapKerTim("partKth", 3);
	fMaxPSR = INT_MIN;
	WrapTim("SecondPassInit");
	InitTim();
	for (iFltIndx = 0; iFltIndx < giNumSngCompFlt; iFltIndx++)
	{
		for (iIPIndx = giBegIdxIPInSecond; iIPIndx < giEndIdxIPInSecond; iIPIndx++)
		{
			getPartTplFFT(gd_afCompFlt, iIPIndx, iMaxSzIndx, iFltIndx, &gd_afPadTplOut, ghFFTplanPartFwd, gd_afPartTplFFT);
			Corr(gd_afPadTplOut, gdBlocksPart, gdThreadsPart, gd_afScnPartOut, giTplSz, gd_afMul, ghFFTplanPartInv, gd_afCorr, gh_afArea, &iPeakIndx, &fPSR, giTplW, giTplH, true);
			if (fPSR > fMaxPSR)
			{
				fMaxPSR = fPSR;
				iMaxFltIndx = iFltIndx;
				iMaxIPIndx = iIPIndx;
			}
		}
	}
	WrapTim("SecondPassLoop");


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
}


void ssd_fft_gpu_exit() {


	printf("Shutting down...\n");
	CUFFT_SAFE_CALL(cufftDestroy(ghFFTplanWholeFwd));
	CUFFT_SAFE_CALL(cufftDestroy(ghFFTplanWholeInv));
	CUFFT_SAFE_CALL(cufftDestroy(ghFFTplanPartFwd));
	CUFFT_SAFE_CALL(cufftDestroy(ghFFTplanPartInv));
	CUDA_SAFE_CALL(cudaFree(gd_ac4Scn));
	CUDA_SAFE_CALL(cudaFree(gd_afPadScnIn));
	CUDA_SAFE_CALL(cudaFree(gd_afScnPartIn));
	CUDA_SAFE_CALL(cudaFree(gd_afScnPartOut));
	CUDA_SAFE_CALL(cudaFree(gd_afCompFlt));
	CUDA_SAFE_CALL(cudaFree(gd_afPadScnOut));
	CUDA_SAFE_CALL(cudaFree(gd_afCorr));
	CUDA_SAFE_CALL(cudaFree(gd_afMul));
	CUDA_SAFE_CALL(cudaFree(gd_pfMax));
	CUDA_SAFE_CALL(cudaFree(gd_afBlockMaxs));
	CUDA_SAFE_CALL(cudaFree(gd_piMaxIdx));
	CUDA_SAFE_CALL(cudaFree(gd_aiBlockMaxIdxs));
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
	free(gstCompFlt.h_afData);
	free(gastAccRes);
#endif
	//CUT_EXIT(argc, argv);
}

