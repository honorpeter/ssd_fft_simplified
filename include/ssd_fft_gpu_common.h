#pragma once
void ssd_fft_gpu_init();

void ssd_fft_gpu_findBestTpl(unsigned char* acScn, int* piMaxPeakIndx, int* piPartW, int* piPartH, int* file_info, unsigned long ulTimeStamp);

void ssd_fft_gpu_returnBestTpl(unsigned char* acScn, int* piMaxPeakIndx, int* piPartW, int* piPartH, int* file_info, unsigned long ulTimeStamp, int* iSLCurFrm, int* iSLResult, char* acClipName);

void ssd_fft_gpu_exit();
//#define US_SIGNS

#ifdef US_SIGNS
#define giScnBegY 0
#define giScnW 640
#define giScnH 390
#define giShowScnW 640
#define giShowScnH 384
#define giOrigScnSz 640 * 390
#define acMeasure " mph"
#else
#define giScnBegY 48
#define giScnW 640
#define giScnH 480
#define giShowScnW 640
#define giShowScnH 300
#define giOrigScnSz 640 * 480
#define acMeasure " km/h"
#endif
/*#ifdef US_SIGNS
const int giScnBegY = 0;
const int giScnW = 640;
const int giScnH = 384;  //actually it is 390, but I make it 384 so that it is divisible by 8 and CLAHE works properly
const int giOrigScnSz = 640 * 390;
char acMeasure[5] = "mph";
#else
const int giScnBegY = 48; //48,0 y coordinate where the window begins (zero based index)
const int giScnW = 640; //window has tha same width as the scn.
const int giScnH = 240; //240,480 window height
const int giOrigScnSz = 640 * 480;
char acMeasure[5] = "km/h";
#endif*/