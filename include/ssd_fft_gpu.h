//parameters
//kth law value
#define FK 0.1

#define PINNED_MEM
//display device info and program initialization steps
//#define DISP_DEV_INIT 
//#define POT_FFT
#define BLOCKDIMX 256
#define HALFWARP 16
#define BLOCKDIMX_MAX 512
//in the max_k each thread reads:
#define EACHTHREADREADS 16

//compare CPU and GPU FFT with using a small test image
//#define CMPTEST
//compare against CPU results (if video is processed it has no effect)
//also if DoIPInSecond is on, it has no effect, turn off to get accurate timing.
//#define CHECKRES
	//#define GENTXTOUTPUT //(default=OFF) generate .txt file of the results. Makes sense when CHECKRES is defined.
//show timing of whole findBestTpl time 
//#define ALLTIM
//Show partial timings. If it is defined all timigs would be inaccurate.
//#define PARTIM
//kernel time, if defined ALLTIM and PARTIM would be inaccurate
//#define KERTIM 
//save tpl ffts
#define SAVEFFT
//try IP rotations in second pass
#define DoIPInSecond
//save the fixed scene in a bin file (to be opened in matlab) to visually see how the dead pixels are fixed.
//makes sense if gbFixDead is on, turn off to get accurate timing.
//#define SAVEFIXEDSCN
//Do majority voting (makes sense when processing video), it should be on to write to the stats files.
#define MAJVOT
//display psr/iprot/sz recognition info for each frame
#define DISP_FRM_RECOG
//copy back the preprocessed scene (fix dead and clahe) to the host memory. Turn off to get accurate timing
//#define COPYBACK
//collect results in stats file 
//#define STATS 
//see the effect of clahe with this switch
#define RUNCLAHE
//approximate realtime environment 
//#define REALTIME
//#define US_SIGNS
//show the box (in GUI or realis) when the PSR pass the trashold, otherwise hide it at the upper left corner
#define SHOWBOX_WHENRECOG
//save the scn in bin file (makes sense when we work in Realis)
//#define SAVESCNBIN
//copyback right after fixing the deadpixels (for generating the DAGM video, we want clahe, but we do not want to show it)
//define it when COPYBACK is not defined! Turn off to get accurate timing...
#define COPYBACKAFTERDEADFIX



