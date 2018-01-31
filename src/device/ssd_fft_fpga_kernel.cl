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

#ifdef US_SIGNS
//dead rows starts at the first row, no need to add +1 in iDataIndx calculation in fixDeadPixels kernel
__constant int iAddOneRow = 0;
#else
//in EU videos, dead rows start at second row, need to add +1 in iDataIndx calculation in fixDeadPixels kernel
__constant int iAddOneRow = 1;
#endif
#define FK 0.1
#define cufftComplex float2
#define cufftReal float
#define BLOCKDIMX 256
#define BLOCKDIMX_MAX 512
#define EACHTHREADREADS 16
#define HALFWARP 16
#define IMUL(a, b) a*b
#define NULL 0
#define LOGNScn 9
#define LOGNTpl 6
__constant unsigned char d_acLUT[256] = { 0 }; // currently can't initial from cuda 

#include "fft_8.cl" 
#define LOGPOINTS 3
#define POINTS (1 << LOGPOINTS)

											   // Declare channels for kernel to kernel communication

#pragma OPENCL EXTENSION cl_altera_channels : enable

channel float2 chan00 __attribute__((depth(0)));
channel float2 chan01 __attribute__((depth(0)));
channel float2 chan02 __attribute__((depth(0)));
channel float2 chan03 __attribute__((depth(0)));

channel float2 chan04 __attribute__((depth(0)));
channel float2 chan05 __attribute__((depth(0)));
channel float2 chan06 __attribute__((depth(0)));
channel float2 chan07 __attribute__((depth(0)));


channel float2 chanin00 __attribute__((depth(0)));
channel float2 chanin01 __attribute__((depth(0)));
channel float2 chanin02 __attribute__((depth(0)));
channel float2 chanin03 __attribute__((depth(0)));

channel float2 chanin04 __attribute__((depth(0)));
channel float2 chanin05 __attribute__((depth(0)));
channel float2 chanin06 __attribute__((depth(0)));
channel float2 chanin07 __attribute__((depth(0)));

channel float2 chan10 __attribute__((depth(0)));
channel float2 chan11 __attribute__((depth(0)));
channel float2 chan12 __attribute__((depth(0)));
channel float2 chan13 __attribute__((depth(0)));

channel float2 chan14 __attribute__((depth(0)));
channel float2 chan15 __attribute__((depth(0)));
channel float2 chan16 __attribute__((depth(0)));
channel float2 chan17 __attribute__((depth(0)));


channel float2 chanin10 __attribute__((depth(0)));
channel float2 chanin11 __attribute__((depth(0)));
channel float2 chanin12 __attribute__((depth(0)));
channel float2 chanin13 __attribute__((depth(0)));

channel float2 chanin14 __attribute__((depth(0)));
channel float2 chanin15 __attribute__((depth(0)));
channel float2 chanin16 __attribute__((depth(0)));
channel float2 chanin17 __attribute__((depth(0)));


// This utility function bit-reverses an integer 'x' of width 'bits'.

int bit_reversed(int x, int bits) {
	int y = 0;
#pragma unroll 
	for (int i = 0; i < bits; i++) {
		y <<= 1;
		y |= x & 1;
		x >>= 1;
	}
	return y;
}

/* Accesses to DDR memory are efficient if the memory locations are accessed
* in order. There is significant overhead when accesses are not in order.
* The penalty is higher if the accesses stride a large number of locations.
*
* This function provides the mapping for an alternative memory layout. This
* layout preserves some amount of linearity when accessing elements from the
* same matrix row, while bringing closer locations from the same matrix
* column. The matrix offsets are represented using 2 * log(N) bits. This
* function swaps bits log(N) - 1 ... log(N) / 2 with bits
* log(N) + log(N) / 2 - 1 ... log(N).
*
* The end result is that 2^(N/2) locations from the same row would still be
* consecutive in memory, while the distance between locations from the same
* column would be only 2^(N/2)
*/

int mangle_bits(int x) {
	const int NB = LOGNScn / 2;
	int a95 = x & (((1 << NB) - 1) << NB);
	int a1410 = x & (((1 << NB) - 1) << (2 * NB));
	int mask = ((1 << (2 * NB)) - 1) << NB;
	a95 = a95 << NB;
	a1410 = a1410 >> NB;
	return (x & ~mask) | a95 | a1410;
}

/* This kernel reads the matrix data and provides 8 parallel streams to the
* FFT engine. Each workgroup reads 8 matrix rows to local memory. Once this
* data has been buffered, the workgroup produces 8 streams from strided
* locations in local memory, according to the requirements of the FFT engine.
*/
__attribute__((reqd_work_group_size((1 << LOGNScn), 1, 1)))
kernel void fetch0(global float2 * restrict src, global float2 * restrict src2, int mangle, int inverse, int pass) {
	const int N = (1 << LOGNScn);

	// Local memory for storing 8 rows
	local float2 buf[8 * N];
	local float2 buf2[8];

	float2x8 data;
	// Each read fetches 8 matrix points
	int x = get_global_id(0) << LOGPOINTS;

	/* When using the alternative memory layout, each row consists of a set of
	* segments placed far apart in memory. Instead of reading all segments from
	* one row in order, read one segment from each row before switching to the
	*  next segment. This requires swapping bits log(N) + 2 ... log(N) with
	*  bits log(N) / 2 + 2 ... log(N) / 2 in the offset.
	*/

	int inrow, incol, where, where_global;
	if (mangle) {
		const int NB = LOGNScn / 2;
		int a1210 = x & ((POINTS - 1) << (2 * NB));
		int a75 = x & ((POINTS - 1) << NB);
		int mask = ((POINTS - 1) << NB) | ((POINTS - 1) << (2 * NB));
		a1210 >>= NB;
		a75 <<= NB;
		where = (x & ~mask) | a1210 | a75;
		where_global = mangle_bits(where);
	}
	else {
		where = x;
		where_global = where;
	}
	int gid = get_global_id(0);
	int lid = get_local_id(0);
	if (inverse&&lid < POINTS)
	{
		buf2[lid] = src2[gid / N*POINTS + lid];
	}

	if (x < N*(N + 1))
	{
		buf[(where & ((1 << (LOGNScn + LOGPOINTS)) - 1))] = src[where_global];
		buf[(where & ((1 << (LOGNScn + LOGPOINTS)) - 1)) + 1] = src[where_global + 1];
		buf[(where & ((1 << (LOGNScn + LOGPOINTS)) - 1)) + 2] = src[where_global + 2];
		buf[(where & ((1 << (LOGNScn + LOGPOINTS)) - 1)) + 3] = src[where_global + 3];
		buf[(where & ((1 << (LOGNScn + LOGPOINTS)) - 1)) + 4] = src[where_global + 4];
		buf[(where & ((1 << (LOGNScn + LOGPOINTS)) - 1)) + 5] = src[where_global + 5];
		buf[(where & ((1 << (LOGNScn + LOGPOINTS)) - 1)) + 6] = src[where_global + 6];
		buf[(where & ((1 << (LOGNScn + LOGPOINTS)) - 1)) + 7] = src[where_global + 7];
	}
	//for (int i = 0; i < 8; i++)
	// printf("fetch=(%.1f,%.1f)\n", src[where_global + i].x, src[where_global + i].y);
	barrier(CLK_LOCAL_MEM_FENCE);

	int row = get_local_id(0) >> (LOGNScn - LOGPOINTS);
	int col = get_local_id(0) & (N / POINTS - 1);

	// Stream fetched data over 8 channels to the FFT engine
	/*if (x < N*(N + 1)) {
	write_channel_intel(chanin0, buf[row * N + col]);
	write_channel_intel(chanin1, buf[row * N + 4 * N / 8 + col]);
	write_channel_intel(chanin2, buf[row * N + 2 * N / 8 + col]);
	write_channel_intel(chanin3, buf[row * N + 6 * N / 8 + col]);
	write_channel_intel(chanin4, buf[row * N + N / 8 + col]);
	write_channel_intel(chanin5, buf[row * N + 5 * N / 8 + col]);
	write_channel_intel(chanin6, buf[row * N + 3 * N / 8 + col]);
	write_channel_intel(chanin7, buf[row * N + 7 * N / 8 + col]);
	}*/

	float2 write[8];
	if (inverse&& pass) {
		int pos;
		pos = col;
		//for(int i=0;i<16;i++)
		// printf("IA=%f", IA[8]);
		if (col == 0) {
			write[0].x = buf[row * N + pos].x*IA[2 * pos] - buf[row * N + pos].y*IA[2 * pos + 1] + buf2[row].x*IB[2 * pos] + buf2[row].y*IB[2 * pos + 1];
			write[0].y = buf[row * N + pos].y*IA[2 * pos] + buf[row * N + pos].x*IA[2 * pos + 1] + buf2[row].x*IB[2 * pos + 1] - buf2[row].y*IB[2 * pos];
		}

		else {
			write[0].x = buf[row * N + pos].x*IA[2 * pos] - buf[row * N + pos].y*IA[2 * pos + 1] + buf[row * N + N - pos].x*IB[2 * pos] + buf[row * N + N - pos].y*IB[2 * pos + 1];
			write[0].y = buf[row * N + pos].y*IA[2 * pos] + buf[row * N + pos].x*IA[2 * pos + 1] + buf[row * N + N - pos].x*IB[2 * pos + 1] - buf[row * N + N - pos].y*IB[2 * pos];
		}

		//  printf("write0[pos=%d](buf[%d].x=%f)(%f,%f)\n",pos, row * N + pos, buf[row * N + pos].x, write.x, write.y);
		//write_channel_intel(chanin0, /*buf[row * N + col]*/write[0]);
		pos = col + 4 * N / 8;
		write[1].x = buf[row * N + pos].x*IA[2 * pos] - buf[row * N + pos].y*IA[2 * pos + 1] + buf[row * N + N - pos].x*IB[2 * pos] + buf[row * N + N - pos].y*IB[2 * pos + 1];
		write[1].y = buf[row * N + pos].y*IA[2 * pos] + buf[row * N + pos].x*IA[2 * pos + 1] + buf[row * N + N - pos].x*IB[2 * pos + 1] - buf[row * N + N - pos].y*IB[2 * pos];
		//  printf("write1[pos=%d](buf[%d].x=%f)write.x=%fx%f-%fx%f+%fx%f+%fx%f\n", pos, row * N + pos, buf[row * N + pos].x, buf[row * N + pos].x,IA[2 * pos] , buf[row * N + pos].y,IA[2 * pos + 1] , buf[row * N + N - pos].x,IB[2 * pos] , buf[row * N + N - pos].y,IB[2 * pos + 1]);
		//write_channel_intel(chanin1, /*buf[row * N + 4 * N / 8 + col] */ write[1]);
		pos = col + 2 * N / 8;
		write[2].x = buf[row * N + pos].x*IA[2 * pos] - buf[row * N + pos].y*IA[2 * pos + 1] + buf[row * N + N - pos].x*IB[2 * pos] + buf[row * N + N - pos].y*IB[2 * pos + 1];
		write[2].y = buf[row * N + pos].y*IA[2 * pos] + buf[row * N + pos].x*IA[2 * pos + 1] + buf[row * N + N - pos].x*IB[2 * pos + 1] - buf[row * N + N - pos].y*IB[2 * pos];
		//  printf("write2[pos=%d](buf[%d].x=%f)write.x=%fx%f-%fx%f+%fx%f+%fx%f=(%f,%f)\n", pos, row * N + pos, buf[row * N + pos].x, buf[row * N + pos].x, IA[2 * pos], buf[row * N + pos].y, IA[2 * pos + 1], buf[row * N + N - pos].x, IB[2 * pos], buf[row * N + N - pos].y, IB[2 * pos + 1], write.x, write.y);
		//write_channel_intel(chanin2, /*buf[row * N + 2 * N / 8 + col] */ write[2]);
		pos = col + 6 * N / 8;
		write[3].x = buf[row * N + pos].x*IA[2 * pos] - buf[row * N + pos].y*IA[2 * pos + 1] + buf[row * N + N - pos].x*IB[2 * pos] + buf[row * N + N - pos].y*IB[2 * pos + 1];
		write[3].y = buf[row * N + pos].y*IA[2 * pos] + buf[row * N + pos].x*IA[2 * pos + 1] + buf[row * N + N - pos].x*IB[2 * pos + 1] - buf[row * N + N - pos].y*IB[2 * pos];
		// printf("write3[pos=%d](buf[%d].x=%f)(%f,%f)\n", pos, row * N + pos, buf[row * N + pos].x, write.x, write.y);
		//write_channel_intel(chanin3, /*buf[row * N + 6 * N / 8 + col] */ write[3]);
		pos = col + N / 8;
		write[4].x = buf[row * N + pos].x*IA[2 * pos] - buf[row * N + pos].y*IA[2 * pos + 1] + buf[row * N + N - pos].x*IB[2 * pos] + buf[row * N + N - pos].y*IB[2 * pos + 1];
		write[4].y = buf[row * N + pos].y*IA[2 * pos] + buf[row * N + pos].x*IA[2 * pos + 1] + buf[row * N + N - pos].x*IB[2 * pos + 1] - buf[row * N + N - pos].y*IB[2 * pos];
		// printf("write4[pos=%d](buf[%d].x=%f)(%f,%f)\n", pos, row * N + pos, buf[row * N + pos].x, write.x, write.y);
		//write_channel_intel(chanin4, /*buf[row * N + N / 8 + col] */ write[4]);
		pos = col + 5 * N / 8;
		write[5].x = buf[row * N + pos].x*IA[2 * pos] - buf[row * N + pos].y*IA[2 * pos + 1] + buf[row * N + N - pos].x*IB[2 * pos] + buf[row * N + N - pos].y*IB[2 * pos + 1];
		write[5].y = buf[row * N + pos].y*IA[2 * pos] + buf[row * N + pos].x*IA[2 * pos + 1] + buf[row * N + N - pos].x*IB[2 * pos + 1] - buf[row * N + N - pos].y*IB[2 * pos];
		// printf("write5[pos=%d](buf[%d].x=%f)(%f,%f)\n", pos, row * N + pos, buf[row * N + pos].x, write.x, write.y);
		//write_channel_intel(chanin5, /*buf[row * N + 5 * N / 8 + col] */ write[5]);
		pos = col + 3 * N / 8;
		write[6].x = buf[row * N + pos].x*IA[2 * pos] - buf[row * N + pos].y*IA[2 * pos + 1] + buf[row * N + N - pos].x*IB[2 * pos] + buf[row * N + N - pos].y*IB[2 * pos + 1];
		write[6].y = buf[row * N + pos].y*IA[2 * pos] + buf[row * N + pos].x*IA[2 * pos + 1] + buf[row * N + N - pos].x*IB[2 * pos + 1] - buf[row * N + N - pos].y*IB[2 * pos];
		// printf("write6[pos=%d](buf[%d].x=%f)(%f,%f)\n", pos, row * N + pos, buf[row * N + pos].x, write.x, write.y);
		//write_channel_intel(chanin6, /*buf[row * N + 3 * N / 8 + col] */ write[6]);
		pos = col + 7 * N / 8;
		write[7].x = buf[row * N + pos].x*IA[2 * pos] - buf[row * N + pos].y*IA[2 * pos + 1] + buf[row * N + N - pos].x*IB[2 * pos] + buf[row * N + N - pos].y*IB[2 * pos + 1];
		write[7].y = buf[row * N + pos].y*IA[2 * pos] + buf[row * N + pos].x*IA[2 * pos + 1] + buf[row * N + N - pos].x*IB[2 * pos + 1] - buf[row * N + N - pos].y*IB[2 * pos];
		// printf("write7[pos=%d](buf[%d].x=%f)(%f,%f)\n", pos, row * N + pos, buf[row * N + pos].x, write.x, write.y);
		//write_channel_intel(chanin7, /*buf[row * N + 7 * N / 8 + col] */ write[7]);
	}
	else //if (x < N*(N + 1))
	{
		write[0] = buf[row * N + col];
		write[1] = buf[row * N + 4 * N / 8 + col];
		write[2] = buf[row * N + 2 * N / 8 + col];
		write[3] = buf[row * N + 6 * N / 8 + col];
		write[4] = buf[row * N + N / 8 + col];
		write[5] = buf[row * N + 5 * N / 8 + col];
		write[6] = buf[row * N + 3 * N / 8 + col];
		write[7] = buf[row * N + 7 * N / 8 + col];
	}
	if (x < N*(N + 1)) {
		write_channel_intel(chanin00, write[0]);
		write_channel_intel(chanin01, write[1]);
		write_channel_intel(chanin02, write[2]);
		write_channel_intel(chanin03, write[3]);
		write_channel_intel(chanin04, write[4]);
		write_channel_intel(chanin05, write[5]);
		write_channel_intel(chanin06, write[6]);
		write_channel_intel(chanin07, write[7]);
	}
}

/* This single work-item task wraps the FFT engine
* 'inverse' toggles between the direct and the inverse transform
*/
kernel void fft2d0(int inverse, int pass) {
	const int N = (1 << LOGNScn);

	/* The FFT engine requires a sliding window for data reordering; data stored
	* in this array is carried across loop iterations and shifted by 1 element
	* every iteration; all loop dependencies derived from the uses of this
	* array are simple transfers between adjacent array elements
	*/

	float2 fft_delay_elements[N + POINTS * (LOGNScn - 2)];

	// needs to run "N / 8 - 1" additional iterations to drain the last outputs
	int iterations;
	if (!pass)
		iterations = N * (N / POINTS);
	else
		iterations = (N + 1) * (N / POINTS);


	// needs to run "N / 8 - 1" additional iterations to drain the last outputs
	for (unsigned i = 0; i < iterations + N / POINTS - 1; i++) {
		float2x8 data;
		// Read data from channels
		if (i < iterations) {
			data.i0 = read_channel_intel(chanin00);
			data.i1 = read_channel_intel(chanin01);
			data.i2 = read_channel_intel(chanin02);
			data.i3 = read_channel_intel(chanin03);
			data.i4 = read_channel_intel(chanin04);
			data.i5 = read_channel_intel(chanin05);
			data.i6 = read_channel_intel(chanin06);
			data.i7 = read_channel_intel(chanin07);
		}
		else {
			data.i0 = data.i1 = data.i2 = data.i3 =
				data.i4 = data.i5 = data.i6 = data.i7 = 0;
		}

		// Perform one FFT step
		data = fft_step(data, i % (N / POINTS), fft_delay_elements, inverse, LOGNScn);

		// Write result to channels
		if (i >= N / POINTS - 1) {
			/*if (inverse) {
			data.i0 = data.i0 / N;
			data.i1 = data.i1 / N;
			data.i2 = data.i2 / N;
			data.i3 = data.i3 / N;
			data.i4 = data.i4 / N;
			data.i5 = data.i5 / N;
			data.i6 = data.i6 / N;
			data.i7 = data.i7 / N;
			}*/
			write_channel_intel(chan00, data.i0);
			write_channel_intel(chan01, data.i1);
			write_channel_intel(chan02, data.i2);
			write_channel_intel(chan03, data.i3);
			write_channel_intel(chan04, data.i4);
			write_channel_intel(chan05, data.i5);
			write_channel_intel(chan06, data.i6);
			write_channel_intel(chan07, data.i7);
			//printf("write=(%f,%f)\n", data.i7.x, data.i7.y);
		}
	}
}

/* This kernel receives the FFT results, buffers 8 rows and then writes the
* results transposed in memory. Because 8 rows are buffered, 8 consecutive
* columns can be written at a time on each transposed row. This provides some
* degree of locality. In addition, when using the alternative matrix format,
* consecutive rows are closer in memory, and this is also beneficial for
* higher memory access efficiency
*/
channel float2 chanEx00 __attribute__((depth(0)));
channel float2 chanEx01 __attribute__((depth(0)));
__attribute__((reqd_work_group_size((1 << LOGNScn), 1, 1)))
kernel void transpose0(global float2 * restrict dest, int mangle, int inverse, int pass) {
	const int N = (1 << LOGNScn);
	local float2 buf[POINTS * N];
	local float2 buf2[POINTS * N];
	if (get_global_id(0) < N*(N + 1) / POINTS) {
		buf[8 * get_local_id(0)] = read_channel_intel(chan00);
		buf[8 * get_local_id(0) + 1] = read_channel_intel(chan01);
		buf[8 * get_local_id(0) + 2] = read_channel_intel(chan02);
		buf[8 * get_local_id(0) + 3] = read_channel_intel(chan03);
		buf[8 * get_local_id(0) + 4] = read_channel_intel(chan04);
		buf[8 * get_local_id(0) + 5] = read_channel_intel(chan05);
		buf[8 * get_local_id(0) + 6] = read_channel_intel(chan06);
		buf[8 * get_local_id(0) + 7] = read_channel_intel(chan07);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	/*int lid = get_local_id(0);
	int row = lid / (N / 8);
	if (!pass) {
	#pragma unroll
	for (int i = 0; i < 8; i++) {
	int k = lid % (N / 8) * 8 + i;
	int rk = bit_reversed(N - bit_reversed(k, LOGN), LOGN);
	//don't need to specify X(N/2) = X(0) bc bit_reversed already did it
	//if (!inverse) {
	buf2[row*(N)+k].x = buf[row*N + k].x*A[2 * k] - buf[row*N + k].y*A[2 * k + 1] + buf[row*N + rk].x*B[2 * k] + buf[row*N + rk].y*B[2 * k + 1];
	buf2[row*(N)+k].y = buf[row*N + k].y*A[2 * k] + buf[row*N + k].x*A[2 * k + 1] + buf[row*N + rk].x*B[2 * k + 1] - buf[row*N + rk].y*B[2 * k];
	// }
	//else {
	//buf2[row*(N)+k].x = buf[row*N + k].x*A[2 * k] + buf[row*N + k].y*A[2 * k + 1] + buf[row*N + rk].x*B[2 * k] - buf[row*N + rk].y*B[2 * k + 1];
	//buf2[row*(N)+k].y = buf[row*N + k].y*A[2 * k] - buf[row*N + k].x*A[2 * k + 1] - buf[row*N + rk].x*B[2 * k + 1] - buf[row*N + rk].y*B[2 * k];
	//}
	//buf2[row*(N+1)+N].x = buf[row*N].x - buf[row*N].y;
	//buf2[row*(N+1)+N].y = 0;
	//buf2[row*(N) + k].x = buf[row*N + k].x*A[2 * k] - buf[row*N + k].y*A[2 * k + 1] + buf[row*N + rk].x*B[2 * k] + buf[row*N + rk].y*B[2 * k + 1];
	//buf2[row*(N) + k].y = buf[row*N + k].y*A[2 * k] + buf[row*N + k].x*A[2 * k + 1] + buf[row*N + rk].x*B[2 * k + 1] - buf[row*N + rk].y*B[2 * k];
	}
	}
	barrier(CLK_LOCAL_MEM_FENCE);*/
	//for (int i= 0; i < 8; i++) {
	//	int k = bit_reversed((get_local_id(0) % (N / 8))*8+i, LOGN );
	//printf("%d=(%.1f,%.1f)",k+ get_local_id(0) / (N / 8)*(N), buf2[k + get_local_id(0) / (N / 8)*(N)].x, buf2[k + get_local_id(0) / (N / 8)*(N)].y);
	// }
	//printf("%d=(%.1f,%.1f)", get_local_id(0) / (N / 8)*(N + )+ (N ), buf2[get_local_id(0) / (N / 8)*(N ) + (N + 1)].x, buf2[get_local_id(0) / (N / 8)*(N + 1) + (N + 1)].y);
	//printf("\n");
	int colt = get_local_id(0);
	int revcolt = bit_reversed(colt, LOGNScn);
	int i = get_global_id(0) >> LOGNScn;
	int where = colt * N + i * POINTS;
	if (mangle) where = mangle_bits(where);
	if (!pass && !inverse) {
		/*dest[where] = buf2[revcolt];
		dest[where + 1] = buf2[N + revcolt];
		dest[where + 2] = buf2[2 * N + revcolt];
		dest[where + 3] = buf2[3 * N + revcolt];
		dest[where + 4] = buf2[4 * N + revcolt];
		dest[where + 5] = buf2[5 * N + revcolt];
		dest[where + 6] = buf2[6 * N + revcolt];
		dest[where + 7] = buf2[7 * N + revcolt];*/
		int rerevcolt = bit_reversed(N - colt, LOGNScn);

		dest[where].x = buf[revcolt].x*A[2 * revcolt] - buf[revcolt].y*A[2 * revcolt + 1] + buf[rerevcolt].x*B[2 * revcolt] + buf[rerevcolt].y*B[2 * revcolt + 1];
		dest[where].y = buf[revcolt].y*A[2 * revcolt] + buf[revcolt].x*A[2 * revcolt + 1] + buf[rerevcolt].x*B[2 * revcolt + 1] - buf[rerevcolt].y*B[2 * revcolt];
		dest[where + 1].x = buf[N + revcolt].x*A[2 * revcolt] - buf[N + revcolt].y*A[2 * revcolt + 1] + buf[N + rerevcolt].x*B[2 * revcolt] + buf[N + rerevcolt].y*B[2 * revcolt + 1];
		dest[where + 1].y = buf[N + revcolt].y*A[2 * revcolt] + buf[N + revcolt].x*A[2 * revcolt + 1] + buf[N + rerevcolt].x*B[2 * revcolt + 1] - buf[N + rerevcolt].y*B[2 * revcolt];
		dest[where + 2].x = buf[2 * N + revcolt].x*A[2 * revcolt] - buf[2 * N + revcolt].y*A[2 * revcolt + 1] + buf[2 * N + rerevcolt].x*B[2 * revcolt] + buf[2 * N + rerevcolt].y*B[2 * revcolt + 1];
		dest[where + 2].y = buf[2 * N + revcolt].y*A[2 * revcolt] + buf[2 * N + revcolt].x*A[2 * revcolt + 1] + buf[2 * N + rerevcolt].x*B[2 * revcolt + 1] - buf[2 * N + rerevcolt].y*B[2 * revcolt];
		dest[where + 3].x = buf[3 * N + revcolt].x*A[2 * revcolt] - buf[3 * N + revcolt].y*A[2 * revcolt + 1] + buf[3 * N + rerevcolt].x*B[2 * revcolt] + buf[3 * N + rerevcolt].y*B[2 * revcolt + 1];
		dest[where + 3].y = buf[3 * N + revcolt].y*A[2 * revcolt] + buf[3 * N + revcolt].x*A[2 * revcolt + 1] + buf[3 * N + rerevcolt].x*B[2 * revcolt + 1] - buf[3 * N + rerevcolt].y*B[2 * revcolt];
		dest[where + 4].x = buf[4 * N + revcolt].x*A[2 * revcolt] - buf[4 * N + revcolt].y*A[2 * revcolt + 1] + buf[4 * N + rerevcolt].x*B[2 * revcolt] + buf[4 * N + rerevcolt].y*B[2 * revcolt + 1];
		dest[where + 4].y = buf[4 * N + revcolt].y*A[2 * revcolt] + buf[4 * N + revcolt].x*A[2 * revcolt + 1] + buf[4 * N + rerevcolt].x*B[2 * revcolt + 1] - buf[4 * N + rerevcolt].y*B[2 * revcolt];
		dest[where + 5].x = buf[5 * N + revcolt].x*A[2 * revcolt] - buf[5 * N + revcolt].y*A[2 * revcolt + 1] + buf[5 * N + rerevcolt].x*B[2 * revcolt] + buf[5 * N + rerevcolt].y*B[2 * revcolt + 1];
		dest[where + 5].y = buf[5 * N + revcolt].y*A[2 * revcolt] + buf[5 * N + revcolt].x*A[2 * revcolt + 1] + buf[5 * N + rerevcolt].x*B[2 * revcolt + 1] - buf[5 * N + rerevcolt].y*B[2 * revcolt];
		dest[where + 6].x = buf[6 * N + revcolt].x*A[2 * revcolt] - buf[6 * N + revcolt].y*A[2 * revcolt + 1] + buf[6 * N + rerevcolt].x*B[2 * revcolt] + buf[6 * N + rerevcolt].y*B[2 * revcolt + 1];
		dest[where + 6].y = buf[6 * N + revcolt].y*A[2 * revcolt] + buf[6 * N + revcolt].x*A[2 * revcolt + 1] + buf[6 * N + rerevcolt].x*B[2 * revcolt + 1] - buf[6 * N + rerevcolt].y*B[2 * revcolt];
		dest[where + 7].x = buf[7 * N + revcolt].x*A[2 * revcolt] - buf[7 * N + revcolt].y*A[2 * revcolt + 1] + buf[7 * N + rerevcolt].x*B[2 * revcolt] + buf[7 * N + rerevcolt].y*B[2 * revcolt + 1];
		dest[where + 7].y = buf[7 * N + revcolt].y*A[2 * revcolt] + buf[7 * N + revcolt].x*A[2 * revcolt + 1] + buf[7 * N + rerevcolt].x*B[2 * revcolt + 1] - buf[7 * N + rerevcolt].y*B[2 * revcolt];
		/*printf("gid=%d i=%d lid=%d dest[%d]=%f",get_global_id(0),i, colt, where , dest[where ].x);
		printf("dest[%d]=%f",  where+1, dest[where+1].x);
		printf("dest[%d]=%f",  where + 2, dest[where + 2].x);
		printf("dest[%d]=%f\n",  where + 3, dest[where + 3].x);*/

		if (colt >= N - 8) { // there's a difference to if (colt<8)
			float2 tmp;
			tmp.x = buf[N*(colt % 8)].x - buf[N*(colt % 8)].y;
			tmp.y = 0;
			write_channel_intel(chanEx00, tmp);
		}
		//dest2[colt%8+i * POINTS].x= buf[N*colt % 8].x - buf[N*colt % 8].y;
		//dest2[colt % 8 + i * POINTS].y = 0;

	}
	else if (!pass && inverse)
	{
		if (get_global_id(0) < N*N / POINTS) {
			dest[where] = buf[revcolt];
			dest[where + 1] = buf[N + revcolt];
			dest[where + 2] = buf[2 * (N)+revcolt];
			dest[where + 3] = buf[3 * (N)+revcolt];
			dest[where + 4] = buf[4 * (N)+revcolt];
			dest[where + 5] = buf[5 * (N)+revcolt];
			dest[where + 6] = buf[6 * (N)+revcolt];
			dest[where + 7] = buf[7 * (N)+revcolt];
		}
		else if (get_global_id(0) >= N*N / POINTS) {
			//for (int i = 0; i < N; i++)
			//{
			//int revi = bit_reversed(i, LOGN);
			write_channel_intel(chanEx01, buf[revcolt]);
			//printf("ex[colt=%d]=%f", colt, buf[revcolt].x);
			//}
		}
		//dest[where] = buf[revcolt];
	}
	else if (pass&&get_global_id(0) < N*(N + 1) / POINTS)//second pass
	{
		where = colt * 8 + i * POINTS*N;
		dest[where] = buf[colt / (N / 8) * N + bit_reversed(colt % (N / 8) * 8, LOGNScn)];
		dest[where + 1] = buf[colt / (N / 8) * N + bit_reversed(colt % (N / 8) * 8 + 1, LOGNScn)];
		dest[where + 2] = buf[colt / (N / 8) * N + bit_reversed(colt % (N / 8) * 8 + 2, LOGNScn)];
		dest[where + 3] = buf[colt / (N / 8) * N + bit_reversed(colt % (N / 8) * 8 + 3, LOGNScn)];
		dest[where + 4] = buf[colt / (N / 8) * N + bit_reversed(colt % (N / 8) * 8 + 4, LOGNScn)];
		dest[where + 5] = buf[colt / (N / 8) * N + bit_reversed(colt % (N / 8) * 8 + 5, LOGNScn)];
		dest[where + 6] = buf[colt / (N / 8) * N + bit_reversed(colt % (N / 8) * 8 + 6, LOGNScn)];
		dest[where + 7] = buf[colt / (N / 8) * N + bit_reversed(colt % (N / 8) * 8 + 7, LOGNScn)];
	}

}



__attribute__((reqd_work_group_size(8, 1, 1)))
kernel void extraCol0(global float2 * restrict dest2, int inverse)
{
	float2 wirte;
	if (!inverse)
		dest2[get_global_id(0)] = read_channel_intel(chanEx00);
	else
		dest2[get_global_id(0)] = read_channel_intel(chanEx01);


}

////////////////////////////////////////////////////////////////////////////////////
//Template FFT
////////////////////////////////////////////////////////////////////////////////////
__attribute__((reqd_work_group_size((1 << (LOGNTpl-1)), 1, 1)))
kernel void fetch1(global float2 * restrict src, global float2 * restrict src2, int mangle, int inverse, int pass) {
	const int N = (1 << (LOGNTpl - 1));

	// Local memory for storing 8 rows
	local float2 buf[8 * N];
	local float2 buf2[8];

	float2x8 data;
	// Each read fetches 8 matrix points
	int x = get_global_id(0) << LOGPOINTS;

	/* When using the alternative memory layout, each row consists of a set of
	* segments placed far apart in memory. Instead of reading all segments from
	* one row in order, read one segment from each row before switching to the
	*  next segment. This requires swapping bits log(N) + 2 ... log(N) with
	*  bits log(N) / 2 + 2 ... log(N) / 2 in the offset.
	*/

	int inrow, incol, where, where_global;
	where = x;
	where_global = where;

	int gid = get_global_id(0);
	int lid = get_local_id(0);
	if (inverse&&lid < POINTS)
	{
		buf2[lid] = src2[gid / N*POINTS + lid];
	}


	int shif = pass ? ((LOGNTpl - 1) + LOGPOINTS) : ((LOGNTpl - 1) + LOGPOINTS);
	if (x < N * 2 * (N + 1))
	{
		buf[(where & ((1 << (shif)) - 1))] = src[where_global];
		buf[(where & ((1 << (shif)) - 1)) + 1] = src[where_global + 1];
		buf[(where & ((1 << (shif)) - 1)) + 2] = src[where_global + 2];
		buf[(where & ((1 << (shif)) - 1)) + 3] = src[where_global + 3];
		buf[(where & ((1 << (shif)) - 1)) + 4] = src[where_global + 4];
		buf[(where & ((1 << (shif)) - 1)) + 5] = src[where_global + 5];
		buf[(where & ((1 << (shif)) - 1)) + 6] = src[where_global + 6];
		buf[(where & ((1 << (shif)) - 1)) + 7] = src[where_global + 7];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	int row, col, iN;

	if ((pass^inverse) == 0) {
		row = get_local_id(0) >> ((LOGNTpl - 1) - LOGPOINTS);
		col = get_local_id(0) & (N / POINTS - 1);
		iN = N;
	}
	else {
		row = get_local_id(0) >> ((LOGNTpl - 1) - LOGPOINTS + 1);
		col = get_local_id(0) & (N * 2 / POINTS - 1);
		iN = 2 * N;

	}


	float2 write[8];
	if (inverse&& pass) {
		int pos;
		pos = col;
		//for(int i=0;i<16;i++)
		// printf("IAT=%f", IAT[8]);
		if (col == 0) {
			write[0].x = buf[row * N + pos].x*IAT[2 * pos] - buf[row * N + pos].y*IAT[2 * pos + 1] + buf2[row].x*IBT[2 * pos] + buf2[row].y*IBT[2 * pos + 1];
			write[0].y = buf[row * N + pos].y*IAT[2 * pos] + buf[row * N + pos].x*IAT[2 * pos + 1] + buf2[row].x*IBT[2 * pos + 1] - buf2[row].y*IBT[2 * pos];

		}

		else {
			write[0].x = buf[row * N + pos].x*IAT[2 * pos] - buf[row * N + pos].y*IAT[2 * pos + 1] + buf[row * N + N - pos].x*IBT[2 * pos] + buf[row * N + N - pos].y*IBT[2 * pos + 1];
			write[0].y = buf[row * N + pos].y*IAT[2 * pos] + buf[row * N + pos].x*IAT[2 * pos + 1] + buf[row * N + N - pos].x*IBT[2 * pos + 1] - buf[row * N + N - pos].y*IBT[2 * pos];
		}


		pos = col + 4 * N / 8;
		write[1].x = buf[row * N + pos].x*IAT[2 * pos] - buf[row * N + pos].y*IAT[2 * pos + 1] + buf[row * N + N - pos].x*IBT[2 * pos] + buf[row * N + N - pos].y*IBT[2 * pos + 1];
		write[1].y = buf[row * N + pos].y*IAT[2 * pos] + buf[row * N + pos].x*IAT[2 * pos + 1] + buf[row * N + N - pos].x*IBT[2 * pos + 1] - buf[row * N + N - pos].y*IBT[2 * pos];

		pos = col + 2 * N / 8;
		write[2].x = buf[row * N + pos].x*IAT[2 * pos] - buf[row * N + pos].y*IAT[2 * pos + 1] + buf[row * N + N - pos].x*IBT[2 * pos] + buf[row * N + N - pos].y*IBT[2 * pos + 1];
		write[2].y = buf[row * N + pos].y*IAT[2 * pos] + buf[row * N + pos].x*IAT[2 * pos + 1] + buf[row * N + N - pos].x*IBT[2 * pos + 1] - buf[row * N + N - pos].y*IBT[2 * pos];

		pos = col + 6 * N / 8;
		write[3].x = buf[row * N + pos].x*IAT[2 * pos] - buf[row * N + pos].y*IAT[2 * pos + 1] + buf[row * N + N - pos].x*IBT[2 * pos] + buf[row * N + N - pos].y*IBT[2 * pos + 1];
		write[3].y = buf[row * N + pos].y*IAT[2 * pos] + buf[row * N + pos].x*IAT[2 * pos + 1] + buf[row * N + N - pos].x*IBT[2 * pos + 1] - buf[row * N + N - pos].y*IBT[2 * pos];

		pos = col + N / 8;
		write[4].x = buf[row * N + pos].x*IAT[2 * pos] - buf[row * N + pos].y*IAT[2 * pos + 1] + buf[row * N + N - pos].x*IBT[2 * pos] + buf[row * N + N - pos].y*IBT[2 * pos + 1];
		write[4].y = buf[row * N + pos].y*IAT[2 * pos] + buf[row * N + pos].x*IAT[2 * pos + 1] + buf[row * N + N - pos].x*IBT[2 * pos + 1] - buf[row * N + N - pos].y*IBT[2 * pos];

		pos = col + 5 * N / 8;
		write[5].x = buf[row * N + pos].x*IAT[2 * pos] - buf[row * N + pos].y*IAT[2 * pos + 1] + buf[row * N + N - pos].x*IBT[2 * pos] + buf[row * N + N - pos].y*IBT[2 * pos + 1];
		write[5].y = buf[row * N + pos].y*IAT[2 * pos] + buf[row * N + pos].x*IAT[2 * pos + 1] + buf[row * N + N - pos].x*IBT[2 * pos + 1] - buf[row * N + N - pos].y*IBT[2 * pos];

		pos = col + 3 * N / 8;
		write[6].x = buf[row * N + pos].x*IAT[2 * pos] - buf[row * N + pos].y*IAT[2 * pos + 1] + buf[row * N + N - pos].x*IBT[2 * pos] + buf[row * N + N - pos].y*IBT[2 * pos + 1];
		write[6].y = buf[row * N + pos].y*IAT[2 * pos] + buf[row * N + pos].x*IAT[2 * pos + 1] + buf[row * N + N - pos].x*IBT[2 * pos + 1] - buf[row * N + N - pos].y*IBT[2 * pos];

		pos = col + 7 * N / 8;
		write[7].x = buf[row * N + pos].x*IAT[2 * pos] - buf[row * N + pos].y*IAT[2 * pos + 1] + buf[row * N + N - pos].x*IBT[2 * pos] + buf[row * N + N - pos].y*IBT[2 * pos + 1];
		write[7].y = buf[row * N + pos].y*IAT[2 * pos] + buf[row * N + pos].x*IAT[2 * pos + 1] + buf[row * N + N - pos].x*IBT[2 * pos + 1] - buf[row * N + N - pos].y*IBT[2 * pos];

	}
	else if (x < N * 2 * (N + 1)) 
	{
		write[0] = buf[row * iN + col];
		write[1] = buf[row * iN + 4 * iN / 8 + col];
		write[2] = buf[row * iN + 2 * iN / 8 + col];
		write[3] = buf[row * iN + 6 * iN / 8 + col];
		write[4] = buf[row * iN + iN / 8 + col];
		write[5] = buf[row * iN + 5 * iN / 8 + col];
		write[6] = buf[row * iN + 3 * iN / 8 + col];
		write[7] = buf[row * iN + 7 * iN / 8 + col];
		int a = row * iN + col;
	}
 
	if (x < N * 2 * (N + 1))
	{
		write_channel_intel(chanin10, write[0]);
		write_channel_intel(chanin11, write[1]);
		write_channel_intel(chanin12, write[2]);
		write_channel_intel(chanin13, write[3]);
		write_channel_intel(chanin14, write[4]);
		write_channel_intel(chanin15, write[5]);
		write_channel_intel(chanin16, write[6]);
		write_channel_intel(chanin17, write[7]);
	}
}

kernel void fft2d1(int inverse, int pass) {
	const int N = (1 << (LOGNTpl - 1));

	/* The FFT engine requires a sliding window for data reordering; data stored
	* in this array is carried across loop iterations and shifted by 1 element
	* every iteration; all loop dependencies derived from the uses of this
	* array are simple transfers between adjacent array elements
	*/

	float2 fft_delay_elements[2 * N + POINTS * ((LOGNTpl - 1) + 1 - 2)];

	// needs to run "N / 8 - 1" additional iterations to drain the last outputs
	int iterations;
	if (!pass)//here pass= pass^inverse
		iterations = 2 * N * (N / POINTS);
	else
		iterations = 2 * (N + 1) * (N / POINTS);


	int addition = pass ? (N * 2 / POINTS - 1) : (N / POINTS - 1);
	// needs to run "N / 8 - 1" additional iterations to drain the last outputs
	for (unsigned i = 0; i < iterations + addition; i++) {
		float2x8 data;
		// Read data from channels
		if (i < iterations) {
			data.i0 = read_channel_intel(chanin10);
			data.i1 = read_channel_intel(chanin11);
			data.i2 = read_channel_intel(chanin12);
			data.i3 = read_channel_intel(chanin13);
			data.i4 = read_channel_intel(chanin14);
			data.i5 = read_channel_intel(chanin15);
			data.i6 = read_channel_intel(chanin16);
			data.i7 = read_channel_intel(chanin17);
		}
		else {
			data.i0 = data.i1 = data.i2 = data.i3 =
				data.i4 = data.i5 = data.i6 = data.i7 = 0;
		}

		// Perform one FFT step
		if (!pass)
			data = fft_step(data, i % (N / POINTS), fft_delay_elements, inverse, (LOGNTpl - 1));
		else
			data = fft_step(data, i % (2 * N / POINTS), fft_delay_elements, inverse, (LOGNTpl - 1) + 1);// i % (2*N / POINTS)!!
																							   // Write result to channels
		if (i >= addition) {
			/*if (inverse) {
				int d = pass ? 2 * N : N;
				data.i0 = data.i0 / d;
				data.i1 = data.i1 / d;
				data.i2 = data.i2 / d;
				data.i3 = data.i3 / d;
				data.i4 = data.i4 / d;
				data.i5 = data.i5 / d;
				data.i6 = data.i6 / d;
				data.i7 = data.i7 / d;
			}*/
			write_channel_intel(chan10, data.i0);
			write_channel_intel(chan11, data.i1);
			write_channel_intel(chan12, data.i2);
			write_channel_intel(chan13, data.i3);
			write_channel_intel(chan14, data.i4);
			write_channel_intel(chan15, data.i5);
			write_channel_intel(chan16, data.i6);
			write_channel_intel(chan17, data.i7);
		}
	}
}

channel float2 chanEx10 __attribute__((depth(0)));
channel float2 chanEx11 __attribute__((depth(0)));
channel float2 chanEx12 __attribute__((depth(0)));
__attribute__((reqd_work_group_size((1 << (LOGNTpl - 1)), 1, 1)))
kernel void transpose1(global float2 * restrict dest, int mangle, int inverse, int pass) {
	const int N = (1 << (LOGNTpl - 1));
	local float2 buf[POINTS * N];
	local float2 buf2[POINTS * N];
	if (get_global_id(0) < 2 * N*(N + 1) / POINTS) {
		buf[8 * get_local_id(0)] = read_channel_intel(chan10);
		buf[8 * get_local_id(0) + 1] = read_channel_intel(chan11);
		buf[8 * get_local_id(0) + 2] = read_channel_intel(chan12);
		buf[8 * get_local_id(0) + 3] = read_channel_intel(chan13);
		buf[8 * get_local_id(0) + 4] = read_channel_intel(chan14);
		buf[8 * get_local_id(0) + 5] = read_channel_intel(chan15);
		buf[8 * get_local_id(0) + 6] = read_channel_intel(chan16);
		buf[8 * get_local_id(0) + 7] = read_channel_intel(chan17);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	

	int colt = get_local_id(0);
	int revcolt = bit_reversed(colt, (LOGNTpl - 1));
	int i = get_global_id(0) >> (LOGNTpl - 1);
	int where = (colt * 2 * N + i * POINTS);
	if (!pass && !inverse) {
		int rerevcolt = bit_reversed(N - colt, (LOGNTpl - 1));

		dest[where].x = buf[revcolt].x*A[2 * revcolt] - buf[revcolt].y*A[2 * revcolt + 1] + buf[rerevcolt].x*B[2 * revcolt] + buf[rerevcolt].y*B[2 * revcolt + 1];
		dest[where].y = buf[revcolt].y*A[2 * revcolt] + buf[revcolt].x*A[2 * revcolt + 1] + buf[rerevcolt].x*B[2 * revcolt + 1] - buf[rerevcolt].y*B[2 * revcolt];
		dest[where + 1].x = buf[N + revcolt].x*A[2 * revcolt] - buf[N + revcolt].y*A[2 * revcolt + 1] + buf[N + rerevcolt].x*B[2 * revcolt] + buf[N + rerevcolt].y*B[2 * revcolt + 1];
		dest[where + 1].y = buf[N + revcolt].y*A[2 * revcolt] + buf[N + revcolt].x*A[2 * revcolt + 1] + buf[N + rerevcolt].x*B[2 * revcolt + 1] - buf[N + rerevcolt].y*B[2 * revcolt];
		dest[where + 2].x = buf[2 * N + revcolt].x*A[2 * revcolt] - buf[2 * N + revcolt].y*A[2 * revcolt + 1] + buf[2 * N + rerevcolt].x*B[2 * revcolt] + buf[2 * N + rerevcolt].y*B[2 * revcolt + 1];
		dest[where + 2].y = buf[2 * N + revcolt].y*A[2 * revcolt] + buf[2 * N + revcolt].x*A[2 * revcolt + 1] + buf[2 * N + rerevcolt].x*B[2 * revcolt + 1] - buf[2 * N + rerevcolt].y*B[2 * revcolt];
		dest[where + 3].x = buf[3 * N + revcolt].x*A[2 * revcolt] - buf[3 * N + revcolt].y*A[2 * revcolt + 1] + buf[3 * N + rerevcolt].x*B[2 * revcolt] + buf[3 * N + rerevcolt].y*B[2 * revcolt + 1];
		dest[where + 3].y = buf[3 * N + revcolt].y*A[2 * revcolt] + buf[3 * N + revcolt].x*A[2 * revcolt + 1] + buf[3 * N + rerevcolt].x*B[2 * revcolt + 1] - buf[3 * N + rerevcolt].y*B[2 * revcolt];
		dest[where + 4].x = buf[4 * N + revcolt].x*A[2 * revcolt] - buf[4 * N + revcolt].y*A[2 * revcolt + 1] + buf[4 * N + rerevcolt].x*B[2 * revcolt] + buf[4 * N + rerevcolt].y*B[2 * revcolt + 1];
		dest[where + 4].y = buf[4 * N + revcolt].y*A[2 * revcolt] + buf[4 * N + revcolt].x*A[2 * revcolt + 1] + buf[4 * N + rerevcolt].x*B[2 * revcolt + 1] - buf[4 * N + rerevcolt].y*B[2 * revcolt];
		dest[where + 5].x = buf[5 * N + revcolt].x*A[2 * revcolt] - buf[5 * N + revcolt].y*A[2 * revcolt + 1] + buf[5 * N + rerevcolt].x*B[2 * revcolt] + buf[5 * N + rerevcolt].y*B[2 * revcolt + 1];
		dest[where + 5].y = buf[5 * N + revcolt].y*A[2 * revcolt] + buf[5 * N + revcolt].x*A[2 * revcolt + 1] + buf[5 * N + rerevcolt].x*B[2 * revcolt + 1] - buf[5 * N + rerevcolt].y*B[2 * revcolt];
		dest[where + 6].x = buf[6 * N + revcolt].x*A[2 * revcolt] - buf[6 * N + revcolt].y*A[2 * revcolt + 1] + buf[6 * N + rerevcolt].x*B[2 * revcolt] + buf[6 * N + rerevcolt].y*B[2 * revcolt + 1];
		dest[where + 6].y = buf[6 * N + revcolt].y*A[2 * revcolt] + buf[6 * N + revcolt].x*A[2 * revcolt + 1] + buf[6 * N + rerevcolt].x*B[2 * revcolt + 1] - buf[6 * N + rerevcolt].y*B[2 * revcolt];
		dest[where + 7].x = buf[7 * N + revcolt].x*A[2 * revcolt] - buf[7 * N + revcolt].y*A[2 * revcolt + 1] + buf[7 * N + rerevcolt].x*B[2 * revcolt] + buf[7 * N + rerevcolt].y*B[2 * revcolt + 1];
		dest[where + 7].y = buf[7 * N + revcolt].y*A[2 * revcolt] + buf[7 * N + revcolt].x*A[2 * revcolt + 1] + buf[7 * N + rerevcolt].x*B[2 * revcolt + 1] - buf[7 * N + rerevcolt].y*B[2 * revcolt];
		

		if (colt >= N - 8) { // there's a difference to if (colt<8)
			float2 tmp;
			tmp.x = buf[N*(colt % 8)].x - buf[N*(colt % 8)].y;
			tmp.y = 0;
			write_channel_intel(chanEx10, tmp);
		}
		

	}
	else if (!pass && inverse)
	{
		if (get_global_id(0) <2 * N*N / POINTS) {
			where = (colt * 2 * N + i * POINTS / 2);
			int revcolt1 = bit_reversed(colt * 2, (LOGNTpl - 1) + 1);
			int revcolt2 = bit_reversed(colt * 2 + 1, (LOGNTpl - 1) + 1);
			dest[where] = buf[revcolt1];
			dest[where + 1] = buf[2 * N + revcolt1];
			dest[where + 2] = buf[2 * (2 * N) + revcolt1];
			dest[where + 3] = buf[3 * (2 * N) + revcolt1];
			dest[where + N] = buf[revcolt2];
			dest[where + N + 1] = buf[2 * N + revcolt2];
			dest[where + N + 2] = buf[2 * (2 * N) + revcolt2];
			dest[where + N + 3] = buf[3 * (2 * N) + revcolt2];
		}
		else if (get_global_id(0) >= 2 * N*N / POINTS) {

			int revcolt1 = bit_reversed(colt * 2, (LOGNTpl - 1) + 1);
			int revcolt2 = bit_reversed(colt * 2 + 1, (LOGNTpl - 1) + 1);
			write_channel_intel(chanEx11, buf[revcolt1]);
			write_channel_intel(chanEx12, buf[revcolt2]);// probably can't compile to aocx
														//printf("ex1[colt=%d]=%f, ex2=%f\n", colt, buf[revcolt].x, buf[revcolt2].x);
		}
	}
	else if (pass&&get_global_id(0) < 2 * N*(N + 1) / POINTS)//second pass
	{
		where = colt * 8 + i * (POINTS)* N;
		int start = inverse ? colt / (N / 8)  * N : colt / (2 * N / 8) * 2 * N;
		int offset[8];
		for (int i = 0; i < 8; i++)
			offset[i] = inverse ? (bit_reversed(colt % (N / 8) * 8 + i, (LOGNTpl - 1))) : (bit_reversed(colt % (2 * N / 8) * 8 + i, (LOGNTpl - 1) + 1));
		dest[where] = buf[start + offset[0]];//colt / (2* N / 8)=row
		dest[where + 1] = buf[start + offset[1]];
		dest[where + 2] = buf[start + offset[2]];
		dest[where + 3] = buf[start + offset[3]];
		dest[where + 4] = buf[start + offset[4]];
		dest[where + 5] = buf[start + offset[5]];
		dest[where + 6] = buf[start + offset[6]];
		dest[where + 7] = buf[start + offset[7]];
	}

}

__attribute__((reqd_work_group_size(8, 1, 1)))
kernel void extraCol1(global float2 * restrict dest2, int inverse)
{
	float2 wirte;
	if (!inverse)
		dest2[get_global_id(0)] = read_channel_intel(chanEx10);
	else
	{
		dest2[get_global_id(0) * 2] = read_channel_intel(chanEx11);
		dest2[get_global_id(0) * 2 + 1] = read_channel_intel(chanEx12);
	}
}

__kernel void convertChar4ToFloatDoConGam(__global uchar4* restrict gd_ac4Scn, __global float4* restrict d_afScn, int dataN, int bConGam)
{
	//printf("------------------------%d", bConGam);
	uchar4 c4DataIn;
	float4 f4DataOut;
    for (int iIndx = 0; iIndx < dataN; iIndx++)
	{
		c4DataIn = gd_ac4Scn[iIndx];
		if (bConGam)
		{
			//doing ConGam takes 0.1 ms more //invalid for now
			f4DataOut.x = (float)d_acLUT[(int)(c4DataIn.x)];
			f4DataOut.y = (float)d_acLUT[(int)(c4DataIn.y)];
			f4DataOut.z = (float)d_acLUT[(int)(c4DataIn.z)];
			f4DataOut.w = (float)d_acLUT[(int)(c4DataIn.w)];
			//printf("%d",bConGam);
		}
		else
		{
			f4DataOut.x = (float)c4DataIn.x;
			f4DataOut.y = (float)c4DataIn.y;
			f4DataOut.z = (float)c4DataIn.z;
			f4DataOut.w = (float)c4DataIn.w;
			//printf("here2");
		}
		d_afScn[iIndx] = f4DataOut;
	}
}

//fix dead pixel with averaging 8 immediate neighbors. 
/*
__kernel void fixDeadPixels(__global float* d_afScn, int iScnSz, int iScnW, int iScnH)
{
	__local cufftReal afTopRow[(BLOCKDIMX + (HALFWARP + 1))];
	__local cufftReal afMidRow[(BLOCKDIMX + (HALFWARP + 1))];
	__local cufftReal afBotRow[(BLOCKDIMX + (HALFWARP + 1))];
	for (int b = 0; b < 600; b++) { // emulate the GridDim as it is in CUDA
		for (int t = 0; t < BLOCKDIMX; t++) { // emulate the BlockDim as it is in CUDA
			int blockIdx = t / BLOCKDIMX;
			int threadIdx = t % BLOCKDIMX;
			int blockDim = BLOCKDIMX;

			int iDeadRowDataIndx = IMUL(blockIdx, BLOCKDIMX) + (threadIdx - HALFWARP);
			int iDataIndx = iDeadRowDataIndx + IMUL((iDeadRowDataIndx / iScnW) + iAddOneRow, iScnW);

			afTopRow[threadIdx] = 0;
			afMidRow[threadIdx] = 0;
			afBotRow[threadIdx] = 0;

			if (iDataIndx >= 0 && iDataIndx < iScnSz && threadIdx >= HALFWARP - 1 && threadIdx <= (blockDim - 1))
			{
				int iRow = iDataIndx / iScnW;
				//read top row
				if (iRow > 0)
					afTopRow[threadIdx] = d_afScn[iDataIndx - iScnW];
				//read middle row
				afMidRow[threadIdx] = d_afScn[iDataIndx];
				//read bottom row
				if (iRow < iScnH - 1)
					afBotRow[threadIdx] = d_afScn[iDataIndx + iScnW];
			}

			barrier(CLK_LOCAL_MEM_FENCE);

			if (iDataIndx >= 0 && iDataIndx < iScnSz && threadIdx > HALFWARP - 1 && threadIdx < (blockDim - 1))
			{
				cufftReal fSum = 0;
				int iLeftIndx, iRightIndx;
				int iCol = iDataIndx%iScnW;
				if (iCol % 2 == 0)
				{
					fSum = fSum + afTopRow[threadIdx] + afBotRow[threadIdx];
					int iNumNeigh = 2;
					if (iCol > 0)
					{
						iLeftIndx = threadIdx - 1;
						fSum = fSum + afTopRow[iLeftIndx] + afMidRow[iLeftIndx] + afBotRow[iLeftIndx];
						iNumNeigh = iNumNeigh + 3;
						//fSum = fSum + afMidRow[iLeftIndx];
						//iNumNeigh = iNumNeigh + 1;
					}
					if (iCol < iScnW - 1)
					{
						iRightIndx = threadIdx + 1;
						fSum = fSum + afTopRow[iRightIndx] + afMidRow[iRightIndx] + afBotRow[iRightIndx];
						iNumNeigh = iNumNeigh + 3;
						//fSum = fSum + afMidRow[iRightIndx];
						//iNumNeigh = iNumNeigh + 1;
					}
					d_afScn[iDataIndx] = fSum / (float)iNumNeigh;
				}
			}
		}
	}
}*/

__kernel void kthLaw(__global float2* d_afPadScn, int dataN)
{
	//int iIndx = get_global_id(0);
	for(int iIndx=0; iIndx < dataN; iIndx++)
	{
		//afVals(:) = (abs(afVals(:)).^k) .* (cos(angle(afVals(:))) + sin(angle(afVals(:)))*i);
		float2 cDat = d_afPadScn[iIndx];
		//float fNewAbsDat = pow(sqrtf(pow(cDat.x, 2) + pow(cDat.y, 2)), FK);
		float fNewAbsDat = pow(sqrtf(cDat.x*cDat.x + cDat.y*cDat.y), FK);
		float fAngDat = atan2(cDat.y, cDat.x);
		cDat.x = fNewAbsDat*cosf(fAngDat);
		cDat.y = fNewAbsDat*sinf(fAngDat);
		d_afPadScn[iIndx] = cDat;
	}
}

__kernel void pointWiseMul(__global float2* restrict d_afCorr, __global float2* restrict d_afPadScn, __global float2* restrict d_afPadTpl, int dataN, float fScale)
{
	//int iIndx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	for (int iIndx = 0; iIndx < dataN; iIndx++)
	{
		float2 cDat = d_afPadScn[iIndx];
		float2 cKer = d_afPadTpl[iIndx];
		//take the conjugate of the kernel
		cKer.y = -cKer.y;
		float2 cMul = { cDat.x* cKer.x - cDat.y * cKer.y, cDat.y * cKer.x + cDat.x * cKer.y };
		//const float     q = 1.0f / (float)dataN;
		//cMul.x = q * cMul.x;
		//cMul.y = q * cMul.y;

		cMul.x = fScale * cMul.x;
		cMul.y = fScale * cMul.y;
		d_afCorr[iIndx] = cMul;
	}
}

__kernel void ComplexScale(__global float2* a, int size, float scale)
{
	for (int i = 0; i < size; i ++)
	{
		a[i].x = scale * a[i].x;
		a[i].y = scale * a[i].y;
	}
}

__kernel void max_k(__global  cufftReal* restrict  afData, __global int* restrict aiDataIdxs, int iSizeOfData,
	__global cufftReal* restrict afBlockMaxs, __global int* restrict aiBlockMaxIdxs)
{
	float Max;
	int MaxIdx;
	for (int t = 0; t < iSizeOfData; t++)
	{
		Max = fmax(Max, afData[t]);
		if (Max == afData[t])
			MaxIdx = t;
	}


	afBlockMaxs[0] = Max;
	aiBlockMaxIdxs[0] = MaxIdx;

}
