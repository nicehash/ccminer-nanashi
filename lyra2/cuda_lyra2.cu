/**
* Lyra2 (v1) cuda implementation based on djm34 work - SM 5/5.2
* tpruvot@github 2015
*/

#include <stdio.h>
#include <memory.h>

#define TPB52 32
#define TPB50 16
#define TPB30 16
#define TPB20 16

#ifdef __INTELLISENSE__
/* just for vstudio code colors */
#define __CUDA_ARCH__ 500
#endif

#include "cuda_lyra2_vectors.h"

#ifdef __INTELLISENSE__
/* just for vstudio code colors */
#if __CUDA_ARCH__ >= 300
__device__ uint32_t __shfl(uint32_t a, uint32_t b, uint32_t c);
#endif
#endif

#define Nrow 8
#define Ncol 8
#define memshift 3

__device__ uint2x4 *DState;

__device__ __forceinline__ uint2 LD4S(const int index)
{
	extern __shared__ uint2 shared_mem[];

	return shared_mem[(index * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x];
}

__device__ __forceinline__ void ST4S(const int index, const uint2 data)
{
	extern __shared__ uint2 shared_mem[];

	shared_mem[(index * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x] = data;
}

#if __CUDA_ARCH__ >= 300
__device__ __forceinline__ uint32_t WarpShuffle(uint32_t a, uint32_t b, uint32_t c)
{
	return __shfl(a, b, c);
}

__device__ __forceinline__ uint2 WarpShuffle(uint2 a, uint32_t b, uint32_t c)
{
	return make_uint2(__shfl(a.x, b, c), __shfl(a.y, b, c));
}

__device__ __forceinline__ void WarpShuffle3(uint2 &a1, uint2 &a2, uint2 &a3, uint32_t b1, uint32_t b2, uint32_t b3, uint32_t c)
{
	a1 = WarpShuffle(a1, b1, c);
	a2 = WarpShuffle(a2, b2, c);
	a3 = WarpShuffle(a3, b3, c);
}

#else
__device__ __forceinline__ uint32_t WarpShuffle(uint32_t a, uint32_t b, uint32_t c)
{
	extern __shared__ uint2 shared_mem[];

	const uint32_t thread = blockDim.x * threadIdx.y + threadIdx.x;
	uint32_t *_ptr = (uint32_t*)shared_mem;

	__threadfence_block();
	uint32_t buf = _ptr[thread];

	_ptr[thread] = a;
	__threadfence_block();
	uint32_t result = _ptr[(thread&~(c - 1)) + (b&(c - 1))];

	__threadfence_block();
	_ptr[thread] = buf;

	__threadfence_block();
	return result;
}

__device__ __forceinline__ uint2 WarpShuffle(uint2 a, uint32_t b, uint32_t c)
{
	extern __shared__ uint2 shared_mem[];

	const uint32_t thread = blockDim.x * threadIdx.y + threadIdx.x;

	__threadfence_block();
	uint2 buf = shared_mem[thread];

	shared_mem[thread] = a;
	__threadfence_block();
	uint2 result = shared_mem[(thread&~(c - 1)) + (b&(c - 1))];

	__threadfence_block();
	shared_mem[thread] = buf;

	__threadfence_block();
	return result;
}

__device__ __forceinline__ void WarpShuffle3(uint2 &a1, uint2 &a2, uint2 &a3, uint32_t b1, uint32_t b2, uint32_t b3, uint32_t c)
{
	extern __shared__ uint2 shared_mem[];

	const uint32_t thread = blockDim.x * threadIdx.y + threadIdx.x;

	__threadfence_block();
	uint2 buf = shared_mem[thread];

	shared_mem[thread] = a1;
	__threadfence_block();
	a1 = shared_mem[(thread&~(c - 1)) + (b1&(c - 1))];
	__threadfence_block();
	shared_mem[thread] = a2;
	__threadfence_block();
	a2 = shared_mem[(thread&~(c - 1)) + (b2&(c - 1))];
	__threadfence_block();
	shared_mem[thread] = a3;
	__threadfence_block();
	a3 = shared_mem[(thread&~(c - 1)) + (b3&(c - 1))];

	__threadfence_block();
	shared_mem[thread] = buf;
	__threadfence_block();
}

#endif

static __device__ __forceinline__
void Gfunc(uint2 &a, uint2 &b, uint2 &c, uint2 &d)
{
	a += b; d = eorswap32(a, d);
	c += d; b ^= c; b = ROR24(b);
	a += b; d ^= a; d = ROR16(d);
	c += d; b ^= c; b = ROR2(b, 63);
}

__device__ __forceinline__ void round_lyra(uint2 s[4])
{
	Gfunc(s[0], s[1], s[2], s[3]);
	WarpShuffle3(s[1], s[2], s[3], threadIdx.x + 1, threadIdx.x + 2, threadIdx.x + 3, 4);
	Gfunc(s[0], s[1], s[2], s[3]);
	WarpShuffle3(s[1], s[2], s[3], threadIdx.x + 3, threadIdx.x + 2, threadIdx.x + 1, 4);
}

static __device__ __forceinline__
void round_lyra(uint2x4* s)
{
	Gfunc(s[0].x, s[1].x, s[2].x, s[3].x);
	Gfunc(s[0].y, s[1].y, s[2].y, s[3].y);
	Gfunc(s[0].z, s[1].z, s[2].z, s[3].z);
	Gfunc(s[0].w, s[1].w, s[2].w, s[3].w);
	Gfunc(s[0].x, s[1].y, s[2].z, s[3].w);
	Gfunc(s[0].y, s[1].z, s[2].w, s[3].x);
	Gfunc(s[0].z, s[1].w, s[2].x, s[3].y);
	Gfunc(s[0].w, s[1].x, s[2].y, s[3].z);
}

static __device__ __forceinline__
void reduceDuplex(uint2 state[4], uint32_t thread)
{
	uint2 state1[3];

#if __CUDA_ARCH__ > 500
#pragma unroll
#endif
	for (int i = 0; i < Nrow; i++)
	{
		const uint32_t s0 = memshift * Ncol * 0 + memshift * (Ncol - 1) - i * memshift;
#pragma unroll
		for (int j = 0; j < 3; j++)
			ST4S(s0 + j, state[j]);

		round_lyra(state);
	}

#pragma unroll 4
	for (int i = 0; i < Nrow; i++)
	{
		const uint32_t s0 = memshift * Ncol * 0 + i * memshift;
		const uint32_t s1 = memshift * Ncol * 1 + memshift * (Ncol - 1) - i*memshift;

		for (int j = 0; j < 3; j++)
			state1[j] = LD4S(s0 + j);
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j];

		round_lyra(state);

		for (int j = 0; j < 3; j++)
			state1[j] ^= state[j];
		for (int j = 0; j < 3; j++)
			ST4S(s1 + j, state1[j]);
	}
}

static __device__ __forceinline__
void reduceDuplexRowSetup(const int rowIn, const int rowInOut, const int rowOut, uint2 state[4], uint32_t thread)
{
	uint2 state1[3], state2[3];

#pragma unroll 1
	for (int i = 0; i < Nrow; i++)
	{
		const uint32_t s1 = memshift * Ncol * rowIn + i * memshift;
		const uint32_t s2 = memshift * Ncol * rowInOut + i * memshift;
		const uint32_t s3 = memshift * Ncol * rowOut + memshift * (Ncol - 1) - i*memshift;

		for (int j = 0; j < 3; j++)
			state1[j] = LD4S(s1 + j);
		for (int j = 0; j < 3; j++)
			state2[j] = LD4S(s2 + j);
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

		for (int j = 0; j < 3; j++)
			ST4S(s3 + j, state1[j] ^ state[j]);

		//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		}
		else
		{
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

		for (int j = 0; j < 3; j++)
			ST4S(s2 + j, state2[j]);
	}
}

static __device__ __forceinline__
void reduceDuplexRowt(const int rowIn, const int rowInOut, const int rowOut, uint2 state[4], const uint32_t thread)
{
	for (int i = 0; i < Nrow; i++)
	{
		uint2 state1[3], state2[3];

		const uint32_t s1 = memshift * Ncol * rowIn + i * memshift;
		const uint32_t s2 = memshift * Ncol * rowInOut + i * memshift;
		const uint32_t s3 = memshift * Ncol * rowOut + i * memshift;

#pragma unroll
		for (int j = 0; j < 3; j++) {
			state1[j] = LD4S(s1 + j);
			state2[j] = LD4S(s2 + j);
		}

#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

		//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		}
		else
		{
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

#pragma unroll
		for (int j = 0; j < 3; j++)
			ST4S(s2 + j, state2[j]);
#pragma unroll
		for (int j = 0; j < 3; j++)
			ST4S(s3 + j, LD4S(s3 + j) ^ state[j]);
	}
}

static __device__ __forceinline__
void reduceDuplexRowt_8(const int rowInOut, uint2* state, const uint32_t thread)
{

	uint2 state1[3], state2[3], last[3];

	const uint32_t ps1 = memshift * Ncol * 2;
	const uint32_t ps2 = memshift * Ncol * rowInOut;
	const uint32_t ps3 = memshift * Ncol * 5;

#pragma unroll
	for (int j = 0; j < 3; j++)
		last[j] = LD4S(ps2 + j);

#pragma unroll 
	for (int j = 0; j < 3; j++)
		state[j] ^= LD4S(ps1 + j) + last[j];

	round_lyra(state);

	//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
	uint2 Data0 = state[0];
	uint2 Data1 = state[1];
	uint2 Data2 = state[2];
	WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

	if (threadIdx.x == 0)
	{
		last[0] ^= Data2;
		last[1] ^= Data0;
		last[2] ^= Data1;
	}
	else
	{
		last[0] ^= Data0;
		last[1] ^= Data1;
		last[2] ^= Data2;
	}

	if (rowInOut == 5)
	{
#pragma unroll 
		for (int j = 0; j < 3; j++)
			last[j] ^= state[j];
	}

	for (int i = 1; i < Nrow; i++)
	{

		const uint32_t s1 = memshift * Ncol * 2 + i * memshift;
		const uint32_t s2 = memshift * Ncol * rowInOut + i * memshift;

#pragma unroll 
		for (int j = 0; j < 3; j++)
			state[j] ^= LD4S(s1 + j) + LD4S(s2 + j);

		round_lyra(state);
	}


#pragma unroll
	for (int j = 0; j < 3; j++)
		state[j] ^= last[j];
}

__constant__ uint28 blake2b_IV[2] = {
	0xf3bcc908lu, 0x6a09e667lu,
	0x84caa73blu, 0xbb67ae85lu,
	0xfe94f82blu, 0x3c6ef372lu,
	0x5f1d36f1lu, 0xa54ff53alu,
	0xade682d1lu, 0x510e527flu,
	0x2b3e6c1flu, 0x9b05688clu,
	0xfb41bd6blu, 0x1f83d9ablu,
	0x137e2179lu, 0x5be0cd19lu
};

__global__ __launch_bounds__(64, 1)
void lyra2_gpu_hash_32_1(uint32_t threads, uint32_t startNounce, uint2 *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads)
	{
		uint2x4 state[4];

		state[0].x = state[1].x = __ldg(&g_hash[thread + threads * 0]);
		state[0].y = state[1].y = __ldg(&g_hash[thread + threads * 1]);
		state[0].z = state[1].z = __ldg(&g_hash[thread + threads * 2]);
		state[0].w = state[1].w = __ldg(&g_hash[thread + threads * 3]);
		state[2] = blake2b_IV[0];
		state[3] = blake2b_IV[1];

		for (int i = 0; i<24; i++)
			round_lyra(state); //because 12 is not enough

		DState[blockDim.x * gridDim.x * 0 + blockDim.x * blockIdx.x + threadIdx.x] = state[0];
		DState[blockDim.x * gridDim.x * 1 + blockDim.x * blockIdx.x + threadIdx.x] = state[1];
		DState[blockDim.x * gridDim.x * 2 + blockDim.x * blockIdx.x + threadIdx.x] = state[2];
		DState[blockDim.x * gridDim.x * 3 + blockDim.x * blockIdx.x + threadIdx.x] = state[3];
	}
}

#if __CUDA_ARCH__ < 300
__global__ __launch_bounds__(TPB20, 1)
#elif __CUDA_ARCH__ < 500
__global__ __launch_bounds__(TPB30, 1)
#elif __CUDA_ARCH__ == 500
__global__ __launch_bounds__(TPB50, 1)
#else
__global__ __launch_bounds__(TPB52, 1)
#endif
void lyra2_gpu_hash_32_2(uint32_t threads, uint32_t startNounce, uint64_t *g_hash)
{
	const uint32_t thread = blockDim.y * blockIdx.x + threadIdx.y;

	if (thread < threads)
	{
		uint2 state[4];
		state[0] = ((uint2*)DState)[(0 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x];
		state[1] = ((uint2*)DState)[(1 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x];
		state[2] = ((uint2*)DState)[(2 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x];
		state[3] = ((uint2*)DState)[(3 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x];

		reduceDuplex(state, thread);

		reduceDuplexRowSetup(1, 0, 2, state, thread);
		reduceDuplexRowSetup(2, 1, 3, state, thread);
		reduceDuplexRowSetup(3, 0, 4, state, thread);
		reduceDuplexRowSetup(4, 3, 5, state, thread);
		reduceDuplexRowSetup(5, 2, 6, state, thread);
		reduceDuplexRowSetup(6, 1, 7, state, thread);

		uint32_t rowa = WarpShuffle(state[0].x, 0, 4) & 7;
		reduceDuplexRowt(7, rowa, 0, state, thread);
		rowa = WarpShuffle(state[0].x, 0, 4) & 7;
		reduceDuplexRowt(0, rowa, 3, state, thread);
		rowa = WarpShuffle(state[0].x, 0, 4) & 7;
		reduceDuplexRowt(3, rowa, 6, state, thread);
		rowa = WarpShuffle(state[0].x, 0, 4) & 7;
		reduceDuplexRowt(6, rowa, 1, state, thread);
		rowa = WarpShuffle(state[0].x, 0, 4) & 7;
		reduceDuplexRowt(1, rowa, 4, state, thread);
		rowa = WarpShuffle(state[0].x, 0, 4) & 7;
		reduceDuplexRowt(4, rowa, 7, state, thread);
		rowa = WarpShuffle(state[0].x, 0, 4) & 7;
		reduceDuplexRowt(7, rowa, 2, state, thread);
		rowa = WarpShuffle(state[0].x, 0, 4) & 7;
		reduceDuplexRowt_8(rowa, state, thread);

		((uint2*)DState)[(0 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x] = state[0];
		((uint2*)DState)[(1 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x] = state[1];
		((uint2*)DState)[(2 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x] = state[2];
		((uint2*)DState)[(3 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x] = state[3];
	}
}

__global__ __launch_bounds__(64, 1)
void lyra2_gpu_hash_32_3(uint32_t threads, uint32_t startNounce, uint2 *g_hash)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;

	uint28 state[4];

	if (thread < threads)
	{
		state[0] = __ldg4(&DState[blockDim.x * gridDim.x * 0 + blockDim.x * blockIdx.x + threadIdx.x]);
		state[1] = __ldg4(&DState[blockDim.x * gridDim.x * 1 + blockDim.x * blockIdx.x + threadIdx.x]);
		state[2] = __ldg4(&DState[blockDim.x * gridDim.x * 2 + blockDim.x * blockIdx.x + threadIdx.x]);
		state[3] = __ldg4(&DState[blockDim.x * gridDim.x * 3 + blockDim.x * blockIdx.x + threadIdx.x]);

		for (int i = 0; i < 12; i++)
			round_lyra(state);

		g_hash[thread + threads * 0] = state[0].x;
		g_hash[thread + threads * 1] = state[0].y;
		g_hash[thread + threads * 2] = state[0].z;
		g_hash[thread + threads * 3] = state[0].w;

	} //thread
}

__host__
void lyra2_cpu_init(int thr_id, uint32_t threads, uint64_t *d_matrix)
{
	int dev_id = device_map[thr_id % MAX_GPUS];
	// just assign the device pointer allocated in main loop
	cudaMemcpyToSymbol(DState, &d_matrix, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice);
}

__host__
void lyra2_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_hash, int order)
{
	int dev_id = device_map[thr_id % MAX_GPUS];

	uint32_t tpb = TPB52;

	if (cuda_arch[dev_id] > 500) tpb = TPB52;
	else if (cuda_arch[dev_id] == 500) tpb = TPB50;
	else if (cuda_arch[dev_id] >= 300) tpb = TPB30;
	else if (cuda_arch[dev_id] >= 200) tpb = TPB20;

	dim3 grid1((threads * 4 + tpb - 1) / tpb);
	dim3 block1(4, tpb >> 2);

	dim3 grid2((threads + 64 - 1) / 64);
	dim3 block2(64);

	if (cuda_arch[dev_id] < 500)
		cudaFuncSetCacheConfig(lyra2_gpu_hash_32_2, cudaFuncCachePreferShared);

	lyra2_gpu_hash_32_1 << <grid2, block2 >> > (threads, startNounce, (uint2*)d_hash);

	lyra2_gpu_hash_32_2 << <grid1, block1, 192 * sizeof(uint2) * tpb >> > (threads, startNounce, d_hash);

	lyra2_gpu_hash_32_3 << <grid2, block2 >> > (threads, startNounce, (uint2*)d_hash);
	//MyStreamSynchronize(NULL, order, thr_id);
}
