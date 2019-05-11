#include "../rgb.h"
#include <omp.h>
#include <math.h>
#include <stdio.h>

#define cuda_check_error(err_expr)\
{\
	cudaError_t err = (err_expr);\
	if(err != cudaSuccess){\
		printf("FATAL: Encountered cuda error: %s (%s) (%s,%u).\n Quitting.\n", cudaGetErrorName(err), cudaGetErrorString(err), __FILE__, __LINE__);\
		assert(0);\
	}\
}

//requirements:
//   blockDim.x == c, blockDim.y == c
//   gridDim.x == num_cells_x , gridDim.y == num_cells_y
__global__ void gather(rgb* data, unsigned int width, unsigned int height) {
	extern __shared__ rgb sdata[];
	unsigned int cell_start_y = blockIdx.y*blockDim.y;
	unsigned int cell_start_x = blockIdx.x*blockDim.x;
	unsigned int px_x = cell_start_x + threadIdx.x;
	unsigned int px_y = cell_start_y + threadIdx.y;
	unsigned int px_pos = px_x + (px_y*width);

	unsigned int i = threadIdx.x + threadIdx.y*blockDim.x;

	if (px_x < width && px_y < height) {
		sdata[i].r = data[px_pos].r;
		sdata[i].g = data[px_pos].g;
		sdata[i].b = data[px_pos].b;
	}
	__syncthreads();
	for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride && px_x + stride < width) {
			sdata[i].r = ((unsigned int)sdata[i].r + (unsigned int)sdata[i + stride].r) / 2;
			sdata[i].g = ((unsigned int)sdata[i].g + (unsigned int)sdata[i + stride].g) / 2;
			sdata[i].b = ((unsigned int)sdata[i].b + (unsigned int)sdata[i + stride].b) / 2;
		}
		__syncthreads();
	}

	__syncthreads();
	for (unsigned int stride = blockDim.y / 2; stride > 0; stride >>= 1) {
		if (threadIdx.y < stride && px_y + stride < height) {
			sdata[i].r = ((unsigned int)sdata[i].r + (unsigned int)sdata[i + (stride * blockDim.x)].r) / 2;
			sdata[i].g = ((unsigned int)sdata[i].g + (unsigned int)sdata[i + (stride * blockDim.x)].g) / 2;
			sdata[i].b = ((unsigned int)sdata[i].b + (unsigned int)sdata[i + (stride * blockDim.x)].b) / 2;
		}
		__syncthreads();
	}

	if (i == 0) {
		data[px_pos].r = sdata[i].r;
		data[px_pos].g = sdata[i].g;
		data[px_pos].b = sdata[i].b;
	}
}

//requirements:
//   blockDim.x == c, blockDim.y == c
//   gridDim.x == num_cells_x , gridDim.y == num_cells_y
__global__ void scatter(rgb* data, unsigned int width, unsigned int height) {
	extern __shared__ rgb sdata[];
	unsigned int cell_start_y = blockIdx.y*blockDim.y;
	unsigned int cell_start_x = blockIdx.x*blockDim.x;
	unsigned int px_x = cell_start_x + threadIdx.x;
	unsigned int px_y = cell_start_y + threadIdx.y;
	unsigned int px_pos = px_x + (px_y*width);

	unsigned int i = threadIdx.x + threadIdx.y*blockDim.x;

	if (i == 0) {
		sdata[i].r = data[px_pos].r;
		sdata[i].g = data[px_pos].g;
		sdata[i].b = data[px_pos].b;
	}

	__syncthreads();
	for (unsigned int stride = 1; stride < blockDim.x; stride <<= 1) {
		if (threadIdx.x < stride && px_x + stride < width) {
			sdata[i + stride].r = sdata[i].r;
			sdata[i + stride].g = sdata[i].g;
			sdata[i + stride].b = sdata[i].b;
		}
		__syncthreads();
	}

	__syncthreads();
	for (unsigned int stride = 1; stride < blockDim.y; stride <<= 1) {
		if (threadIdx.y < stride && px_y + stride < height) {
			sdata[i + (stride * blockDim.x)].r = sdata[i].r;
			sdata[i + (stride * blockDim.x)].g = sdata[i].g;
			sdata[i + (stride * blockDim.x)].b = sdata[i].b;
		}
		__syncthreads();
	}

	if (px_x < width && px_y < height) {
		data[px_pos].r = sdata[i].r;
		data[px_pos].g = sdata[i].g;
		data[px_pos].b = sdata[i].b;
	}
}

//requirements:
//   blockDim.x == num_cells_x/gridDim.x, blockDim.y == num_cells_y/gridDim.y
__global__ void global_avg(rgb* data, rgb* global_avg, unsigned int width, unsigned int height, unsigned int c) {
	extern __shared__ rgb sdata[];
	unsigned int px_x = threadIdx.x*c + blockIdx.x*blockDim.x;
	unsigned int px_y = threadIdx.y*c + blockIdx.y*blockDim.y;
	unsigned int y_offset = (px_y*width);
	unsigned int px_pos = px_x + y_offset;

	unsigned int i = threadIdx.x + threadIdx.y*blockDim.x;

	if (px_x < width && px_y < height) {
		sdata[i].r = data[px_pos].r;
		sdata[i].g = data[px_pos].g;
		sdata[i].b = data[px_pos].b;
	}

	__syncthreads();
	for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride && px_x + stride < width) {
			sdata[i].r = ((unsigned int)sdata[i].r + (unsigned int)sdata[i + stride].r) / 2;
			sdata[i].g = ((unsigned int)sdata[i].g + (unsigned int)sdata[i + stride].g) / 2;
			sdata[i].b = ((unsigned int)sdata[i].b + (unsigned int)sdata[i + stride].b) / 2;
		}
		__syncthreads();
	}

	__syncthreads();
	for (unsigned int stride = blockDim.y / 2; stride > 0; stride >>= 1) {
		//top left exists on all cells, so no need to check edge case
		if (threadIdx.y < stride && px_y + stride < height) {
			sdata[i].r = ((unsigned int)sdata[i].r + (unsigned int)sdata[i + (stride*blockDim.x)].r) / 2;
			sdata[i].g = ((unsigned int)sdata[i].g + (unsigned int)sdata[i + (stride*blockDim.x)].g) / 2;
			sdata[i].b = ((unsigned int)sdata[i].b + (unsigned int)sdata[i + (stride*blockDim.x)].b) / 2;
		}
		__syncthreads();
	}

	if (i == 0) {
		global_avg[blockIdx.x + blockIdx.y*gridDim.x].r = sdata[i].r;
		global_avg[blockIdx.x + blockIdx.y*gridDim.x].g = sdata[i].g;
		global_avg[blockIdx.x + blockIdx.y*gridDim.x].b = sdata[i].b;
	}
}

void run_cuda(rgb* data, unsigned int width, unsigned int height, unsigned int wb_width, unsigned int wb_height, unsigned int c) {
	int cudaDevice;
	int maxThreadsPerBlock;

	cudaStream_t avg_stream, scatter_stream;
	cuda_check_error(cudaStreamCreate(&avg_stream));
	cuda_check_error(cudaStreamCreate(&scatter_stream));

	cuda_check_error(cudaGetDevice(&cudaDevice));
	cuda_check_error(cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, cudaDevice));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//starting cpu based timing here
	double begin = omp_get_wtime();

	int num_cells_x = (width + (c - 1)) / c;
	int num_cells_y = (height + (c - 1)) / c;

	int global_avg_dim = (int)ceil(sqrt((c * c) + (maxThreadsPerBlock - 1) / maxThreadsPerBlock));

	rgb* gpu_global_avg;
	rgb* gpu_data;
	rgb* pre_summed_avgs = (rgb*)malloc((global_avg_dim*global_avg_dim) * sizeof(rgb));

	cuda_check_error(cudaMalloc((void**)&gpu_global_avg, global_avg_dim *global_avg_dim * sizeof(rgb)));
	cuda_check_error(cudaMalloc((void**)&gpu_data, width*height * sizeof(rgb)));
	cuda_check_error(cudaMemcpy(gpu_data, data, width*height * sizeof(rgb), cudaMemcpyHostToDevice));

	//run kernel code
	cuda_check_error(cudaEventRecord(start));
	{
		gather << < dim3(num_cells_x, num_cells_y, 1), dim3(c, c, 1), c * c * sizeof(rgb) >> > (gpu_data, width, height);
		cuda_check_error(cudaGetLastError());
		cuda_check_error(cudaDeviceSynchronize());
	}
	{
		scatter << < dim3(num_cells_y, num_cells_y, 1), dim3(c, c, 1), c * c * sizeof(rgb), scatter_stream >> > (gpu_data, width, height);
		cuda_check_error(cudaGetLastError());
		cuda_check_error(cudaMemcpyAsync(data, gpu_data, width*height * sizeof(rgb), cudaMemcpyDeviceToHost, scatter_stream));
	}

	{
		global_avg << < dim3(global_avg_dim, global_avg_dim, 1), dim3(num_cells_x / global_avg_dim, num_cells_y / global_avg_dim, 1), (num_cells_x / global_avg_dim) * (num_cells_y / global_avg_dim) * sizeof(rgb), avg_stream >> > (gpu_data, gpu_global_avg, width, height, c);
		cuda_check_error(cudaGetLastError());
		cuda_check_error(cudaMemcpyAsync(pre_summed_avgs, gpu_global_avg, (global_avg_dim*global_avg_dim) * sizeof(rgb), cudaMemcpyDeviceToHost, avg_stream));
	}
	//wait for average to complete, then do the final small gather on the cpu
	cuda_check_error(cudaStreamSynchronize(avg_stream));

	big_rgb global_avg = { 0,0,0 };

	for (int i = 0; i < (global_avg_dim*global_avg_dim); i++) {
		global_avg.r += pre_summed_avgs[i].r;
		global_avg.g += pre_summed_avgs[i].g;
		global_avg.b += pre_summed_avgs[i].b;
	}

	free(pre_summed_avgs);

	global_avg.r /= (global_avg_dim*global_avg_dim);
	global_avg.g /= (global_avg_dim*global_avg_dim);
	global_avg.b /= (global_avg_dim*global_avg_dim);

	cuda_check_error(cudaFree(gpu_global_avg));

	//wait for scatter to complete
	cuda_check_error(cudaStreamSynchronize(scatter_stream));

	//free scatter memory
	cuda_check_error(cudaFree(gpu_data));

	cuda_check_error(cudaStreamSynchronize(avg_stream));
	cuda_check_error(cudaEventRecord(stop));
	cuda_check_error(cudaEventSynchronize(stop));

	cudaStreamDestroy(scatter_stream);
	cudaStreamDestroy(avg_stream);

	// Output the average colour value for the image
	printf("CUDA Average image colour red = %u, green = %u, blue = %u \n", global_avg.r, global_avg.g, global_avg.b);

	//end timing here
	double end = omp_get_wtime();
	double seconds = (end - begin);

	float cudaMs;

	cudaEventElapsedTime(&cudaMs, start, stop);

	double s;
	double ms = modf(seconds, &s)*1000.0;
	printf("CUDA mode execution time took %d s and %f ms (%f ms as measured by cuda)\n", (int)s, ms, cudaMs);
}