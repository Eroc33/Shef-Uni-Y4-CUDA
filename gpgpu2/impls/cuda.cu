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
	unsigned int y_offset = (px_y*width);
	unsigned int px_pos = px_x + y_offset;

	unsigned int i = threadIdx.x+threadIdx.y*blockDim.x;

	if (px_x < width && px_y < height) {
		sdata[i].r = data[px_pos].r;
		sdata[i].g = data[px_pos].g;
		sdata[i].b = data[px_pos].b;
	}
	__syncthreads();
	for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride && px_x+stride < width) {
			sdata[i].r = ((unsigned int)sdata[i].r + (unsigned int)sdata[i].r) / 2;
			sdata[i].g = ((unsigned int)sdata[i].g + (unsigned int)sdata[i].g) / 2;
			sdata[i].b = ((unsigned int)sdata[i].b + (unsigned int)sdata[i].b) / 2;
		}
		__syncthreads();
	}

	for (unsigned int stride = blockDim.y / 2; stride > 0; stride >>= 1) {
		if (threadIdx.y < stride && px_y + stride < height) {
			sdata[i].r = ((unsigned int)sdata[i].r + (unsigned int)sdata[i].r) / 2;
			sdata[i].g = ((unsigned int)sdata[i].g + (unsigned int)sdata[i].g) / 2;
			sdata[i].b = ((unsigned int)sdata[i].b + (unsigned int)sdata[i].b) / 2;
		}
		__syncthreads();
	}

	if (threadIdx.x == threadIdx.y == 0) {
		data[px_pos].r = sdata[threadIdx.x].r;
		data[px_pos].g = sdata[threadIdx.x].g;
		data[px_pos].b = sdata[threadIdx.x].b;
	}
}

//requirements:
//   blockDim.x == c, blockDim.y == c
//   gridDim.x == num_cells_x , gridDim.y == 1
__global__ void scatter(rgb* data, unsigned int width, unsigned int height) {
	extern __shared__ rgb sdata[];
	unsigned int cell_start_y = blockIdx.y*blockDim.y;
	unsigned int cell_start_x = blockIdx.x*blockDim.x;
	unsigned int px_x = cell_start_x + threadIdx.x;
	unsigned int px_y = cell_start_y + threadIdx.y;
	unsigned int y_offset = (px_y*width);
	unsigned int px_pos = px_x + y_offset;

	unsigned int i = threadIdx.x + threadIdx.y*blockDim.x;

	if (threadIdx.x == threadIdx.y == 0) {
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

	for (unsigned int stride = 1; stride < blockDim.y; stride <<= 1) {
		if (threadIdx.y < stride && px_y + stride < height) {
			sdata[i + stride].r = sdata[i].r;
			sdata[i + stride].g = sdata[i].g;
			sdata[i + stride].b = sdata[i].b;
		}
		__syncthreads();
	}

	if (px_x < width && px_y < height) {
		data[px_pos].r = sdata[i].r;
		data[px_pos].g = sdata[i].g;
		data[px_pos].b = sdata[i].b;
	}
}

void run_cuda(rgb* data, unsigned int width, unsigned int height, unsigned int wb_width, unsigned int wb_height, unsigned int c){
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//starting cpu based timing here
	double begin = omp_get_wtime();

	int num_cells_x = (width + (c - 1)) / c;
	int num_cells_y = (height + (c - 1)) / c;

    big_rgb global_avg = {0,0,0};
    
    rgb* gpu_data;

    cuda_check_error(cudaMalloc((void**)&gpu_data,width*height*sizeof(rgb)));
    cuda_check_error(cudaMemcpy(gpu_data, data, width*height*sizeof(rgb), cudaMemcpyHostToDevice));

    //run kernel code
	cuda_check_error(cudaEventRecord(start));
	gather <<< dim3(num_cells_x, num_cells_y,1), dim3(c,c,1), c * c * sizeof(rgb) >>> (gpu_data, width, height);
	cuda_check_error(cudaGetLastError());
	scatter << < dim3(num_cells_y, num_cells_y, 1), dim3(c, c, 1), c * c * sizeof(rgb) >> > (gpu_data, width, height);
	cuda_check_error(cudaGetLastError());
	cuda_check_error(cudaEventRecord(stop));
	cuda_check_error(cudaEventSynchronize(stop));

	//TODO: global avg

    cuda_check_error(cudaMemcpy(data, gpu_data, width*height*sizeof(rgb), cudaMemcpyDeviceToHost));
    
    cuda_check_error(cudaFree(gpu_data));

	// Output the average colour value for the image
	printf("CUDA Average image colour red = %u, green = %u, blue = %u \n",(unsigned char)global_avg.r,(unsigned char)global_avg.g,(unsigned char)global_avg.b);

	//end timing here
	double end = omp_get_wtime();
	double seconds = (end - begin);

	float cudaMs;

	cudaEventElapsedTime(&cudaMs,start,stop);

	double s;
	double ms = modf(seconds,&s)*1000.0;
	printf("CUDA mode execution time took %d s and %f ms (%f ms as measured by cuda)\n",(int)s,ms,cudaMs);
}