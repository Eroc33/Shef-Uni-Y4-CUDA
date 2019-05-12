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
//   blockDim.x == c, blockDim.y == 1
//   gridDim.x == num_cells_x , gridDim.y == 1
__global__ void row_reduction(rgb* data, unsigned int width, unsigned int height) {
	extern __shared__ rgb sdata[];
	unsigned int y = blockIdx.y;
	unsigned int cell_start = blockIdx.x*blockDim.x;
	unsigned int px_x = cell_start + threadIdx.x;
	unsigned int y_offset = (y*width);

	if (px_x < width) {
		sdata[threadIdx.x].r = data[px_x + y_offset].r;
		sdata[threadIdx.x].g = data[px_x + y_offset].g;
		sdata[threadIdx.x].b = data[px_x + y_offset].b;
	}
	__syncthreads();
	for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride && px_x+stride < width) {
			sdata[threadIdx.x].r += sdata[threadIdx.x + stride].r;
			sdata[threadIdx.x].g += sdata[threadIdx.x + stride].g;
			sdata[threadIdx.x].b += sdata[threadIdx.x + stride].b;

			sdata[threadIdx.x].r /= 2;
			sdata[threadIdx.x].g /= 2;
			sdata[threadIdx.x].b /= 2;
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		data[px_x + y_offset].r = sdata[threadIdx.x].r;
		data[px_x + y_offset].g = sdata[threadIdx.x].g;
		data[px_x + y_offset].b = sdata[threadIdx.x].b;
	}
}

//col_reduction, similar to row_reduction but reduce the already reduced rows into columns
__global__ void col_reduction(rgb* data, unsigned int width, unsigned int height, unsigned int c) {
	extern __shared__ rgb sdata[];
	//load sdata
	unsigned int x = blockIdx.y*c;
	unsigned int cell_start_y = blockIdx.x*blockDim.x;
	unsigned int px_y = cell_start_y + threadIdx.x;
	unsigned int px_pos = x+(px_y*width);
	if (px_y < height) {
		sdata[threadIdx.x].r = data[px_pos].r;
		sdata[threadIdx.x].g = data[px_pos].g;
		sdata[threadIdx.x].b = data[px_pos].b;
	}
	__syncthreads();
	for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride && px_y+stride < height) {
			sdata[threadIdx.x].r = ((unsigned int)sdata[threadIdx.x].r + (unsigned int)sdata[threadIdx.x + stride].r) / 2;
			sdata[threadIdx.x].g = ((unsigned int)sdata[threadIdx.x].g + (unsigned int)sdata[threadIdx.x + stride].g) / 2;
			sdata[threadIdx.x].b = ((unsigned int)sdata[threadIdx.x].b + (unsigned int)sdata[threadIdx.x + stride].b) / 2;
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		data[px_pos].r = sdata[threadIdx.x].r;
		data[px_pos].g = sdata[threadIdx.x].g;
		data[px_pos].b = sdata[threadIdx.x].b;
	}
}

__global__ void scatter(rgb* data, unsigned int width, unsigned int height, unsigned int wb_width, unsigned wb_height, unsigned int c){

    unsigned int x = (blockIdx.x * 32) + threadIdx.x;
    unsigned int y = (blockIdx.y * 32) + threadIdx.y;

	if (x >= wb_width || y >= wb_height) {
		//this is not at all efficient :(
		return;
	}
    
    //cell height is input cell size
    unsigned int cell_height = c;
    //unless this is the last row of cells, and c does not evenly divide height
    if (height%cell_height != 0 && y + 1 == wb_height) {
        cell_height = height % cell_height;
    }
    unsigned int cell_y = y*c;
    //cell width is input cell size
    unsigned int cell_width = c;
    //unless this is the last column of cells, and c does not evenly divide width
    if (width%cell_width != 0 && x + 1 == wb_width) {
        cell_width = width % cell_width;
    }
    unsigned int cell_size = cell_width * cell_height;
    unsigned int cell_x = x*c;

	rgb out = data[cell_y*width + cell_x];

    //copy to all ouput buffer cells
    for (unsigned int cy = 0; cy < cell_height; cy++) {
        for (unsigned int cx = 0; cx < cell_width; cx++) {
            int i = ((cell_y + cy)*width) + cell_x + cx;
            data[i] = out;
        }
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
		row_reduction << < dim3(num_cells_x, height, 1), c, c * sizeof(rgb) >> > (gpu_data, width, height);
		cuda_check_error(cudaGetLastError());
	}
	{
		col_reduction << < dim3(num_cells_y, width, 1), c, c * sizeof(rgb) >> > (gpu_data, width, height, c);
		cuda_check_error(cudaGetLastError());
	}
	{
		dim3 blocksPerGrid((wb_width + (32 - 1)) / 32, (wb_height + (32 - 1)) / 32, 1);
		dim3 threadsPerBlock(32, 32, 1);
		scatter << <blocksPerGrid, threadsPerBlock >> > (gpu_data, width, height, wb_width, wb_height, c);
		cuda_check_error(cudaGetLastError());
	}
	{
		global_avg << < dim3(global_avg_dim, global_avg_dim, 1), dim3(num_cells_x / global_avg_dim, num_cells_y / global_avg_dim, 1), (num_cells_x / global_avg_dim) * (num_cells_y / global_avg_dim) * sizeof(rgb) >> > (gpu_data, gpu_global_avg, width, height, c);
		cuda_check_error(cudaGetLastError());
	}

	cuda_check_error(cudaMemcpy(pre_summed_avgs, gpu_global_avg, (global_avg_dim*global_avg_dim) * sizeof(rgb), cudaMemcpyDeviceToHost));

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
	cuda_check_error(cudaMemcpy(data, gpu_data, width*height * sizeof(rgb), cudaMemcpyDeviceToHost));
	cuda_check_error(cudaFree(gpu_data));

	cuda_check_error(cudaEventRecord(stop));
	cuda_check_error(cudaEventSynchronize(stop));


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