#include "../rgb.h"
#include <omp.h>
#include <math.h>
#include <stdio.h>

#define cuda_check_error(err_expr)\
{\
	cudaError_t err = (err_expr);\
	if(err != cudaSuccess){\
		printf("FATAL: Encountered cuda error: %s (%s) (%s,%u).\n Quitting.\n", cudaGetErrorName(err), cudaGetErrorString(err), __FILE__, __LINE__);\
		exit(EXIT_FAILURE);\
	}\
}


//Averages a block of pixels from right to left into the leftmost column of
//their cell.
//requirements:
//   blockDim.x == c, blockDim.y == n
//   gridDim.x == num_cells_x , gridDim.y == ceil(height/n)
__global__ void row_reduction(rgb* data, unsigned int width, unsigned int height) {
	extern __shared__ rgb sdata[];

	unsigned int px_x = (blockIdx.x*blockDim.x) + threadIdx.x;
	unsigned int px_y = (blockIdx.y*blockDim.y) + threadIdx.y;
	unsigned int px_pos = px_x + (px_y*width);

	unsigned int i = threadIdx.x + (threadIdx.y * blockDim.x);

	if (px_x < width && px_y < height) {
		sdata[i].r = data[px_pos].r;
		sdata[i].g = data[px_pos].g;
		sdata[i].b = data[px_pos].b;
	}
	__syncthreads();
	for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride && px_x + stride < width && px_y < height) {
			sdata[i].r = ((unsigned int)sdata[i].r + (unsigned int)sdata[i + stride].r) / 2;
			sdata[i].g = ((unsigned int)sdata[i].g + (unsigned int)sdata[i + stride].g) / 2;
			sdata[i].b = ((unsigned int)sdata[i].b + (unsigned int)sdata[i + stride].b) / 2;
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		data[px_pos].r = sdata[i].r;
		data[px_pos].g = sdata[i].g;
		data[px_pos].b = sdata[i].b;
	}
}

//Averages a block of pixels from bottom to top into the topmost row of
//their cell.
//requirements:
//   blockDim.x == 1, blockDim.y == c
//   gridDim.x == height , gridDim.y == num_cells_y
__global__ void col_reduction(rgb* data, unsigned int width, unsigned int height) {
	extern __shared__ rgb sdata[];

	unsigned int px_x = (blockIdx.x*blockDim.x) + threadIdx.x;
	unsigned int px_y = (blockIdx.y*blockDim.y) + threadIdx.y;
	unsigned int px_pos = px_x + (px_y*width);

	unsigned int i = threadIdx.y + (threadIdx.x * blockDim.y);

	if (px_y < height && px_x < width) {
		sdata[i].r = data[px_pos].r;
		sdata[i].g = data[px_pos].g;
		sdata[i].b = data[px_pos].b;
	}
	__syncthreads();
	for (unsigned int stride = blockDim.y / 2; stride > 0; stride >>= 1) {
		if (threadIdx.y < stride && px_y + stride < height && px_x < width) {
			sdata[i].r = ((unsigned int)sdata[i].r + (unsigned int)sdata[i + stride].r) / 2;
			sdata[i].g = ((unsigned int)sdata[i].g + (unsigned int)sdata[i + stride].g) / 2;
			sdata[i].b = ((unsigned int)sdata[i].b + (unsigned int)sdata[i + stride].b) / 2;
		}
		__syncthreads();
	}

	if (threadIdx.y == 0) {
		data[px_pos].r = sdata[i].r;
		data[px_pos].g = sdata[i].g;
		data[px_pos].b = sdata[i].b;
	}
}

//Copies the top left most pixel back across the cell
//requirements:
//   blockDim.x == c, blockDim.y == 1
//   gridDim.x == num_cells_x , gridDim.y == height
__global__ void row_scatter(rgb* data, unsigned int width, unsigned int height) {
	extern __shared__ rgb sdata[];

	unsigned int px_x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int px_y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int px_pos = px_x + (px_y*width);

	unsigned int i = threadIdx.x + (threadIdx.y * blockDim.x);

	if (threadIdx.x == 0) {
		sdata[i].r = data[px_pos].r;
		sdata[i].g = data[px_pos].g;
		sdata[i].b = data[px_pos].b;
	}

	__syncthreads();
	for (unsigned int stride = 1; stride < blockDim.x; stride <<= 1) {
		if (threadIdx.x < stride && px_x + stride < width && px_y < height) {
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

//Copies the top row down acrss the cell
//requirements:
//   blockDim.x == 1, blockDim.y == c
//   gridDim.x == width , gridDim.y == num_cells_y
__global__ void col_scatter(rgb* data, unsigned int width, unsigned int height) {
	extern __shared__ rgb sdata[];
	//load sdata
	unsigned int px_x = (blockIdx.x*blockDim.x) + threadIdx.x;
	unsigned int px_y = (blockIdx.y*blockDim.y) + threadIdx.y;
	unsigned int px_pos = px_x + (px_y*width);

	unsigned int i = threadIdx.y + (threadIdx.x * blockDim.y);

	if (threadIdx.y == 0) {
		sdata[i].r = data[px_pos].r;
		sdata[i].g = data[px_pos].g;
		sdata[i].b = data[px_pos].b;
	}

	__syncthreads();
	for (unsigned int stride = 1; stride < blockDim.y; stride <<= 1) {
		if (threadIdx.y < stride && px_y + stride < height && px_x < width) {
			sdata[i + stride].r = sdata[i].r;
			sdata[i + stride].g = sdata[i].g;
			sdata[i + stride].b = sdata[i].b;
		}
		__syncthreads();
	}

	if (px_y < height && px_x < width) {
		data[px_pos].r = sdata[i].r;
		data[px_pos].g = sdata[i].g;
		data[px_pos].b = sdata[i].b;
	}
}

//computes a single average for the whole image
//requirements:
//   blockDim.x == num_cells_x/gridDim.x, blockDim.y == num_cells_y/gridDim.y
__global__ void global_avg(const rgb* data, rgb* global_avg, unsigned int width, unsigned int height, unsigned int c) {
	extern __shared__ rgb sdata[];
	unsigned int px_x = (blockIdx.x*blockDim.x) + threadIdx.x;
	unsigned int px_y = (blockIdx.y*blockDim.y) + threadIdx.y;
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
		//top left exists on all cells, so no need to check edge case
		if (threadIdx.y < stride && px_y + stride < height) {
			sdata[i].r = ((unsigned int)sdata[i].r + (unsigned int)sdata[i + (stride*blockDim.x)].r) / 2;
			sdata[i].g = ((unsigned int)sdata[i].g + (unsigned int)sdata[i + (stride*blockDim.x)].g) / 2;
			sdata[i].b = ((unsigned int)sdata[i].b + (unsigned int)sdata[i + (stride*blockDim.x)].b) / 2;
		}
		__syncthreads();
	}

	if (i == 0 && px_x < width && px_y < height) {
		global_avg[blockIdx.x + blockIdx.y*gridDim.x].r = sdata[i].r;
		global_avg[blockIdx.x + blockIdx.y*gridDim.x].g = sdata[i].g;
		global_avg[blockIdx.x + blockIdx.y*gridDim.x].b = sdata[i].b;
	}
}

int global_avg_sm(int blockSize) {
	return blockSize * sizeof(rgb);
}

void run_cuda(rgb* data, unsigned int width, unsigned int height, unsigned int wb_width, unsigned int wb_height, unsigned int c) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//starting cpu based timing here
	double begin = omp_get_wtime();

	int num_cells_x = wb_width;
	int num_cells_y = wb_height;

	rgb* gpu_global_avg;
	rgb* gpu_data;

	//cudaOccupancyMaxPotentialBlockSizeVariableSMem was being used here initially,
	//but the averaging kernel is limited by blocks, so we actually want a fairly small block size
	//even with a blocksize of only 4 it still is limited by blocks so it might be better
	//to use the cpu for this
	int avg_block_size = 4;

	int avg_grid_dim = (num_cells_x*num_cells_y + (avg_block_size-1)) / avg_block_size;

	rgb* pre_summed_avgs = (rgb*)malloc(avg_grid_dim * sizeof(rgb));
	if (pre_summed_avgs == NULL) {
		fprintf(stderr, "Could not allocate enough memory");
		exit(EXIT_FAILURE);
	}

	cuda_check_error(cudaMalloc((void**)&gpu_global_avg, avg_grid_dim * sizeof(rgb)));
	cuda_check_error(cudaMalloc((void**)&gpu_data, width*height * sizeof(rgb)));
	cuda_check_error(cudaMemcpy(gpu_data, data, width*height * sizeof(rgb), cudaMemcpyHostToDevice));


	//run kernel code
	cuda_check_error(cudaEventRecord(start));
	//scale the overall block dimension by c to get as large a block as possible while keeping under
	//the limit for shared memory as best we can
	//56 is based on the max shared memory size for most cuda devices
	int sub_block_size = (56 + (c - 1)) / c;
	{
		row_reduction << < dim3(num_cells_x, height/sub_block_size, 1), dim3(c, sub_block_size, 1), sub_block_size * c * sizeof(rgb) >> > (gpu_data, width, height);
		cuda_check_error(cudaGetLastError());
		fprintf(stderr, "row_reduction done\n");
		col_reduction << < dim3(width/ sub_block_size, num_cells_y, 1), dim3(sub_block_size, c, 1), sub_block_size * c * sizeof(rgb) >> > (gpu_data, width, height);
		cuda_check_error(cudaGetLastError());
		fprintf(stderr, "col_reduction done\n");
	}
	{
		col_scatter << < dim3(width/ sub_block_size, num_cells_y, 1), dim3(sub_block_size, c, 1), sub_block_size * c * sizeof(rgb) >> > (gpu_data, width, height);
		cuda_check_error(cudaGetLastError());
		fprintf(stderr, "col_scatter done\n");
		row_scatter << < dim3(num_cells_x, height/ sub_block_size, 1), dim3(c, sub_block_size, 1), sub_block_size * c * sizeof(rgb) >> > (gpu_data, width, height);
		cuda_check_error(cudaGetLastError());
		fprintf(stderr, "row_scatter done\n");
	}
	{
		int block_x = sqrt(avg_block_size);
		int block_y = block_x;
		int grid_dim_x = (num_cells_x + (block_x - 1)) / block_x;
		int grid_dim_y = (num_cells_y + (block_y - 1)) / block_y;
		global_avg << < dim3(grid_dim_x, grid_dim_y, 1), dim3(block_x, block_y, 1), block_x * block_y * sizeof(rgb) >> > (gpu_data, gpu_global_avg, width, height, c);
		cuda_check_error(cudaGetLastError());
	}
	cuda_check_error(cudaDeviceSynchronize());

	cuda_check_error(cudaMemcpy(data, gpu_data, width*height * sizeof(rgb), cudaMemcpyDeviceToHost));
	cuda_check_error(cudaMemcpy(pre_summed_avgs, gpu_global_avg, avg_grid_dim * sizeof(rgb), cudaMemcpyDeviceToHost));

	big_rgb global_avg = { 0,0,0 };

	for (int i = 0; i < avg_grid_dim; i++) {
		global_avg.r += pre_summed_avgs[i].r;
		global_avg.g += pre_summed_avgs[i].g;
		global_avg.b += pre_summed_avgs[i].b;
	}

	free(pre_summed_avgs);

	global_avg.r /= avg_grid_dim;
	global_avg.g /= avg_grid_dim;
	global_avg.b /= avg_grid_dim;

	cuda_check_error(cudaFree(gpu_global_avg));
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