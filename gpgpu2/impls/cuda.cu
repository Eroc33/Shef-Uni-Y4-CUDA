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
	extern __shared__ unsigned char sdata[];

	unsigned char* r = &sdata[0 * blockDim.x];
	unsigned char* g = &sdata[1 * blockDim.x];
	unsigned char* b = &sdata[2 * blockDim.x];

	unsigned int y = blockIdx.y;
	unsigned int cell_start = blockIdx.x*blockDim.x;
	unsigned int px_x = cell_start + threadIdx.x;
	unsigned int y_offset = (y*width);

	if (px_x < width) {
		r[threadIdx.x] = data[px_x + y_offset].r;
		g[threadIdx.x] = data[px_x + y_offset].g;
		b[threadIdx.x] = data[px_x + y_offset].b;
	}
	__syncthreads();
	for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride && px_x+stride < width) {
			r[threadIdx.x] = ((unsigned int)r[threadIdx.x] + (unsigned int)r[threadIdx.x + stride]) / 2;
			g[threadIdx.x] = ((unsigned int)g[threadIdx.x] + (unsigned int)g[threadIdx.x + stride]) / 2;
			b[threadIdx.x] = ((unsigned int)b[threadIdx.x] + (unsigned int)b[threadIdx.x + stride]) / 2;
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		data[px_x + y_offset].r = r[threadIdx.x];
		data[px_x + y_offset].g = g[threadIdx.x];
		data[px_x + y_offset].b = b[threadIdx.x];
	}
}

//col_reduction, similar to row_reduction but reduce the already reduced rows into columns
__global__ void col_reduction(rgb* data, unsigned int width, unsigned int height, unsigned int c) {
	extern __shared__ unsigned char sdata[];

	unsigned char* r = &sdata[0 * blockDim.x];
	unsigned char* g = &sdata[1 * blockDim.x];
	unsigned char* b = &sdata[2 * blockDim.x];

	//load sdata
	unsigned int x = blockIdx.y*c;
	unsigned int cell_start_y = blockIdx.x*blockDim.x;
	unsigned int px_y = cell_start_y + threadIdx.x;
	unsigned int px_pos = x+(px_y*width);
	if (px_y < height) {
		r[threadIdx.x] = data[px_pos].r;
		g[threadIdx.x] = data[px_pos].g;
		b[threadIdx.x] = data[px_pos].b;
	}
	__syncthreads();
	for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride && px_y+stride < height) {
			r[threadIdx.x] = ((unsigned int)r[threadIdx.x] + (unsigned int)r[threadIdx.x + stride]) / 2;
			g[threadIdx.x] = ((unsigned int)g[threadIdx.x] + (unsigned int)g[threadIdx.x + stride]) / 2;
			b[threadIdx.x] = ((unsigned int)b[threadIdx.x] + (unsigned int)b[threadIdx.x + stride]) / 2;
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		data[px_pos].r = r[threadIdx.x];
		data[px_pos].g = g[threadIdx.x];
		data[px_pos].b = b[threadIdx.x];
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
    for (unsigned int cy = cell_y; cy < cell_y+cell_height; cy++) {
        for (unsigned int cx = cell_x; cx < cell_x+cell_width; cx++) {
            data[(cy*width) + cx] = out;
        }
    }
}

void run_cuda(big_rgb* work_buffer, rgb* data, unsigned int width, unsigned int height, unsigned int wb_width, unsigned int wb_height, unsigned int c){
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
    dim3 blocksPerGrid((wb_width+(32 - 1))/32,(wb_height+(32 - 1))/32,1);
    dim3 threadsPerBlock(32,32,1);

	cuda_check_error(cudaEventRecord(start));
	row_reduction <<< dim3(num_cells_x,height,1), c, c * sizeof(rgb) >>> (gpu_data, width, height);
	cuda_check_error(cudaGetLastError());
	col_reduction <<< dim3(num_cells_y,width,1), c, c * sizeof(rgb) >>> (gpu_data, width, height, c);
	cuda_check_error(cudaGetLastError());
	scatter<<<blocksPerGrid, threadsPerBlock>>>(gpu_data, width, height, wb_width, wb_height, c);
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
	printf("CUDA mode execution time took %d s and %dms (%f ms as measured by cuda)\n",(int)s,(int)ms,cudaMs);
}