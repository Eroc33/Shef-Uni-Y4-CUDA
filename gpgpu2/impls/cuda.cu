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

int global_avg_sm(int blockSize) {
	return blockSize * sizeof(rgb);
}

void CUDART_CB cpu_avg(void* user_data) {
	void** user_data_array = (void**)user_data;
	big_rgb* global_avg = (big_rgb*)user_data_array[0];
	rgb* pre_summed_avgs = (rgb*)user_data_array[1];
	int avg_block_size = *(int*)user_data_array[2];

	for (int i = 0; i < avg_block_size; i++) {
		global_avg->r += pre_summed_avgs[i].r;
		global_avg->g += pre_summed_avgs[i].g;
		global_avg->b += pre_summed_avgs[i].b;
	}

	free(pre_summed_avgs);

	global_avg->r /= avg_block_size;
	global_avg->g /= avg_block_size;
	global_avg->b /= avg_block_size;
}

void run_cuda(rgb* data, unsigned int width, unsigned int height, unsigned int wb_width, unsigned int wb_height, unsigned int c) {
	int cudaDevice;
	int maxThreadsPerBlock;
	cuda_check_error(cudaGetDevice(&cudaDevice));
	cuda_check_error(cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, cudaDevice));

	cudaStream_t stream;
	cuda_check_error(cudaStreamCreate(&stream));

	//starting cpu based timing here
	double begin = omp_get_wtime();

	int num_cells_x = (width + (c - 1)) / c;
	int num_cells_y = (height + (c - 1)) / c;

	int avg_block_size;
	int avg_grid_size;
	cuda_check_error(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&avg_grid_size, &avg_block_size, &global_avg, global_avg_sm, 0));

	big_rgb host_global_avg = { 0,0,0 };

	rgb* gpu_global_avg;
	rgb* gpu_data;
	int avg_grid_dim = (num_cells_x*num_cells_y / avg_block_size);
	rgb* pre_summed_avgs = (rgb*)malloc(avg_grid_dim * sizeof(rgb));
	if (pre_summed_avgs == NULL) {
		fprintf(stderr, "Could not allocate enough memory");
		exit(EXIT_FAILURE);
	}

	//copy data
	cuda_check_error(cudaMalloc((void**)&gpu_global_avg, avg_grid_dim * sizeof(rgb)));
	cuda_check_error(cudaMalloc((void**)&gpu_data, width*height * sizeof(rgb)));

	//setup graph
	cudaGraph_t graph;
	cudaGraphNode_t cpy_data_in_node, cpy_data_out_node, gather_node, scatter_node, avg_node, cpy_avg_out, cpu_avg_node;

	cuda_check_error(cudaGraphCreate(&graph, 0));

	{
		cudaMemcpy3DParms params = { 0 };
		params.srcArray = NULL;
		params.srcPos = make_cudaPos(0, 0, 0);
		params.srcPtr =
			make_cudaPitchedPtr(data, sizeof(rgb) * width * height, width * height, 1);
		params.dstArray = NULL;
		params.dstPos = make_cudaPos(0, 0, 0);
		params.dstPtr = make_cudaPitchedPtr(gpu_data, sizeof(rgb) * width * height, width * height, 1);
		params.extent = make_cudaExtent(sizeof(rgb) * width * height, 1, 1);
		params.kind = cudaMemcpyHostToDevice;

		cuda_check_error(cudaGraphAddMemcpyNode(&cpy_data_in_node, graph, NULL, 0, &params));
	}

	{
		void *args[3] = { &gpu_data, &width, &height };

		cudaGraphNode_t gather_dependencies[1] = { cpy_data_in_node };

		cudaKernelNodeParams gather_params = {0};
		gather_params.func = &gather;
		gather_params.gridDim = dim3(num_cells_x, num_cells_y, 1);
		gather_params.blockDim = dim3(c, c, 1);
		gather_params.sharedMemBytes = c * c * sizeof(rgb);
		gather_params.kernelParams = args;
		gather_params.extra = NULL;

		cuda_check_error(cudaGraphAddKernelNode(&gather_node, graph, gather_dependencies, 1, &gather_params));
	}

	{
		void *args[3] = { &gpu_data, &width, &height };

		cudaGraphNode_t dependencies[1] = { gather_node };

		cudaKernelNodeParams params = { 0 };
		params.func = &scatter;
		params.gridDim = dim3(num_cells_x, num_cells_y, 1);
		params.blockDim = dim3(c, c, 1);
		params.sharedMemBytes = c * c * sizeof(rgb);
		params.kernelParams = args;
		params.extra = NULL;

		cuda_check_error(cudaGraphAddKernelNode(&scatter_node, graph, dependencies, 1, &params));
	}

	{

		int block_x = sqrt(avg_block_size);
		int block_y = block_x;
		int grid_dim_x = num_cells_x + (block_x - 1) / block_x;
		int grid_dim_y = num_cells_y + (block_y - 1) / block_y;

		void *args[5] = { &gpu_data, &gpu_global_avg, &width, &height, &c };

		cudaGraphNode_t dependencies[1] = { gather_node };

		cudaKernelNodeParams params = { 0 };
		params.func = &global_avg;
		params.gridDim = dim3(grid_dim_x, grid_dim_y, 1);
		params.blockDim = dim3(block_x, block_y, 1);
		params.sharedMemBytes = block_x* block_y * sizeof(rgb);
		params.kernelParams = args;
		params.extra = NULL;

		fprintf(stderr, "avg_block_size: %d, block_x: %d, block_y: %d, grid_dim_x: %d, grid_dim_y: %d, required_sm: %d\n", avg_block_size, block_x, block_y, grid_dim_x, grid_dim_y, block_x * block_y * sizeof(rgb));

		cuda_check_error(cudaGraphAddKernelNode(&avg_node, graph, dependencies, 1, &params));
	}

	{
		cudaGraphNode_t dependencies[1] = { scatter_node };

		cudaMemcpy3DParms params = { 0 };
		params.srcArray = NULL;
		params.srcPos = make_cudaPos(0, 0, 0);
		params.srcPtr =
			make_cudaPitchedPtr(gpu_data, sizeof(rgb) * width * height, width * height, 1);
		params.dstArray = NULL;
		params.dstPos = make_cudaPos(0, 0, 0);
		params.dstPtr = make_cudaPitchedPtr(data, sizeof(rgb) * width * height, width * height, 1);
		params.extent = make_cudaExtent(sizeof(rgb) * width * height, 1, 1);
		params.kind = cudaMemcpyDeviceToHost;

		cuda_check_error(cudaGraphAddMemcpyNode(&cpy_data_out_node, graph, dependencies, 1, &params));
	}

	{
		cudaGraphNode_t dependencies[1] = { avg_node };

		cudaMemcpy3DParms params = { 0 };
		params.srcArray = NULL;
		params.srcPos = make_cudaPos(0, 0, 0);
		params.srcPtr = make_cudaPitchedPtr(gpu_global_avg, sizeof(rgb) * avg_grid_dim, avg_grid_dim, 1);
		params.dstArray = NULL;
		params.dstPos = make_cudaPos(0, 0, 0);
		params.dstPtr = make_cudaPitchedPtr(pre_summed_avgs, sizeof(rgb) * avg_grid_dim, avg_grid_dim, 1);
		params.extent = make_cudaExtent(sizeof(rgb) * avg_grid_dim, 1, 1);
		params.kind = cudaMemcpyDeviceToHost;

		cuda_check_error(cudaGraphAddMemcpyNode(&cpy_avg_out, graph, dependencies, 1, &params));
	}

	{
		cudaGraphNode_t dependencies[1] = { cpy_avg_out };

		void* user_data[3] = {&host_global_avg, pre_summed_avgs, &avg_block_size };

		cudaHostNodeParams params = { 0 };
		params.fn = cpu_avg;
		params.userData = user_data;
		
		cuda_check_error(cudaGraphAddHostNode(&cpu_avg_node, graph, dependencies, 1, &params));
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaGraphExec_t executable_graph;

	cuda_check_error(cudaGraphInstantiate(&executable_graph, graph, NULL, NULL, 0));

	//run kernel code
	cuda_check_error(cudaEventRecord(start));
	cuda_check_error(cudaGraphLaunch(executable_graph, stream));
	cuda_check_error(cudaStreamSynchronize(stream));

	cuda_check_error(cudaFree(gpu_global_avg));
	cuda_check_error(cudaFree(gpu_data));

	cuda_check_error(cudaEventRecord(stop));
	cuda_check_error(cudaEventSynchronize(stop));

	cudaStreamDestroy(stream);

	// Output the average colour value for the image
	printf("CUDA Average image colour red = %u, green = %u, blue = %u \n", host_global_avg.r, host_global_avg.g, host_global_avg.b);

	//end timing here
	double end = omp_get_wtime();
	double seconds = (end - begin);

	float cudaMs;

	cudaEventElapsedTime(&cudaMs, start, stop);

	double s;
	double ms = modf(seconds, &s)*1000.0;
	printf("CUDA mode execution time took %d s and %f ms (%f ms as measured by cuda)\n", (int)s, ms, cudaMs);
}