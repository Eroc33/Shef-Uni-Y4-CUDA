#include "../rgb.h"
#include <omp.h>
#include <math.h>
#include <stdio.h>

__global__ void kernel(rgb* data, big_rgb* work_buffer, unsigned int width, unsigned int height, unsigned int wb_width, unsigned wb_height, unsigned int c){

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
    big_rgb avg = { 0,0,0 };

    unsigned int cell_x = x*c;

    //averages
    for (unsigned int cy = 0; cy < cell_height; cy++) {
        for (unsigned int cx = 0; cx < cell_width; cx++) {
            int i = ((cell_y + cy)*height) + cell_x + cx;
            //partial global sum
            rgb_generic_add_assign(&work_buffer[y*wb_width+x], &data[i]);
            //cell sum
            rgb_generic_add_assign(&avg, &data[i]);
        }
    }

    rgb_generic_div_assign(&avg, cell_size);
    rgb out = {
        (unsigned char)avg.r,
        (unsigned char)avg.g,
        (unsigned char)avg.b
    };

    //copy to all ouput buffer cells
    for (unsigned int cy = 0; cy < cell_height; cy++) {
        for (unsigned int cx = 0; cx < cell_width; cx++) {
            int i = ((cell_y + cy)*height) + cell_x + cx;
            data[i] = out;
        }
    }
}

#define bail_on_cuda_error(err) _bail_on_cuda_error(err,__FILE__,__LINE__)

void _bail_on_cuda_error(cudaError_t err, const char* file, unsigned int line){
    if(err == cudaSuccess){
        return;
    }
    fprintf(stderr,"FATAL: Encountered cuda error: %s (%s) (%s,%u).\n Quitting.\n",cudaGetErrorName(err),cudaGetErrorString(err),file,line);
    exit(EXIT_FAILURE);
}

void run_cuda(big_rgb* work_buffer, rgb* data, unsigned int width, unsigned int height, unsigned int wb_width, unsigned int wb_height, unsigned int c){
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//starting cpu based timing here
	double begin = omp_get_wtime();

    big_rgb global_avg = {0,0,0};
    
    rgb* gpu_data;
    big_rgb* gpu_wb;

    bail_on_cuda_error(cudaMalloc((void**)&gpu_data,width*height*sizeof(rgb)));
    bail_on_cuda_error(cudaMalloc((void**)&gpu_wb,wb_width*wb_height*sizeof(big_rgb)));

    bail_on_cuda_error(cudaMemcpy(gpu_data, data, width*height*sizeof(rgb), cudaMemcpyHostToDevice));
    bail_on_cuda_error(cudaMemset(gpu_wb, 0, wb_width*wb_height*sizeof(rgb)));

    //run kernel code
    dim3 blocksPerGrid((wb_width+(32 - 1))/32,(wb_height+(32 - 1))/32,1);
    dim3 threadsPerBlock(32,32,1);

	cudaEventRecord(start);
    kernel<<<blocksPerGrid, threadsPerBlock>>>(gpu_data,gpu_wb,width,height,wb_width,wb_height,c);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
    bail_on_cuda_error(cudaGetLastError());

    bail_on_cuda_error(cudaMemcpy(work_buffer, gpu_wb, wb_width*wb_height*sizeof(big_rgb), cudaMemcpyDeviceToHost));

    //TODO: use a cuda reduction to speed this up & avoid copying wb to cpu
    for(int i=0;i<wb_width*wb_height;i++){
		global_avg.r += work_buffer[i].r;
		global_avg.g += work_buffer[i].g;
		global_avg.b += work_buffer[i].b;
	}
	//divide by count for global average
    big_rgb_div_assign(&global_avg, width*height);

    bail_on_cuda_error(cudaMemcpy(data, gpu_data, width*height*sizeof(rgb), cudaMemcpyDeviceToHost));
    
    bail_on_cuda_error(cudaFree(gpu_wb));
    bail_on_cuda_error(cudaFree(gpu_data));

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