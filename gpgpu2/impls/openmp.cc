#include "../rgb.h"
#include <omp.h>
#include <math.h>
#include <stdio.h>

void run_openmp(big_rgb* work_buffer, rgb* data, unsigned int width, unsigned int height, unsigned int wb_width, unsigned int wb_height, unsigned int c){
	//starting timing here
	double begin = omp_get_wtime();

	big_rgb global_avg = {0,0,0};

	int y;
	#pragma omp for
	//loop over all work buffer cells
	for (y = 0; y<wb_height; y++) {
		//cell height is input cell size
		unsigned int cell_height = c;
		//unless this is the last row of cells, and c does not evenly divide height
		if (height%cell_height != 0 && y + 1 == wb_height) {
			cell_height = height % cell_height;
		}
		unsigned int cell_y = y*c;
		for (unsigned int x = 0; (unsigned int)x<wb_width; x++) {
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
					big_rgb_add_assign(&work_buffer[y*wb_width+x], &data[i]);
					//cell sum
					big_rgb_add_assign(&avg, &data[i]);
				}
			}

			big_rgb_div_assign(&avg, cell_size);
			rgb out = from_big_rgb(&avg);

			//copy to all ouput buffer cells
			for (unsigned int cy = 0; cy < cell_height; cy++) {
				for (unsigned int cx = 0; cx < cell_width; cx++) {
					int i = ((cell_y + cy)*height) + cell_x + cx;
					data[i] = out;
				}
			}
		}
	}
	for(int i=0;i<wb_width*wb_height;i++){
		global_avg.r += work_buffer[i].r;
		global_avg.g += work_buffer[i].g;
		global_avg.b += work_buffer[i].b;
	}
	//divide by count for global average
	big_rgb_div_assign(&global_avg, width*height);

	// Output the average colour value for the image
	printf("OPENMP Average image colour red = %u, green = %u, blue = %u \n",(unsigned char)global_avg.r,(unsigned char)global_avg.g,(unsigned char)global_avg.b);

	//end timing here
	double end = omp_get_wtime();
	double seconds = (end - begin);

	double s;
	double ms = modf(seconds,&s)*1000.0;
	printf("OPENMP mode execution time took %d s and %dms\n",(int)s,(int)ms);
}