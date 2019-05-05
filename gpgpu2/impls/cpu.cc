#include "../rgb.h"
#include <omp.h>
#include <math.h>
#include <stdio.h>

void run_cpu(big_rgb* work_buffer, rgb* data, unsigned int width, unsigned int height, unsigned int wb_width, unsigned int wb_height, unsigned int c){
	//starting timing here
	double begin = omp_get_wtime();
	big_rgb global_avg = {0,0,0};

	//sum cells into an appropriately sized grid
	for(int y=0;y<height;y++){
		int wb_y = y/c;
		assert(wb_y<wb_height);
		for(int x=0;x<width;x++){						
			int wb_x = x/c;
			assert(wb_x<wb_width);
			rgb* curr = &data[(y*width)+x];
			//global sum
			big_rgb_add_assign(&global_avg,curr);
			//cell sum
			big_rgb_add_assign(&work_buffer[(wb_y*wb_width)+wb_x],curr);
		}
	}
	//divide by count for global average
	big_rgb_div_assign(&global_avg,width*height);

	//compute cell-wise averages
	for(int y=0;y<wb_height;y++){
		//cell height is input cell size
		unsigned int cell_height = c;
		//unless this is the last row of cells, and c does not evenly divide height
		if (height%cell_height != 0 && y+1 == wb_height){
			//in which case it is wahtever the remainder is
			cell_height = height%cell_height;
		}
		for(int x=0;x<wb_width;x++){
			//cell width is input cell size
			unsigned int cell_width = c;
			//unless this is the last column of cells, and c does not evenly divide width
			if (width%cell_width != 0 && x+1 == wb_width){
				//in which case it is wahtever the remainder is
				cell_width = width%cell_width;
			}
		
			unsigned int cell_size = cell_width*cell_height;

			//divide cell by count for cell average
			big_rgb_div_assign(&work_buffer[(y*wb_width)+x],cell_size);
		}
	}

	//rescale to original image size, storing in original inut buffer for memory efficiency
	for(int y=0;y<height;y++){
		int wb_y = y/c;
		assert(wb_y<wb_height);
		for(int x=0;x<width;x++){						
			int wb_x = x/c;
			assert(wb_x<wb_width);
			data[(y*width)+x] = from_big_rgb(&work_buffer[(wb_y*wb_width)+wb_x]);
		}
	}

	// Output the average colour value for the image
	printf("CPU Average image colour red = %u, green = %u, blue = %u \n",(unsigned char)global_avg.r,(unsigned char)global_avg.g,(unsigned char)global_avg.b);

	//end timing here
	double end = omp_get_wtime();
	double seconds = (end - begin);

	double s;
	double ms = modf(seconds,&s)*1000.0;
	printf("CPU mode execution time took %d s and %dms\n",(int)s,(int)ms);
}