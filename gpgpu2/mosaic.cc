#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "ppm.h"

#define FAILURE 0
#define SUCCESS !FAILURE

#define USER_NAME "aca15er"		//replace with your user name

void print_help();
int process_command_line(int argc, char *argv[]);

typedef enum MODE { CPU, OPENMP, CUDA, ALL } MODE;

unsigned int c = 0;
MODE execution_mode = CPU;
char* input_filename;
char* output_filename;
unsigned char output_binary = 1;

#include "impls.h"


int main(int argc, char *argv[]) {
	
	if (process_command_line(argc, argv) == FAILURE)
		return 1;

	
	//read input image file (either binary or plain text PPM) 
    rgb* data = NULL;
    unsigned int width, height;
    int binary;
    read_ppm(input_filename,&data,&width,&height,&binary);
    if(data == NULL){
        fprintf(stderr,"Error while reading input file");
        return 1;
    }
	//pixel size of workbuffer without scaling back up
	//+(c-1) causes rounding up, as integer divison always truncated towards 0 in c
	unsigned int wb_width = (width+(c-1))/c,
		wb_height = (height+(c-1))/c;
	printf("wb_width: %u, wb_height: %u\n",wb_width,wb_height);
	//create the scaled workbuffer
	big_rgb* work_buffer = (big_rgb*)malloc(sizeof(big_rgb)*wb_width*wb_height);
	//must zero the memory, otherwise uninitialized values can cause artifacting
	memset(work_buffer, 0, sizeof(big_rgb)*wb_width*wb_height);

	//execute the mosaic filter based on the mode
	switch (execution_mode){
		case (CPU) : {
			run_cpu(work_buffer,data,width,height,wb_width,wb_height,c);
			break;
		}
		case (OPENMP) : {
			run_openmp(work_buffer,data,width,height,wb_width,wb_height,c);
			break;
		}
		case (CUDA) : {
			run_cuda(work_buffer,data,width,height,wb_width,wb_height,c);
			break;
		}
		case (ALL) : {
			run_cpu(work_buffer,data,width,height,wb_width,wb_height,c);

			//reset work buffer between implementations
			memset(work_buffer, 0, sizeof(big_rgb)*wb_width*wb_height);

			run_openmp(work_buffer,data,width,height,wb_width,wb_height,c);
			break;
		}
	}

	//save the output image file (from last executed mode)
    write_ppm(output_filename,data,width,height,output_binary);
    
    //free image data
	free(work_buffer);
    free(data);

	return 0;
}

void print_help(){
	printf("mosaic_%s C M -i input_file -o output_file [options]\n", USER_NAME);

	printf("where:\n");
	printf("\tC              Is the mosaic cell size which should be any positive\n"
		   "\t               power of 2 number \n");
	printf("\tM              Is the mode with a value of either CPU, OPENMP, CUDA or\n"
		   "\t               ALL. The mode specifies which version of the simulation\n"
		   "\t               code should execute. ALL should execute each mode in\n"
		   "\t               turn.\n");
	printf("\t-i input_file  Specifies an input image file\n");
	printf("\t-o output_file Specifies an output image file which will be used\n"
		   "\t               to write the mosaic image\n");
	printf("[options]:\n");
	printf("\t-f ppm_format  PPM image output format either PPM_BINARY (default) or \n"
		   "\t               PPM_PLAIN_TEXT\n ");
}

int process_command_line(int argc, char *argv[]){
	if (argc < 7){
		fprintf(stderr, "Error: Missing program arguments. Correct usage is...\n");
		print_help();
		return FAILURE;
	}

	//first argument is always the executable name
    
	//read in the non optional command line arguments
	c = (unsigned int)atoi(argv[1]);
	if (c > 4096){
		fprintf(stderr, "Error: c must be less than or equal to 4096 and greater than 0");
		return FAILURE;
	}

	//read in the mode
    if (strcmp("CPU",argv[2]) == 0){
        execution_mode = CPU;
    }else if (strcmp("OPENMP",argv[2]) == 0){
        execution_mode = OPENMP;
    }else if (strcmp("CUDA",argv[2]) == 0){
        execution_mode = CUDA;
    }


	//read in the input image name
    input_filename = argv[4];

	//read in the output image name
    output_filename = argv[6];
    

	//read in any optional arguments
	if(argc >= 9 && strncmp("-f",argv[7],2) == 0){
		if(strncmp("PPM_BINARY",argv[8],10) == 0){
			output_binary = 1;
		}
		if(strncmp("PPM_PLAIN_TEXT",argv[8],14) == 0){
			output_binary = 0;
		}
	}

	return SUCCESS;
}
