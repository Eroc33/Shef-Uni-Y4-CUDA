#include "rgb.h"
void run_openmp(big_rgb* work_buffer, rgb* data, unsigned int width, unsigned int height, unsigned int wb_width, unsigned int wb_height, unsigned int cell_size);
void run_cpu(big_rgb* work_buffer, rgb* data, unsigned int width, unsigned int height, unsigned int wb_width, unsigned int wb_height, unsigned int cell_size);
void run_cuda(big_rgb* work_buffer, rgb* data, unsigned int width, unsigned int height, unsigned int wb_width, unsigned int wb_height, unsigned int cell_size);