#include "rgb.h"
#pragma once
void read_ppm(char* file_name, rgb** data, unsigned int* width, unsigned int* height, int* binary);
int write_ppm(char* file_name, rgb* data, unsigned int width, unsigned int height, int binary);
