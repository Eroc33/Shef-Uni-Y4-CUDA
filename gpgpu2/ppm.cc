#include "ppm.h"
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

int read_ascii_ppm(FILE* file, rgb* data, unsigned int width, unsigned int height);
int read_bin_ppm(FILE* file, rgb* data, unsigned int width, unsigned int height);
void discard_comments(FILE* file);
void read_ppm(char* file_name, rgb** data, unsigned int* width, unsigned int* height, int* binary){
    //open the file
    FILE* file = fopen(file_name,"rb");
    if(file == NULL){
        return;
    }

    //read the header
    char magic[2];
    unsigned int maxval;

    if(fscanf(file,"%2c ",magic) != 1){
        fclose(file);
        fprintf(stderr,"couldn't find magic number\n");
        return;
    }
    discard_comments(file);
    if(fscanf(file,"%d %d ",width,height) != 2){
        fclose(file);
        fprintf(stderr,"couldn't find width and height\n");
        return;
    }
    discard_comments(file);
    if(fscanf(file,"%d ",&maxval) != 1){
        fclose(file);
        fprintf(stderr,"couldn't find maxval\n");
        return;
    }
    discard_comments(file);

    //allocate the needed memory
    *data = (rgb*)malloc(sizeof(rgb)*(*width)*(*height));

    //dispatch to the correct implementation
    if(strncmp(magic,"P3",2) == 0){
        *binary = 0;
        if (read_ascii_ppm(file,*data,*width,*height) == EXIT_FAILURE){
            //on failure free buffer and null the pointer
            free(*data);
            *data = NULL;
        }
    }else if(strncmp(magic,"P6",2) == 0){
        *binary = 1;
        if(read_bin_ppm(file,*data,*width,*height) == EXIT_FAILURE){
            //on failure free buffer and null the pointer
            free(*data);
            *data = NULL;
        }
    }
    //if neither magic matched, or file read failure occured data will be null, which the caller should check for

    fclose(file);
}
//discards any lines beginning with #
inline void discard_comments(FILE* file){
    while (fscanf(file,"#%*[^\n]") == 1) {
        fprintf(stderr,"Discarded comment\n");
    }
}

int read_ascii_ppm(FILE* file, rgb* data, unsigned int width, unsigned int height){
    printf("read_ascii_ppm: %d, %d\n",width,height);
    unsigned char r,g,b;
    for(unsigned int y=0;y<height;y++){
        for(unsigned int x=0;x<width;x++){
            if (fscanf(file,"%hhu %hhu %hhu ",&r,&g,&b)!=3){
                fprintf(stderr,"File has bad number of entries. Is the file corrupted?\n");
                return EXIT_FAILURE;
            }
            data[(y*width)+x].r = r;
            data[(y*width)+x].g = g;
            data[(y*width)+x].b = b;
        }
    }
    if(feof(file) != 0){
        fprintf(stderr,"File is not at the end. It may be corrupted, and output data may be incorrect.\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
int read_bin_ppm(FILE* file, rgb* data, unsigned int width, unsigned int height){
    printf("read_bin_ppm: %u, %u\n",width,height);
    unsigned char rgb[3];
    for(unsigned int y=0;y<height;y++){
        for(unsigned int x=0;x<width;x++){
            if(fread(rgb,sizeof(unsigned char),3,file) != 3){
                fprintf(stderr,"File has bad number of entries. Is the file corrupted?\n");
                return EXIT_FAILURE;
            }
            data[(y*width)+x].r = rgb[0];
            data[(y*width)+x].g = rgb[1];
            data[(y*width)+x].b = rgb[2];
        }
    }
    if(feof(file) != 0){
        fprintf(stderr,"File is not at the end. It may be corrupted, and output data may be incorrect.\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int write_ppm(char* file_name, rgb* data, unsigned int width, unsigned int height, int binary){
    //open the file
    FILE* file = fopen(file_name,"wb");
    if(file == NULL){
        return 1;
    }

    //write the header
    //write the magic
    if(binary){
        fprintf(file,"P6\n");
    }else{
        fprintf(file,"P3\n");
    }
    //write the size
    fprintf(file,"%d %d\n",width, height);
    //write the maxval
    fprintf(file, "255\n");

    //dispatch to the correct implementation
    if(binary){
        for(int y=0;y<height;y++){
            for(int x=0;x<width;x++){
                fwrite((void*)&data[(y*width)+x],sizeof(unsigned char), 3,file);
            }
        }
    }else{
        for(int y=0;y<height;y++){
            for(int x=0;x<width;x++){
                fprintf(file,"%u %u %u\t",data[(y*width)+x].r,data[(y*width)+x].g,data[(y*width)+x].b);

            }
            fprintf(file,"\n");
        }
    }
    fclose(file);
    return 0;
}
