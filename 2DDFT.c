//
//  main.cpp
//  LowPathFilter
//
//  Created by Ye Guo on 5/9/14.
//  Copyright (c) 2014 Ye Guo. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <memory.h>
#include <sys/time.h>
#include "mpi.h"

#define max(x, y) ((x>y) ? (x):(y))
#define min(x, y) ((x<y) ? (x):(y))

typedef struct{
    double real;
    double vir;
} ComplexNum;

int xdim;
int ydim;
int maxraw;
double radius;
unsigned char *image;

void ReadPGM(FILE*);
void WritePGM(FILE*);
void ForwardFFT(int reverse, int arrSize, ComplexNum *src, ComplexNum *fft);
void SwapQuadrants(ComplexNum *complexImg, ComplexNum *result);
void CopyResult(ComplexNum *src, ComplexNum *dest, int size);
void LowPathFilter(ComplexNum *complexImg);

int main(int argc, char * argv[])
{
    int numproces, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numproces);
    
    if (myid == 0) {
        
        if (argc != 3){
            printf("Usage: MyProgram <input_ppm> <output_ppm> \n");
            printf("       <input_ppm>: PGM file \n");
            printf("       <output_ppm>: PGM file \n");
            exit(0);
        }
        
        FILE *fp;
        
        /* begin reading PGM.... */
        printf("begin reading PGM.... \n");
        if ((fp=fopen(argv[1], "r"))==NULL){
            printf("read error...\n");
            exit(0);
        }
        
        ReadPGM(fp);
        
        double *real, *vir;
        double *real2, *vir2;
        
        int *reverse = (int *)malloc(4);
        int *length = (int *)malloc(4);
        int *ifDone = (int *)malloc(4);
        int *blockSize = (int *)malloc(4);
        int *leftSize = (int *)malloc(4);
        int *length2 = (int *)malloc(4);
        
        *reverse = 1;
        *ifDone = 1;
        
        ComplexNum *img = (ComplexNum *)malloc(xdim * ydim * 16);
        ComplexNum *tempImg = (ComplexNum *)malloc(xdim * ydim * 16);
        
        *blockSize = (ydim % (numproces - 1) == 0) ? ydim / (numproces - 1) : ydim / (numproces - 1) + 1;
        *length = *blockSize * xdim;
        real = (double *)malloc(*length * 8);
        vir = (double *)malloc(*length * 8);
        
        *leftSize = ydim - *blockSize * (numproces - 2);
        *length2 = xdim * *leftSize;
        real2 = (double *)malloc(*length2 * 8);
        vir2 = (double *)malloc(*length2 * 8);
        
        struct timeval t1, t2, total;
        gettimeofday(&t1, NULL);
        
        for (int task=1; task < numproces - 1; task++) {
            for (int i=0; i < *length; i++) {
                real[i] = image[(task - 1) * *length + i];
                vir[i] = 0;
            }
            
            MPI_Send(ifDone, 1, MPI_INT, task, 1, MPI_COMM_WORLD);  //if done
            MPI_Send(reverse, 1, MPI_INT, task, 1, MPI_COMM_WORLD);  //if reverse
            MPI_Send(length, 1, MPI_INT, task, 1, MPI_COMM_WORLD);  //data size
            MPI_Send(blockSize, 1, MPI_INT, task, 1, MPI_COMM_WORLD);  //block size
            MPI_Send(&real[0], *length, MPI_DOUBLE, task, 1, MPI_COMM_WORLD);  //send real data
            MPI_Send(&vir[0], *length, MPI_DOUBLE, task, 1, MPI_COMM_WORLD);  //send vir data
        }
        
        for (int i=0; i < *length2; i++) {
            real2[i] = image[(numproces - 2) * (*length) + i];
            vir2[i] = 0;
        }
        
        MPI_Send(ifDone, 1, MPI_INT, numproces - 1, 1, MPI_COMM_WORLD);  //if done
        MPI_Send(reverse, 1, MPI_INT, numproces - 1, 1, MPI_COMM_WORLD);  //if reverse
        MPI_Send(length2, 1, MPI_INT, numproces - 1, 1, MPI_COMM_WORLD);  //data size
        MPI_Send(leftSize, 1, MPI_INT, numproces - 1, 1, MPI_COMM_WORLD);  //block size
        MPI_Send(&real2[0], *length2, MPI_DOUBLE, numproces - 1, 1, MPI_COMM_WORLD);  //send real data
        MPI_Send(&vir2[0], *length2, MPI_DOUBLE, numproces - 1, 1, MPI_COMM_WORLD);  //send vir data
        
        for (int task=1; task < numproces - 1; task++) {
            MPI_Recv(real, *length, MPI_DOUBLE, task, 2, MPI_COMM_WORLD, NULL);  //receive real data
            MPI_Recv(vir, *length, MPI_DOUBLE, task, 2, MPI_COMM_WORLD, NULL);  //receive vir data
            
            for (int i=0; i < *length; i++) {
                img[(task - 1) * (*length) + i].real = real[i];
                img[(task - 1) * (*length) + i].vir = vir[i];
            }
        }
        
        MPI_Recv(real2, *length2, MPI_DOUBLE, numproces - 1, 2, MPI_COMM_WORLD, NULL);  //receive real data
        MPI_Recv(vir2, *length2, MPI_DOUBLE, numproces - 1, 2, MPI_COMM_WORLD, NULL);  //receive vir data
        
        
        for (int i=0; i < *length2; i++) {
            img[(numproces - 2) * (*length) + i].real = real2[i];
            img[(numproces - 2) * (*length) + i].vir = vir2[i];
        }
        
        *blockSize = (xdim % (numproces - 1) == 0) ? xdim / (numproces - 1) : xdim / (numproces - 1) + 1;
        *length = *blockSize * ydim;
        real = (double *)realloc(real, *length * 8);
        vir = (double *)realloc(vir, *length * 8);
        
        *leftSize = xdim - *blockSize * (numproces - 2);
        *length2 = *leftSize * ydim;
        real2 = (double *)realloc(real2, *length2 * 8);
        vir2 = (double *)realloc(vir2, *length2 * 8);
        
        for (int task=1; task < numproces - 1; task++) {
            for (int i=0; i < ydim; i++) {
                for (int j=0; j < *blockSize; j++) {
                    real[i + j * ydim] = img[i * xdim + j + (task - 1) * (*blockSize)].real;
                    vir[i + j * ydim] = img[i * xdim + j + (task - 1) * (*blockSize)].vir;
                }
            }
            
            MPI_Send(ifDone, 1, MPI_INT, task, 1, MPI_COMM_WORLD);  //if done
            MPI_Send(reverse, 1, MPI_INT, task, 1, MPI_COMM_WORLD);  //if reverse
            MPI_Send(length, 1, MPI_INT, task, 1, MPI_COMM_WORLD);  //data size
            MPI_Send(blockSize, 1, MPI_INT, task, 1, MPI_COMM_WORLD);  //block size
            MPI_Send(&real[0], *length, MPI_DOUBLE, task, 1, MPI_COMM_WORLD);  //send real data
            MPI_Send(&vir[0], *length, MPI_DOUBLE, task, 1, MPI_COMM_WORLD);  //send vir data
        }
        
        for (int i=0; i < ydim; i++) {
            for (int j=0; j < *leftSize; j++) {
                real2[i + j * ydim] = img[i * xdim + j + (numproces - 2) * (*blockSize)].real;
                vir2[i + j * ydim] = img[i * xdim + j + (numproces - 2) * (*blockSize)].vir;
            }
        }
        
        MPI_Send(ifDone, 1, MPI_INT, numproces - 1, 1, MPI_COMM_WORLD);  //if done
        MPI_Send(reverse, 1, MPI_INT, numproces - 1, 1, MPI_COMM_WORLD);  //if reverse
        MPI_Send(length2, 1, MPI_INT, numproces - 1, 1, MPI_COMM_WORLD);  //data size
        MPI_Send(leftSize, 1, MPI_INT, numproces - 1, 1, MPI_COMM_WORLD);  //block size
        MPI_Send(&real2[0], *length2, MPI_DOUBLE, numproces - 1, 1, MPI_COMM_WORLD);  //send real data
        MPI_Send(&vir2[0], *length2, MPI_DOUBLE, numproces - 1, 1, MPI_COMM_WORLD);  //send vir data
        
        for (int task=1; task < numproces - 1; task++) {
            MPI_Recv(real, *length, MPI_DOUBLE, task, 2, MPI_COMM_WORLD, NULL);  //receive real data
            MPI_Recv(vir, *length, MPI_DOUBLE, task, 2, MPI_COMM_WORLD, NULL);  //receive vir data
            
            for (int i=0; i < ydim; i++) {
                for (int j=0; j < *blockSize; j++) {
                    img[i * xdim + j + (task - 1) * (*blockSize)].real = real[i + j * ydim];
                    img[i * xdim + j + (task - 1) * (*blockSize)].vir = vir[i + j * ydim];
                }
            }
        }
        
        MPI_Recv(real2, *length2, MPI_DOUBLE, numproces - 1, 2, MPI_COMM_WORLD, NULL);  //receive real data
        MPI_Recv(vir2, *length2, MPI_DOUBLE, numproces - 1, 2, MPI_COMM_WORLD, NULL);  //receive vir data
        
        for (int i=0; i < ydim; i++) {
            for (int j=0; j < *leftSize; j++) {
                img[i * xdim + j + (numproces - 2) * (*blockSize)].real = real2[i + j * ydim];
                img[i * xdim + j + (numproces - 2) * (*blockSize)].vir = vir2[i + j * ydim];
            }
        }
        
        SwapQuadrants(img, tempImg);
        
        radius = sqrt(xdim/2 * xdim/2 + ydim/2 * ydim/2) / 2;  //you can modify the parameter radius here !!!
        LowPathFilter(img);
        
        SwapQuadrants(img, tempImg);
        
        *reverse = 0;
        
        for (int task=1; task < numproces - 1; task++) {
            for (int i=0; i < ydim; i++) {
                for (int j=0; j < *blockSize; j++) {
                    real[i + j * ydim] = img[i * xdim + j + (task - 1) * (*blockSize)].real;
                    vir[i + j * ydim] = img[i * xdim + j + (task - 1) * (*blockSize)].vir;
                }
            }
            
            MPI_Send(ifDone, 1, MPI_INT, task, 1, MPI_COMM_WORLD);  //if done
            MPI_Send(reverse, 1, MPI_INT, task, 1, MPI_COMM_WORLD);  //if reverse
            MPI_Send(length, 1, MPI_INT, task, 1, MPI_COMM_WORLD);  //data size
            MPI_Send(blockSize, 1, MPI_INT, task, 1, MPI_COMM_WORLD);  //block size
            MPI_Send(&real[0], *length, MPI_DOUBLE, task, 1, MPI_COMM_WORLD);  //send real data
            MPI_Send(&vir[0], *length, MPI_DOUBLE, task, 1, MPI_COMM_WORLD);  //send vir data
        }
        
        for (int i=0; i < ydim; i++) {
            for (int j=0; j < *leftSize; j++) {
                real2[i + j * ydim] = img[i * xdim + j + (numproces - 2) * (*blockSize)].real;
                vir2[i + j * ydim] = img[i * xdim + j + (numproces - 2) * (*blockSize)].vir;
            }
        }
        
        MPI_Send(ifDone, 1, MPI_INT, numproces - 1, 1, MPI_COMM_WORLD);  //if done
        MPI_Send(reverse, 1, MPI_INT, numproces - 1, 1, MPI_COMM_WORLD);  //if reverse
        MPI_Send(length2, 1, MPI_INT, numproces - 1, 1, MPI_COMM_WORLD);  //data size
        MPI_Send(leftSize, 1, MPI_INT, numproces - 1, 1, MPI_COMM_WORLD);  //block size
        MPI_Send(&real2[0], *length2, MPI_DOUBLE, numproces - 1, 1, MPI_COMM_WORLD);  //send real data
        MPI_Send(&vir2[0], *length2, MPI_DOUBLE, numproces - 1, 1, MPI_COMM_WORLD);  //send vir data
        
        for (int task=1; task < numproces - 1; task++) {
            MPI_Recv(real, *length, MPI_DOUBLE, task, 2, MPI_COMM_WORLD, NULL);  //receive real data
            MPI_Recv(vir, *length, MPI_DOUBLE, task, 2, MPI_COMM_WORLD, NULL);  //receive vir data
            
            for (int i=0; i < ydim; i++) {
                for (int j=0; j < *blockSize; j++) {
                    img[i * xdim + j + (task - 1) * (*blockSize)].real = real[i + j * ydim];
                    img[i * xdim + j + (task - 1) * (*blockSize)].vir = vir[i + j * ydim];
                }
            }
        }
        
        MPI_Recv(real2, *length2, MPI_DOUBLE, numproces - 1, 2, MPI_COMM_WORLD, NULL);  //receive real data
        MPI_Recv(vir2, *length2, MPI_DOUBLE, numproces - 1, 2, MPI_COMM_WORLD, NULL);  //receive vir data
        
        for (int i=0; i < ydim; i++) {
            for (int j=0; j < *leftSize; j++) {
                img[i * xdim + j + (numproces - 2) * (*blockSize)].real = real2[i + j * ydim];
                img[i * xdim + j + (numproces - 2) * (*blockSize)].vir = vir2[i + j * ydim];
            }
        }
        
        *blockSize = (ydim % (numproces - 1) == 0) ? ydim / (numproces - 1) : ydim / (numproces - 1) + 1;
        *length = *blockSize * xdim;
        real = (double *)realloc(real, *length * 8);
        vir = (double *)realloc(vir, *length * 8);
        
        *leftSize = ydim - *blockSize * (numproces - 2);
        *length2 = xdim * (*leftSize);
        real2 = (double *)realloc(real2, *length2 * 8);
        vir2 = (double *)realloc(vir2, *length2 * 8);
        
        for (int task=1; task < numproces - 1; task++) {
            for (int i=0; i < *length; i++) {
                real[i] = img[(task - 1) * *length + i].real;
                vir[i] = img[(task - 1) * *length + i].vir;
            }
            
            MPI_Send(ifDone, 1, MPI_INT, task, 1, MPI_COMM_WORLD);  //if done
            MPI_Send(reverse, 1, MPI_INT, task, 1, MPI_COMM_WORLD);  //if reverse
            MPI_Send(length, 1, MPI_INT, task, 1, MPI_COMM_WORLD);  //data size
            MPI_Send(blockSize, 1, MPI_INT, task, 1, MPI_COMM_WORLD);  //block size
            MPI_Send(&real[0], *length, MPI_DOUBLE, task, 1, MPI_COMM_WORLD);  //send real data
            MPI_Send(&vir[0], *length, MPI_DOUBLE, task, 1, MPI_COMM_WORLD);  //send vir data
        }
        
        for (int i=0; i < *length2; i++) {
            real2[i] = img[(numproces - 2) * (*length) + i].real;
            vir2[i] = img[(numproces - 2) * (*length) + i].vir;
        }
        
        MPI_Send(ifDone, 1, MPI_INT, numproces - 1, 1, MPI_COMM_WORLD);  //if done
        MPI_Send(reverse, 1, MPI_INT, numproces - 1, 1, MPI_COMM_WORLD);  //if reverse
        MPI_Send(length2, 1, MPI_INT, numproces - 1, 1, MPI_COMM_WORLD);  //data size
        MPI_Send(leftSize, 1, MPI_INT, numproces - 1, 1, MPI_COMM_WORLD);  //block size
        MPI_Send(&real2[0], *length2, MPI_DOUBLE, numproces - 1, 1, MPI_COMM_WORLD);  //send real data
        MPI_Send(&vir2[0], *length2, MPI_DOUBLE, numproces - 1, 1, MPI_COMM_WORLD);  //send vir data
        
        for (int task=1; task < numproces - 1; task++) {
            MPI_Recv(real, *length, MPI_DOUBLE, task, 2, MPI_COMM_WORLD, NULL);  //receive real data
            MPI_Recv(vir, *length, MPI_DOUBLE, task, 2, MPI_COMM_WORLD, NULL);  //receive vir data
            
            for (int i=0; i < *length; i++) {
                img[(task - 1) * (*length) + i].real = real[i];
                img[(task - 1) * (*length) + i].vir = vir[i];
            }
        }
        
        MPI_Recv(real2, *length2, MPI_DOUBLE, numproces - 1, 2, MPI_COMM_WORLD, NULL);  //receive real data
        MPI_Recv(vir2, *length2, MPI_DOUBLE, numproces - 1, 2, MPI_COMM_WORLD, NULL);  //receive vir data
        
        
        for (int i=0; i < *length2; i++) {
            img[(numproces - 2) * (*length) + i].real = real2[i];
            img[(numproces - 2) * (*length) + i].vir = vir2[i];
        }
        
        gettimeofday(&t2, NULL);
    	timersub(&t2, &t1, &total);
        
        printf("The total time cost is %f seconds\n", total.tv_sec + total.tv_usec/1000000.0);
        
        free(real);
        free(vir);
        free(real2);
        free(vir2);
        
        *ifDone = 0;
        for (int task=1; task < numproces; task++) {
            MPI_Send(ifDone, 1, MPI_INT, task, 1, MPI_COMM_WORLD);
        }
        
        
        for (int i=0; i<xdim * ydim; i++)
            image[i] = img[i].real;
        
        free(img);
        free(tempImg);
        
        /* Begin writing PGM.... */
        printf("Begin writing PGM.... \n");
        if ((fp=fopen(argv[2], "wb")) == NULL){
            printf("write pgm error....\n");
            exit(0);
        }
        
        WritePGM(fp);
        free(image);
    }
    
    if (myid > 0) {
        
        while (1) {
            int reverse, length, ifDone, blockSize, size;
            MPI_Recv(&ifDone, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, NULL); //if done
            
            if (ifDone != 1)
                break;
            
            MPI_Recv(&reverse, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, NULL);  //if reverse
            MPI_Recv(&length, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, NULL);  //data size
            MPI_Recv(&blockSize, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, NULL);  //block size
            
            size = length / blockSize;
            double *real = (double *)malloc(length * 8);
            double *vir = (double *)malloc(length * 8);
            ComplexNum *img = (ComplexNum *)malloc(size * 16);
            ComplexNum *result = (ComplexNum *)malloc(size * 16);
            
            MPI_Recv(&real[0], length, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, NULL); //receive real data
            MPI_Recv(&vir[0], length, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, NULL);  //receive vir data
            
            for (int i=0; i < blockSize; i++) {
                for (int j=0; j < size; j++) {
                    img[j].real = real[i * size + j];
                    img[j].vir = vir[i * size + j];
                }
                
                ForwardFFT(reverse, size, img, result);
                
                for (int j=0; j < size; j++) {
                    real[i * size + j] = result[j].real;
                    vir[i * size + j] = result[j].vir;
                }
            }
            
            MPI_Send(real, length, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);  //send real data
            MPI_Send(vir, length, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);  //send vir data
            
            free(real);
            free(vir);
            free(img);
            free(result);
        }
    }
    
    MPI_Finalize();
    
    return 0;
}

void LowPathFilter(ComplexNum *complexImg)
{
    double r;
    for (int i=0; i < ydim; i++) {
        for (int j=0; j < xdim; j++) {
            r = sqrt((i - ydim/2) * (i - ydim/2) + (j - xdim/2) * (j - xdim/2));
            if (r > radius) {
                complexImg[i * xdim + j].real = 0;
                complexImg[i * xdim + j].vir = 0;
            }
        }
    }
}

void CopyResult(ComplexNum *src, ComplexNum *dest, int size)
{
    for (int i=0; i < size; i++) {
        dest[i].real = src[i].real;
        dest[i].vir = src[i].vir;
    }
}

void SwapQuadrants(ComplexNum *complexImg, ComplexNum *result)
{
    int halfWidth = xdim / 2;
    int halfHeight = ydim / 2;
    
    for (int i=0; i < halfHeight; i++) {
        for (int j=0; j < halfWidth; j++) {
            result[i * xdim + j].real = complexImg[(i + halfHeight) * xdim + j + halfWidth].real;
            result[i * xdim + j].vir = complexImg[(i + halfHeight) * xdim + j + halfWidth].vir;
            result[(i + halfHeight) * xdim + j + halfWidth].real = complexImg[i * xdim + j].real;
            result[(i + halfHeight) * xdim + j + halfWidth].vir = complexImg[i * xdim + j].vir;
            result[i * xdim + j + halfWidth].real = complexImg[(i + halfHeight) * xdim + j].real;
            result[i * xdim + j + halfWidth].vir = complexImg[(i + halfHeight) * xdim + j].vir;
            result[(i + halfHeight) * xdim + j].real = complexImg[i * xdim + j + halfWidth].real;
            result[(i + halfHeight) * xdim + j].vir = complexImg[i * xdim + j + halfWidth].vir;
        }
    }
    
    CopyResult(result, complexImg, xdim * ydim);
}

void ForwardFFT(int reverse, int size, ComplexNum *src, ComplexNum *fft)
{
    double pi2 = 2.0 * M_PI;
    double a, ca, sa;
    double invs = 1.0 / size;
    
    for(int i = 0; i < size; i++) {
        fft[i].real = 0;
        fft[i].vir = 0;
        for(int j = 0; j < size; j++) {
            a = pi2 * i * j * invs;
            ca = cos(a);
            sa = (reverse == 1) ? sin(-a): sin(a);
            fft[i].real += src[j].real * ca - src[j].vir * sa;
            fft[i].vir += src[j].real * sa + src[j].vir * ca;
        }
        if(reverse != 1) {
            fft[i].real *= invs;
            fft[i].vir *= invs;
        }
    }
}

void ReadPGM(FILE* fp)
{
    int c;
    int i,j;
    int val;
    unsigned char *line;
    char buf[1024];
    
    
    while ((c=fgetc(fp)) == '#')
        fgets(buf, 1024, fp);
    ungetc(c, fp);
    if (fscanf(fp, "P%d\n", &c) != 1) {
        printf ("read error ....");
        exit(0);
    }
    if (c != 5 && c != 2) {
        printf ("read error ....");
        exit(0);
    }
    
    if (c==5) {
        while ((c=fgetc(fp)) == '#')
            fgets(buf, 1024, fp);
        ungetc(c, fp);
        if (fscanf(fp, "%d%d%d",&xdim, &ydim, &maxraw) != 3) {
            printf("failed to read width/height/max\n");
            exit(0);
        }
        printf("Width=%d, Height=%d \nMaximum=%d\n",xdim,ydim,maxraw);
        
        image = (unsigned char*)malloc(sizeof(unsigned char)*xdim*ydim);
        getc(fp);
        
        line = (unsigned char *)malloc(sizeof(unsigned char)*xdim);
        for (j=0; j<ydim; j++) {
            fread(line, 1, xdim, fp);
            for (i=0; i<xdim; i++) {
                image[j*xdim+i] = line[i];
            }
        }
        free(line);
        
    }
    
    else if (c==2) {
        while ((c=fgetc(fp)) == '#')
            fgets(buf, 1024, fp);
        ungetc(c, fp);
        if (fscanf(fp, "%d%d%d", &xdim, &ydim, &maxraw) != 3) {
            printf("failed to read width/height/max\n");
            exit(0);
        }
        printf("Width=%d, Height=%d \nMaximum=%d,\n",xdim,ydim,maxraw);
        
        image = (unsigned char*)malloc(sizeof(unsigned char)*xdim*ydim);
        getc(fp);
        
        for (j=0; j<ydim; j++)
            for (i=0; i<xdim; i++) {
                fscanf(fp, "%d", &val);
                image[j*xdim+i] = val;
            }
        
    }
    
    fclose(fp);
}

void WritePGM(FILE* fp)
{
    int i,j;
    fprintf(fp, "P5\n%d %d\n%d\n", xdim, ydim, 255);
    
    for (j=0; j<ydim; j++)
        for (i=0; i<xdim; i++) {
            fputc(image[j*xdim+i], fp);
        }
    
    fclose(fp);
}

