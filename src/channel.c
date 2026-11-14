#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc,char **argv){MPI_Init(&argc,&argv);int rank,size;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);if(size!=2){if(!rank)fprintf(stderr,"Потрібно рівно два процеси\n");MPI_Abort(MPI_COMM_WORLD,2);}int repeats=argc>1?atoi(argv[1]):1000;if(repeats<2)repeats=2;for(size_t n=8;n<=4*1024*1024;n*=2){char *a=malloc(n),*b=malloc(n);memset(a,rank,n);MPI_Barrier(MPI_COMM_WORLD);for(int k=0;k<20;k++)MPI_Sendrecv(a,n,MPI_BYTE,1-rank,9,b,n,MPI_BYTE,1-rank,9,MPI_COMM_WORLD,MPI_STATUS_IGNORE);MPI_Barrier(MPI_COMM_WORLD);double begin=MPI_Wtime();for(int k=0;k<repeats;k++)MPI_Sendrecv(a,n,MPI_BYTE,1-rank,9,b,n,MPI_BYTE,1-rank,9,MPI_COMM_WORLD,MPI_STATUS_IGNORE);double elapsed=(MPI_Wtime()-begin)/repeats,worst;MPI_Reduce(&elapsed,&worst,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);if(!rank)printf("канал байти=%zu повтори=%d час=%.12f\n",n,repeats,worst);free(a);free(b);}MPI_Finalize();}
