#include <mpi.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static uint64_t mix(uint64_t x) { x ^= x >> 30; x *= UINT64_C(0xbf58476d1ce4e5b9); x ^= x >> 27; x *= UINT64_C(0x94d049bb133111eb); return x ^ (x >> 31); }
static unsigned char initial(uint64_t seed, int i, int j, double density) {
    uint64_t x = mix(seed ^ (uint64_t)(i+1)*UINT64_C(0x9e3779b97f4a7c15) ^ (uint64_t)(j+1)*UINT64_C(0xd1b54a32d192ed03));
    unsigned char v = (double)(x >> 11) * 0x1.0p-53 < density ? 1 : 0;
    return j == 0 && v ? 2 : v;
}
static int number(const char *s) { char *end; long v=strtol(s,&end,10); if(*end || v<0 || v>100000000) {fprintf(stderr,"Некоректне ціле число: %s\n",s); MPI_Abort(MPI_COMM_WORLD,2);} return (int)v; }
static void split(int n,int p,int r,int *start,int *count) { *start=(int)((long long)n*r/p); *count=(int)((long long)n*(r+1)/p)-*start; }
static uint64_t digest(const unsigned char *a,size_t n) { uint64_t h=UINT64_C(14695981039346656037); for(size_t i=0;i<n;i++){h^=a[i];h*=UINT64_C(1099511628211);}return h; }
static void save(const char *path,const unsigned char *a,int n) {if(!path || !*path)return; FILE *f=fopen(path,"wb");if(!f){perror(path);MPI_Abort(MPI_COMM_WORLD,5);}if(fwrite(a,1,(size_t)n*n,f)!=(size_t)n*n){perror("Запис");MPI_Abort(MPI_COMM_WORLD,5);}fclose(f);}
static void exchange(unsigned char *a,int rows,int cols,int up,int down,int method,MPI_Comm comm) {
    unsigned char *top=a+cols,*bottom=a+(size_t)rows*cols;
    if(method==0) {
        MPI_Sendrecv(top,cols,MPI_BYTE,up,11,a,cols,MPI_BYTE,up,12,comm,MPI_STATUS_IGNORE);
        MPI_Sendrecv(bottom,cols,MPI_BYTE,down,12,a+(size_t)(rows+1)*cols,cols,MPI_BYTE,down,11,comm,MPI_STATUS_IGNORE);
    } else if(method==1) {
        MPI_Request req[4];
        MPI_Irecv(a,cols,MPI_BYTE,up,12,comm,&req[0]);
        MPI_Irecv(a+(size_t)(rows+1)*cols,cols,MPI_BYTE,down,11,comm,&req[1]);
        MPI_Isend(top,cols,MPI_BYTE,up,11,comm,&req[2]);
        MPI_Isend(bottom,cols,MPI_BYTE,down,12,comm,&req[3]);
        MPI_Waitall(4,req,MPI_STATUSES_IGNORE);
    } else {
        MPI_Send(top,cols,MPI_BYTE,up,11,comm);
        MPI_Send(bottom,cols,MPI_BYTE,down,12,comm);
        MPI_Recv(a,cols,MPI_BYTE,up,12,comm,MPI_STATUS_IGNORE);
        MPI_Recv(a+(size_t)(rows+1)*cols,cols,MPI_BYTE,down,11,comm,MPI_STATUS_IGNORE);
    }
}
static void update(const unsigned char *a,unsigned char *b,int rows,int cols,int first,int last,int threads) {
    (void)rows;
    #pragma omp parallel for if(threads>1) num_threads(threads) schedule(static)
    for(int i=first;i<=last;i++)for(int j=0;j<cols;j++) {
        int x=i*cols+j;unsigned char v=a[x];
        if(v==2)b[x]=3;
        else if(v==1 && (a[x-cols]==2 || a[x+cols]==2 || (j>0 && a[x-1]==2) || (j+1<cols && a[x+1]==2)))b[x]=2;
        else b[x]=v;
    }
}
static void stripe(int argc,char **argv,int rank,int size) {
    int n=number(argv[1]); double d=atof(argv[2]);uint64_t seed=strtoull(argv[3],NULL,10);
    const char *method=argc>4?argv[4]:"ordered",*path=argc>5?argv[5]:"";
    int overlap=!strcmp(method,"overlap"), threads=!strcmp(method,"hybrid")?omp_get_max_threads():1;
    int how=!strcmp(method,"nonblocking")||overlap||!strcmp(method,"hybrid")?1:!strcmp(method,"bad")?2:0;
    if(n<1 || n<size || d<0 || d>1){if(rank==0)fprintf(stderr,"Потрібно N >= P та 0 <= d <= 1\n");MPI_Abort(MPI_COMM_WORLD,3);}
    int begin,rows;split(n,size,rank,&begin,&rows);
    size_t bytes=(size_t)(rows+2)*n; unsigned char *a=calloc(bytes,1),*b=calloc(bytes,1);
    if(!a||!b)MPI_Abort(MPI_COMM_WORLD,4);
    for(int i=1;i<=rows;i++)for(int j=0;j<n;j++)a[(size_t)i*n+j]=initial(seed,begin+i-1,j,d);
    int up=rank?rank-1:MPI_PROC_NULL,down=rank+1<size?rank+1:MPI_PROC_NULL;
    int steps=0,ever=0;long localburn=0;double ctime=0,comtime=0,rtime=0;
    MPI_Barrier(MPI_COMM_WORLD);double start=MPI_Wtime();
    while(1) {
        long count=0;
        for(int i=1;i<=rows;i++)for(int j=0;j<n;j++)count+=a[(size_t)i*n+j]==2;
        double t=MPI_Wtime();MPI_Allreduce(&count,&localburn,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);rtime+=MPI_Wtime()-t;
        if(localburn==0)break;
        t=MPI_Wtime();
        if(overlap) {
            MPI_Request req[4];
            MPI_Irecv(a,n,MPI_BYTE,up,12,MPI_COMM_WORLD,&req[0]);
            MPI_Irecv(a+(size_t)(rows+1)*n,n,MPI_BYTE,down,11,MPI_COMM_WORLD,&req[1]);
            MPI_Isend(a+n,n,MPI_BYTE,up,11,MPI_COMM_WORLD,&req[2]);
            MPI_Isend(a+(size_t)rows*n,n,MPI_BYTE,down,12,MPI_COMM_WORLD,&req[3]);
            comtime+=MPI_Wtime()-t;
            double inside=MPI_Wtime();
            if(rows>2)update(a,b,rows,n,2,rows-1,threads);
            ctime+=MPI_Wtime()-inside;
            double waiting=MPI_Wtime();
            MPI_Waitall(4,req,MPI_STATUSES_IGNORE);
            comtime+=MPI_Wtime()-waiting;
            t=MPI_Wtime();update(a,b,rows,n,1,1,threads);if(rows>1)update(a,b,rows,n,rows,rows,threads);ctime+=MPI_Wtime()-t;
        } else {
            exchange(a,rows,n,up,down,how,MPI_COMM_WORLD);
            comtime+=MPI_Wtime()-t;
            t=MPI_Wtime();update(a,b,rows,n,1,rows,threads);ctime+=MPI_Wtime()-t;
        }
        unsigned char *tmp=a;a=b;b=tmp;steps++;
    }
    double total=MPI_Wtime()-start;
    long burned=0;int reached=0;
    for(int i=1;i<=rows;i++)for(int j=0;j<n;j++){unsigned char v=a[(size_t)i*n+j];burned+=v==3;if(j==n-1 && v==3)reached=1;}
    long allburned=0;MPI_Reduce(&burned,&allburned,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&reached,&ever,1,MPI_INT,MPI_MAX,0,MPI_COMM_WORLD);
    unsigned char *full=NULL;int *counts=NULL,*offsets=NULL;
    if(rank==0){full=malloc((size_t)n*n);counts=malloc(size*sizeof(int));offsets=malloc(size*sizeof(int));for(int r=0;r<size;r++){int s,k;split(n,size,r,&s,&k);counts[r]=k*n;offsets[r]=s*n;}}
    MPI_Gatherv(a+n,rows*n,MPI_BYTE,full,counts,offsets,MPI_BYTE,0,MPI_COMM_WORLD);
    double own[4]={total,comtime,ctime,rtime},maximum[4];MPI_Reduce(own,maximum,4,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(rank==0){save(path,full,n);printf("версія=V1/V3 метод=%s N=%d P=%d d=%.4f seed=%llu кроки=%d згоріло=%ld досягнуто=%d hash=%016llx час=%.9f обмін=%.9f обчислення=%.9f редукція=%.9f\n",method,n,size,d,(unsigned long long)seed,steps,allburned,ever,(unsigned long long)digest(full,(size_t)n*n),maximum[0],maximum[1],maximum[2],maximum[3]);fflush(stdout);free(full);free(counts);free(offsets);}
    free(a);free(b);
}
static void sequential(int argc,char **argv,int rank,int size) {
    if(size!=1){if(rank==0)fprintf(stderr,"V0 потребує одного процесу\n");MPI_Abort(MPI_COMM_WORLD,4);}
    int n=number(argv[1]);double d=atof(argv[2]);uint64_t seed=strtoull(argv[3],NULL,10);const char *path=argc>4?argv[4]:"";
    size_t length=(size_t)n*n;unsigned char *a=malloc(length),*b=malloc(length);if(!a||!b)MPI_Abort(MPI_COMM_WORLD,3);
    for(int i=0;i<n;i++)for(int j=0;j<n;j++)a[(size_t)i*n+j]=initial(seed,i,j,d);
    MPI_Barrier(MPI_COMM_WORLD);double start=MPI_Wtime();int steps=0;
    while(1){long burning=0;for(size_t k=0;k<length;k++)burning+=a[k]==2;if(!burning)break;
        for(int i=0;i<n;i++)for(int j=0;j<n;j++){size_t x=(size_t)i*n+j;unsigned char v=a[x];b[x]=v==2?3:v==1 && ((i>0 && a[x-n]==2)||(i<n-1 && a[x+n]==2)||(j>0 && a[x-1]==2)||(j<n-1 && a[x+1]==2))?2:v;}
        unsigned char *tmp=a;a=b;b=tmp;steps++;
    }
    double elapsed=MPI_Wtime()-start;long burned=0;int reached=0;for(int i=0;i<n;i++)for(int j=0;j<n;j++){unsigned char v=a[(size_t)i*n+j];burned+=v==3;if(j==n-1&&v==3)reached=1;}
    save(path,a,n);printf("версія=V0 метод=послідовний N=%d P=1 d=%.4f seed=%llu кроки=%d згоріло=%ld досягнуто=%d hash=%016llx час=%.9f обмін=0 обчислення=%.9f редукція=0\n",n,d,(unsigned long long)seed,steps,burned,reached,(unsigned long long)digest(a,length),elapsed,elapsed);free(a);free(b);
}
static void cartesian(int argc,char **argv,int rank,int size) {
    int n=number(argv[1]);double d=atof(argv[2]);uint64_t seed=strtoull(argv[3],NULL,10);const char *path=argc>4?argv[4]:"";
    int dims[2]={0,0};MPI_Dims_create(size,2,dims);if(n<dims[0]||n<dims[1])MPI_Abort(MPI_COMM_WORLD,3);
    MPI_Comm cart;int periods[2]={0,0};MPI_Cart_create(MPI_COMM_WORLD,2,dims,periods,0,&cart);int coords[2];MPI_Cart_coords(cart,rank,2,coords);
    int r0,rows,c0,cols;split(n,dims[0],coords[0],&r0,&rows);split(n,dims[1],coords[1],&c0,&cols);
    int stride=cols+2;size_t sz=(size_t)(rows+2)*stride;unsigned char *a=calloc(sz,1),*b=calloc(sz,1);
    for(int i=1;i<=rows;i++)for(int j=1;j<=cols;j++)a[(size_t)i*stride+j]=initial(seed,r0+i-1,c0+j-1,d);
    int up,down,left,right;MPI_Cart_shift(cart,0,1,&up,&down);MPI_Cart_shift(cart,1,1,&left,&right);
    MPI_Datatype column;MPI_Type_vector(rows,1,stride,MPI_BYTE,&column);MPI_Type_commit(&column);
    MPI_Barrier(cart);double start=MPI_Wtime(),comm=0,compute=0,red=0;int steps=0;
    while(1){long own=0,all=0;for(int i=1;i<=rows;i++)for(int j=1;j<=cols;j++)own+=a[(size_t)i*stride+j]==2;
        double t=MPI_Wtime();MPI_Allreduce(&own,&all,1,MPI_LONG,MPI_SUM,cart);red+=MPI_Wtime()-t;if(!all)break;
        t=MPI_Wtime();
        MPI_Sendrecv(a+stride+1,cols,MPI_BYTE,up,11,a+1,cols,MPI_BYTE,up,12,cart,MPI_STATUS_IGNORE);
        MPI_Sendrecv(a+(size_t)rows*stride+1,cols,MPI_BYTE,down,12,a+(size_t)(rows+1)*stride+1,cols,MPI_BYTE,down,11,cart,MPI_STATUS_IGNORE);
        MPI_Sendrecv(a+stride+1,1,column,left,13,a+stride,1,column,left,14,cart,MPI_STATUS_IGNORE);
        MPI_Sendrecv(a+stride+cols,1,column,right,14,a+stride+cols+1,1,column,right,13,cart,MPI_STATUS_IGNORE);
        comm+=MPI_Wtime()-t;t=MPI_Wtime();
        for(int i=1;i<=rows;i++)for(int j=1;j<=cols;j++){int k=i*stride+j;unsigned char v=a[k];b[k]=v==2?3:v==1&&(a[k-stride]==2||a[k+stride]==2||a[k-1]==2||a[k+1]==2)?2:v;}
        compute+=MPI_Wtime()-t;unsigned char *tmp=a;a=b;b=tmp;steps++;
    }
    double elapsed=MPI_Wtime()-start;long burned=0;int reach=0;for(int i=1;i<=rows;i++)for(int j=1;j<=cols;j++){unsigned char v=a[(size_t)i*stride+j];burned+=v==3;if(c0+j==n && v==3)reach=1;}
    long sum=0;int reached=0;MPI_Reduce(&burned,&sum,1,MPI_LONG,MPI_SUM,0,cart);MPI_Reduce(&reach,&reached,1,MPI_INT,MPI_MAX,0,cart);
    unsigned char *local=malloc((size_t)rows*cols);for(int i=0;i<rows;i++)memcpy(local+(size_t)i*cols,a+(size_t)(i+1)*stride+1,cols);
    unsigned char *full=NULL;if(rank==0)full=calloc((size_t)n*n,1);
    if(rank==0){for(int r=0;r<size;r++){int xy[2],rr,cc,rs,cs;MPI_Cart_coords(cart,r,2,xy);split(n,dims[0],xy[0],&rr,&rs);split(n,dims[1],xy[1],&cc,&cs);unsigned char *tmp=r?malloc((size_t)rs*cs):local;if(r)MPI_Recv(tmp,rs*cs,MPI_BYTE,r,71,cart,MPI_STATUS_IGNORE);for(int i=0;i<rs;i++)memcpy(full+(size_t)(rr+i)*n+cc,tmp+(size_t)i*cs,cs);if(r)free(tmp);}}else MPI_Send(local,rows*cols,MPI_BYTE,0,71,cart);
    double own[4]={elapsed,comm,compute,red},max[4];MPI_Reduce(own,max,4,MPI_DOUBLE,MPI_MAX,0,cart);
    if(rank==0){save(path,full,n);printf("версія=2D метод=декартовий N=%d P=%d d=%.4f seed=%llu кроки=%d згоріло=%ld досягнуто=%d hash=%016llx час=%.9f обмін=%.9f обчислення=%.9f редукція=%.9f сітка=%dx%d\n",n,size,d,(unsigned long long)seed,steps,sum,reached,(unsigned long long)digest(full,(size_t)n*n),max[0],max[1],max[2],max[3],dims[0],dims[1]);free(full);}
    free(local);free(a);free(b);MPI_Type_free(&column);MPI_Comm_free(&cart);
}
int main(int argc,char **argv){int supplied;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&supplied);int rank,size;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);if(argc<5){if(!rank)fprintf(stderr,"Запуск: fire v0 N d seed [файл] | fire stripe N d seed [ordered|nonblocking|overlap|hybrid|bad] [файл] | fire 2d N d seed [файл]\n");MPI_Finalize();return 2;}if(!strcmp(argv[1],"v0"))sequential(argc-1,argv+1,rank,size);else if(!strcmp(argv[1],"stripe"))stripe(argc-1,argv+1,rank,size);else if(!strcmp(argv[1],"2d"))cartesian(argc-1,argv+1,rank,size);else{if(!rank)fprintf(stderr,"Невідома версія\n");MPI_Abort(MPI_COMM_WORLD,3);}MPI_Finalize();return 0;}
