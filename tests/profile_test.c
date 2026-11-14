#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include "profile.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) MPI_Abort(MPI_COMM_WORLD, 2);
    Profile profile = profile_create(1);
    const double totals[2][3] = {{10,5,4},{7,12,4}};
    const double parts[2][3][4] = {
        {{2,3,1,2},{1,1,1,1},{1,1,1,0}},
        {{1,1,2,1},{4,2,2,3},{0,2,1,1}}
    };
    for (int i=0; i<3; ++i) {
        StepTime step = {0};
        step.total = totals[rank][i];
        for (int k=0; k<4; ++k) step.part[k] = parts[rank][i][k];
        profile_append(&profile, step);
    }
    Timing timing = profile_finish(&profile, rank ? 23 : 19, MPI_COMM_WORLD);
    if (rank == 0) {
        const double expected[5] = {7,6,4,5,4};
        if (fabs(timing.total-26)>1e-12 || fabs(timing.spread-10)>1e-12 || fabs(timing.wall-23)>1e-12)
            MPI_Abort(MPI_COMM_WORLD, 3);
        for (int k=0; k<5; ++k)
            if (fabs(timing.part[k]-expected[k])>1e-12) MPI_Abort(MPI_COMM_WORLD, 4);
        puts("PASS: changing critical rank, MAXLOC tie, additive profile, spread, wall-time");
    }
    MPI_Finalize();
    return 0;
}
