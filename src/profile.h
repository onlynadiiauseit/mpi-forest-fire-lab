#ifndef FIRE_PROFILE_H
#define FIRE_PROFILE_H

/* Components from the same critical rank add up to the maximum step time.
 * All profiling collectives run after the simulation timer has stopped. */
enum { COMPUTE, EXCHANGE, COUNT, REDUCTION, OTHER, COMPONENTS };

typedef struct {
    double total;
    double part[COMPONENTS];
} StepTime;

typedef struct {
    StepTime *steps;
    int length;
    int capacity;
} Profile;

typedef struct {
    double total;
    double part[COMPONENTS];
    double spread;
    double wall;
    double profiling;
} Timing;

static void *checked_alloc(size_t bytes) {
    void *pointer = calloc(bytes ? bytes : 1, 1);
    if (!pointer) {
        fprintf(stderr, "Недостатньо пам'яті\n");
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
    return pointer;
}

static Profile profile_create(int n) {
    Profile profile = {NULL, 0, 8 * n + 64};
    profile.steps = checked_alloc((size_t)profile.capacity * sizeof(StepTime));
    return profile;
}

static void profile_append(Profile *profile, StepTime step) {
    if (profile->length == profile->capacity) {
        if (profile->capacity > INT_MAX / 2) MPI_Abort(MPI_COMM_WORLD, 4);
        profile->capacity *= 2;
        void *grown = realloc(profile->steps,
                              (size_t)profile->capacity * sizeof(StepTime));
        if (!grown) MPI_Abort(MPI_COMM_WORLD, 4);
        profile->steps = grown;
    }
    double measured = 0;
    for (int i = 0; i < OTHER; ++i) measured += step.part[i];
    step.part[OTHER] = step.total - measured;
    profile->steps[profile->length++] = step;
}

static Timing profile_finish(Profile *profile, double wall, MPI_Comm comm) {
    double started = MPI_Wtime();
    int rank;
    MPI_Comm_rank(comm, &rank);
    typedef struct { double value; int rank; } ValueRank;
    int count = profile->length;
    ValueRank *local = checked_alloc((size_t)count * sizeof(ValueRank));
    ValueRank *critical = checked_alloc((size_t)count * sizeof(ValueRank));
    double *values = checked_alloc((size_t)count * sizeof(double));
    double *minimum = checked_alloc((size_t)count * sizeof(double));
    for (int i = 0; i < count; ++i) {
        local[i].value = values[i] = profile->steps[i].total;
        local[i].rank = rank;
    }
    MPI_Allreduce(local, critical, count, MPI_DOUBLE_INT, MPI_MAXLOC, comm);
    MPI_Reduce(values, minimum, count, MPI_DOUBLE, MPI_MIN, 0, comm);
    double own[COMPONENTS] = {0};
    Timing result = {0};
    for (int i = 0; i < count; ++i) {
        result.total += critical[i].value;
        if (critical[i].rank == rank) {
            for (int k = 0; k < COMPONENTS; ++k)
                own[k] += profile->steps[i].part[k];
        }
        if (rank == 0) result.spread += critical[i].value - minimum[i];
    }
    MPI_Reduce(own, result.part, COMPONENTS, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&wall, &result.wall, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    free(local);
    free(critical);
    free(values);
    free(minimum);
    free(profile->steps);
    double elapsed = MPI_Wtime() - started;
    MPI_Reduce(&elapsed, &result.profiling, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    return result;
}

#endif
