#include <mpi.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <errno.h>
#include "profile.h"

typedef struct {
    int n, rows, cols, row0, col0, stride;
    int rank, size, up, down, left, right;
    MPI_Comm comm;
    MPI_Datatype column;
    unsigned char *current, *next;
} Grid;

static void invalid(const char *message) {
    fprintf(stderr, "%s\n", message);
    MPI_Abort(MPI_COMM_WORLD, 2);
}

static int dimension(const char *value) {
    char *end;
    errno = 0;
    long n = strtol(value, &end, 10);
    if (errno || end == value || *end || n < 1 || n > 46340)
        invalid("Розмір N має бути цілим числом від 1 до 46340");
    return (int)n;
}

static double density_value(const char *value) {
    char *end;
    errno = 0;
    double density = strtod(value, &end);
    if (errno || end == value || *end || !isfinite(density) ||
        density < 0 || density > 1)
        invalid("Щільність має бути числом від 0 до 1");
    return density;
}

static uint64_t seed_value(const char *value) {
    char *end;
    errno = 0;
    unsigned long long seed = strtoull(value, &end, 10);
    if (errno || end == value || *end || value[0] == '-')
        invalid("Некоректне значення seed");
    return (uint64_t)seed;
}

static uint64_t mix(uint64_t x) {
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static unsigned char initial(uint64_t seed, int i, int j, double density) {
    uint64_t x = mix(seed ^ (uint64_t)(i + 1) * UINT64_C(0x9e3779b97f4a7c15)
                         ^ (uint64_t)(j + 1) * UINT64_C(0xd1b54a32d192ed03));
    unsigned char tree = (double)(x >> 11) * 0x1.0p-53 < density;
    return tree && j == 0 ? 2 : tree;
}

static void split(int n, int p, int rank, int *start, int *count) {
    *start = (int)((long long)n * rank / p);
    *count = (int)((long long)n * (rank + 1) / p) - *start;
}

static Grid create_grid(int n, double density, uint64_t seed, int cartesian) {
    Grid grid = {0};
    grid.n = n;
    grid.comm = MPI_COMM_WORLD;
    MPI_Comm_size(grid.comm, &grid.size);
    MPI_Comm_rank(grid.comm, &grid.rank);
    grid.left = grid.right = MPI_PROC_NULL;
    grid.column = MPI_DATATYPE_NULL;
    if (cartesian) {
        int dims[2] = {0, 0}, periods[2] = {0, 0}, coords[2];
        MPI_Dims_create(grid.size, 2, dims);
        if (n < dims[0] || n < dims[1]) invalid("N замале для 2D-сітки процесів");
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid.comm);
        MPI_Cart_coords(grid.comm, grid.rank, 2, coords);
        split(n, dims[0], coords[0], &grid.row0, &grid.rows);
        split(n, dims[1], coords[1], &grid.col0, &grid.cols);
        MPI_Cart_shift(grid.comm, 0, 1, &grid.up, &grid.down);
        MPI_Cart_shift(grid.comm, 1, 1, &grid.left, &grid.right);
    } else {
        if (n < grid.size) invalid("Для смуг потрібно N >= P");
        split(n, grid.size, grid.rank, &grid.row0, &grid.rows);
        grid.cols = n;
        grid.up = grid.rank ? grid.rank - 1 : MPI_PROC_NULL;
        grid.down = grid.rank + 1 < grid.size ? grid.rank + 1 : MPI_PROC_NULL;
    }
    grid.stride = grid.cols + 2;
    size_t bytes = (size_t)(grid.rows + 2) * grid.stride;
    grid.current = checked_alloc(bytes);
    grid.next = checked_alloc(bytes);
    for (int i = 1; i <= grid.rows; ++i)
        for (int j = 1; j <= grid.cols; ++j)
            grid.current[(size_t)i * grid.stride + j] =
                initial(seed, grid.row0 + i - 1, grid.col0 + j - 1, density);
    if (cartesian) {
        MPI_Type_vector(grid.rows, 1, grid.stride, MPI_BYTE, &grid.column);
        MPI_Type_commit(&grid.column);
    }
    return grid;
}

static long count_burning(const Grid *grid) {
    long count = 0;
    for (int i = 1; i <= grid->rows; ++i)
        for (int j = 1; j <= grid->cols; ++j)
            count += grid->current[(size_t)i * grid->stride + j] == 2;
    return count;
}

static void update(Grid *grid, int first, int last, int threads) {
    const unsigned char *a = grid->current;
    unsigned char *b = grid->next;
    int stride = grid->stride, cols = grid->cols;
    #pragma omp parallel for if(threads > 1) num_threads(threads) schedule(static)
    for (int i = first; i <= last; ++i) {
        for (int j = 1; j <= cols; ++j) {
            size_t x = (size_t)i * stride + j;
            unsigned char value = a[x];
            if (value == 2) b[x] = 3;
            else if (value == 1 && (a[x - stride] == 2 || a[x + stride] == 2 ||
                                   a[x - 1] == 2 || a[x + 1] == 2)) b[x] = 2;
            else b[x] = value;
        }
    }
}

static void post_rows(Grid *grid, MPI_Request requests[4]) {
    unsigned char *a = grid->current;
    int stride = grid->stride, cols = grid->cols;
    MPI_Irecv(a + 1, cols, MPI_BYTE, grid->up, 12, grid->comm, &requests[0]);
    MPI_Irecv(a + (size_t)(grid->rows + 1) * stride + 1, cols, MPI_BYTE,
              grid->down, 11, grid->comm, &requests[1]);
    MPI_Isend(a + stride + 1, cols, MPI_BYTE, grid->up, 11, grid->comm, &requests[2]);
    MPI_Isend(a + (size_t)grid->rows * stride + 1, cols, MPI_BYTE,
              grid->down, 12, grid->comm, &requests[3]);
}

static void exchange(Grid *grid, const char *method) {
    unsigned char *a = grid->current;
    int stride = grid->stride, cols = grid->cols;
    unsigned char *top = a + stride + 1;
    unsigned char *bottom = a + (size_t)grid->rows * stride + 1;
    unsigned char *upper_halo = a + 1;
    unsigned char *lower_halo = a + (size_t)(grid->rows + 1) * stride + 1;
    if (!strcmp(method, "bad")) {
        MPI_Send(top, cols, MPI_BYTE, grid->up, 11, grid->comm);
        MPI_Send(bottom, cols, MPI_BYTE, grid->down, 12, grid->comm);
        MPI_Recv(upper_halo, cols, MPI_BYTE, grid->up, 12, grid->comm, MPI_STATUS_IGNORE);
        MPI_Recv(lower_halo, cols, MPI_BYTE, grid->down, 11, grid->comm, MPI_STATUS_IGNORE);
    } else if (!strcmp(method, "nonblocking") || !strcmp(method, "hybrid")) {
        MPI_Request requests[4];
        post_rows(grid, requests);
        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
    } else {
        /* Opposite send/receive neighbours avoid a serialized rank chain. */
        MPI_Sendrecv(top, cols, MPI_BYTE, grid->up, 11,
                     lower_halo, cols, MPI_BYTE, grid->down, 11,
                     grid->comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(bottom, cols, MPI_BYTE, grid->down, 12,
                     upper_halo, cols, MPI_BYTE, grid->up, 12,
                     grid->comm, MPI_STATUS_IGNORE);
    }
    if (grid->column != MPI_DATATYPE_NULL) {
        MPI_Sendrecv(top, 1, grid->column, grid->left, 13,
                     a + stride + cols + 1, 1, grid->column, grid->right, 13,
                     grid->comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(a + stride + cols, 1, grid->column, grid->right, 14,
                     a + stride, 1, grid->column, grid->left, 14,
                     grid->comm, MPI_STATUS_IGNORE);
    }
}

static int simulate(Grid *grid, int sequential, const char *method, int threads,
                    Profile *profile, double *wall) {
    int steps = 0;
    MPI_Barrier(grid->comm);
    double started = MPI_Wtime();
    for (;;) {
        StepTime step = {0};
        double step_started = MPI_Wtime(), timer = MPI_Wtime();
        long own = count_burning(grid), burning = own;
        step.part[COUNT] = MPI_Wtime() - timer;
        if (!sequential) {
            timer = MPI_Wtime();
            MPI_Allreduce(&own, &burning, 1, MPI_LONG, MPI_SUM, grid->comm);
            step.part[REDUCTION] = MPI_Wtime() - timer;
        }
        if (burning) {
            if (!sequential && !strcmp(method, "overlap")) {
                MPI_Request requests[4];
                timer = MPI_Wtime();
                post_rows(grid, requests);
                step.part[EXCHANGE] = MPI_Wtime() - timer;
                timer = MPI_Wtime();
                if (grid->rows > 2) update(grid, 2, grid->rows - 1, threads);
                step.part[COMPUTE] = MPI_Wtime() - timer;
                timer = MPI_Wtime();
                MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
                step.part[EXCHANGE] += MPI_Wtime() - timer;
                timer = MPI_Wtime();
                update(grid, 1, 1, threads);
                if (grid->rows > 1) update(grid, grid->rows, grid->rows, threads);
                step.part[COMPUTE] += MPI_Wtime() - timer;
            } else {
                if (!sequential) {
                    timer = MPI_Wtime();
                    exchange(grid, method);
                    step.part[EXCHANGE] = MPI_Wtime() - timer;
                }
                timer = MPI_Wtime();
                update(grid, 1, grid->rows, threads);
                step.part[COMPUTE] = MPI_Wtime() - timer;
            }
            unsigned char *old = grid->current;
            grid->current = grid->next;
            grid->next = old;
            ++steps;
        }
        step.total = MPI_Wtime() - step_started;
        profile_append(profile, step);
        if (!burning) break;
    }
    *wall = MPI_Wtime() - started;
    return steps;
}

static unsigned char *gather_grid(const Grid *grid) {
    int info[4] = {grid->row0, grid->rows, grid->col0, grid->cols};
    int *all_info = grid->rank == 0 ? checked_alloc((size_t)4 * grid->size * sizeof(int)) : NULL;
    MPI_Gather(info, 4, MPI_INT, all_info, 4, MPI_INT, 0, grid->comm);
    unsigned char *packed = checked_alloc((size_t)grid->rows * grid->cols);
    for (int i = 0; i < grid->rows; ++i)
        memcpy(packed + (size_t)i * grid->cols,
               grid->current + (size_t)(i + 1) * grid->stride + 1, grid->cols);
    unsigned char *full = NULL;
    if (grid->rank == 0) {
        full = checked_alloc((size_t)grid->n * grid->n);
        for (int rank = 0; rank < grid->size; ++rank) {
            int *part = all_info + 4 * rank;
            unsigned char *received = rank ? checked_alloc((size_t)part[1] * part[3]) : packed;
            if (rank) MPI_Recv(received, part[1] * part[3], MPI_BYTE,
                               rank, 71, grid->comm, MPI_STATUS_IGNORE);
            for (int i = 0; i < part[1]; ++i)
                memcpy(full + (size_t)(part[0] + i) * grid->n + part[2],
                       received + (size_t)i * part[3], part[3]);
            if (rank) free(received);
        }
    } else MPI_Send(packed, grid->rows * grid->cols, MPI_BYTE, 0, 71, grid->comm);
    free(packed);
    free(all_info);
    return full;
}

static void output(const Grid *grid, const unsigned char *full, const char *path,
                   const char *version, const char *method, double density,
                   uint64_t seed, int steps, const Timing *timing, int ticks) {
    uint64_t hash = UINT64_C(14695981039346656037);
    long burned = 0;
    int reached = 0;
    for (size_t i = 0; i < (size_t)grid->n * grid->n; ++i) {
        hash = (hash ^ full[i]) * UINT64_C(1099511628211);
        burned += full[i] == 3;
        if (i % grid->n == (size_t)(grid->n - 1) && full[i] == 3) reached = 1;
    }
    if (path && *path) {
        FILE *file = fopen(path, "wb");
        if (!file || fwrite(full, 1, (size_t)grid->n * grid->n, file) != (size_t)grid->n * grid->n)
            invalid("Не вдалося записати кінцеву сітку");
        fclose(file);
    }
    printf("версія=%s метод=%s N=%d P=%d d=%.4f seed=%llu кроки=%d згоріло=%ld "
           "досягнуто=%d hash=%016llx час=%.9f обмін=%.9f обчислення=%.9f "
           "редукція=%.9f облік=%.9f інше=%.9f дисбаланс=%.9f wall=%.9f "
           "профілювання=%.9f такти=%d\n",
           version, method, grid->n, grid->size, density, (unsigned long long)seed,
           steps, burned, reached, (unsigned long long)hash, timing->total,
           timing->part[EXCHANGE], timing->part[COMPUTE], timing->part[REDUCTION],
           timing->part[COUNT], timing->part[OTHER], timing->spread,
           timing->wall, timing->profiling, ticks);
}

int main(int argc, char **argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) invalid("MPI не підтримує MPI_THREAD_FUNNELED");
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc < 5) {
        if (!rank) fprintf(stderr, "fire v0 N d seed [файл] | fire stripe N d seed "
                                   "[ordered|nonblocking|overlap|hybrid|bad] [файл] | fire 2d N d seed [файл]\n");
        MPI_Finalize();
        return 2;
    }
    int sequential = !strcmp(argv[1], "v0");
    int cartesian = !strcmp(argv[1], "2d");
    if (!sequential && !cartesian && strcmp(argv[1], "stripe")) invalid("Невідома версія");
    if (sequential && size != 1) invalid("V0 потребує одного процесу");
    int n = dimension(argv[2]);
    double density = density_value(argv[3]);
    uint64_t seed = seed_value(argv[4]);
    const char *method = sequential ? "v0" : cartesian ? "2d" : argc > 5 ? argv[5] : "ordered";
    if (!sequential && !cartesian && strcmp(method, "ordered") && strcmp(method, "nonblocking") &&
        strcmp(method, "overlap") && strcmp(method, "hybrid") && strcmp(method, "bad")) invalid("Невідомий метод обміну");
    const char *path = sequential || cartesian ? (argc > 5 ? argv[5] : "") : (argc > 6 ? argv[6] : "");
    int threads = !strcmp(method, "hybrid") ? omp_get_max_threads() : 1;
    Grid grid = create_grid(n, density, seed, cartesian);
    Profile profile = profile_create(n);
    double wall;
    int steps = simulate(&grid, sequential, method, threads, &profile, &wall);
    Timing timing = profile_finish(&profile, wall, grid.comm);
    unsigned char *full = gather_grid(&grid);
    if (rank == 0) output(&grid, full, path, sequential ? "V0" : cartesian ? "2D" : "V1/V3",
                          method, density, seed, steps, &timing, profile.length);
    free(full);
    free(grid.current);
    free(grid.next);
    if (grid.column != MPI_DATATYPE_NULL) MPI_Type_free(&grid.column);
    if (cartesian) MPI_Comm_free(&grid.comm);
    MPI_Finalize();
    return 0;
}
