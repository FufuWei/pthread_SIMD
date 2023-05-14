#include <pthread.h>
#include <immintrin.h>
#include <iostream>
#include <cmath>

using namespace std;

// 矩阵行数
const int N = 1024;

// 线程数
const int THREAD_NUM = 4;

// 矩阵和向量
float A[N][N], B[N];

// pthread参数
struct ThreadArgs {
    int start;  // 起始行
    int end;    // 结束行
};

// 高斯消元函数，参数为线程号
void gauss(int tid) {
    // 计算每个线程处理的行数
    int chunk_size = (N + THREAD_NUM - 1) / THREAD_NUM;
    int start = tid * chunk_size;
    int end = min(start + chunk_size, N);

    // 消元过程
    for (int k = 0; k < N - 1; k++) {
        if (k >= start && k < end) {
            // 计算主元素
            float max_val = fabs(A[k][k]);
            int max_row = k;
            for (int i = k + 1; i < N; i++) {
                if (fabs(A[i][k]) > max_val) {
                    max_val = fabs(A[i][k]);
                    max_row = i;
                }
            }
            // 交换行
            if (max_row != k) {
                swap(A[k], A[max_row]);
                swap(B[k], B[max_row]);
            }
            // 消元
            for (int i = k + 1; i < N; i++) {
                float factor = A[i][k] / A[k][k];
                for (int j = k + 1; j < N; j += 8) {
                    __m256 a = _mm256_loadu_ps(&A[k][j]);
                    __m256 b = _mm256_loadu_ps(&A[i][j]);
                    __m256 c = _mm256_set1_ps(factor);
                    __m256 d = _mm256_mul_ps(c, a);
                    __m256 e = _mm256_sub_ps(b, d);
                    _mm256_storeu_ps(&A[i][j], e);
                }
                B[i] -= factor * B[k];
            }
        }
        // 等待其他线程完成当前轮计算
        pthread_barrier_wait(&barrier);
    }
}

int main() {
    // 初始化矩阵和向量
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (float)rand() / RAND_MAX;
        }
        B[i] = (float)rand() / RAND_MAX;
    }

    // 创建线程
    pthread_t threads[THREAD_NUM];
    for (int i = 0; i < THREAD_NUM; i++) {
        ThreadArgs* args = new ThreadArgs;
        args->start = i * (N / THREAD_NUM);
        args->end = (i + 1) * (N / THREAD_NUM);
        pthread_create(&threads[i], NULL, (void* (*)(void*))gauss, (void*)i);
    }

    // 等待线程完成
    for (int i = 0; i < THREAD_NUM; i++) {
        pthread_join(threads[i], NULL);
    }

    // 输出结果
    cout << "x = ";
    for (int i = 0; i < N; i++) {
        cout << B[i] << " ";
    }
    cout << endl;

    return 0;
}