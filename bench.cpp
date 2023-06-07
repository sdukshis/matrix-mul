#include <algorithm>
#include <benchmark/benchmark.h>
#include <random>
#include <thread>
#include <vector>
#include <limits>
#include <new>
#include <future>
#include <x86intrin.h>
#include <iostream>

using namespace benchmark;

constexpr size_t to_index(size_t i, size_t j, size_t ncol) {
  return i * ncol + j;
}

template<typename Vector>
static void generate(Vector &m, size_t n) {
    std::random_device rnd;
    std::mt19937 gen(rnd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    m.resize(n*n);
    std::generate_n(m.begin(), m.size(), [&]() { return dis(gen); });
}

static void BM_generate(State &state) {
  const int n = state.range(0);

  std::vector<float> a;
  for (auto _ : state) {
        generate(a, n);
  }
    DoNotOptimize(a);
}

BENCHMARK(BM_generate)->Range(1 << 1, 1 << 10);

template<typename T>
static void naive(const std::vector<T> &a, const std::vector<T> &b, std::vector<T> &c, const size_t n) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            c[to_index(i, j, n)] = 0.0;
            for (size_t k = 0; k < n; ++k) {
                c[to_index(i, j, n)] += a[to_index(i, k, n)] * b[to_index(k, j, n)];
            }
        }
    }
}


static void BM_naive(State &state) {
    const int n = state.range(0);

    std::vector<float> a(n*n), b(n*n), c(n*n);
    for (auto _: state) {
        generate(a, n);
        generate(b, n);
        naive(a, b, c, n);
    }
    DoNotOptimize(c);
}

BENCHMARK(BM_naive)->Range(1 << 1, 1 << 10);





template<typename T>
static void mt(const std::vector<T> &a, const std::vector<T> &b, std::vector<T> &c, const size_t n) {
    
    size_t nthreads = std::min(n, static_cast<size_t>(std::thread::hardware_concurrency()));

    size_t rows_per_thread = n / nthreads;
    std::vector<std::future<void>> tasks;
    for (size_t t = 0; t < nthreads; ++t) {
        tasks.emplace_back(std::async(std::launch::async, [&](size_t from , size_t to) {
            for (size_t i = from; i < to; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    for (size_t k = 0; k < n; ++k) {
                        c[to_index(i, j, n)] += a[to_index(i, k, n)] * b[to_index(k, j, n)];
                    }
                }
            }
        }, t*rows_per_thread, (t + 1)*rows_per_thread)); 
    }
    for (auto &f : tasks) {
        f.wait();
    }
}

static void BM_mt(State &state) {
    const int n = state.range(0);

    std::vector<float> a(n*n), b(n*n), c(n*n);
    for (auto _: state) {
        generate(a, n);
        generate(b, n);
        mt(a, b, c, n);
    }
    DoNotOptimize(c);
}

BENCHMARK(BM_mt)->Range(1 << 1, 1 << 10);

/**
 * Returns aligned pointers when allocations are requested. Default alignment
 * is 64B = 512b, sufficient for AVX-512 and most cache line sizes.
 *
 * @tparam ALIGNMENT_IN_BYTES Must be a positive power of 2.
 */
template<typename    ElementType,
         std::size_t ALIGNMENT_IN_BYTES = 64>
class AlignedAllocator
{
private:
    static_assert(
        ALIGNMENT_IN_BYTES >= alignof( ElementType ),
        "Beware that types like int have minimum alignment requirements "
        "or access will result in crashes."
    );

public:
    using value_type = ElementType;
    static std::align_val_t constexpr ALIGNMENT{ ALIGNMENT_IN_BYTES };

    /**
     * This is only necessary because AlignedAllocator has a second template
     * argument for the alignment that will make the default
     * std::allocator_traits implementation fail during compilation.
     * @see https://stackoverflow.com/a/48062758/2191065
     */
    template<class OtherElementType>
    struct rebind
    {
        using other = AlignedAllocator<OtherElementType, ALIGNMENT_IN_BYTES>;
    };

public:
    constexpr AlignedAllocator() noexcept = default;

    constexpr AlignedAllocator( const AlignedAllocator& ) noexcept = default;

    template<typename U>
    constexpr AlignedAllocator( AlignedAllocator<U, ALIGNMENT_IN_BYTES> const& ) noexcept
    {}

    [[nodiscard]] ElementType*
    allocate( std::size_t nElementsToAllocate )
    {
        if ( nElementsToAllocate
             > std::numeric_limits<std::size_t>::max() / sizeof( ElementType ) ) {
            throw std::bad_array_new_length();
        }

        auto const nBytesToAllocate = nElementsToAllocate * sizeof( ElementType );
        return reinterpret_cast<ElementType*>(
            ::operator new[]( nBytesToAllocate, ALIGNMENT ) );
    }

    void
    deallocate(                  ElementType* allocatedPointer,
                [[maybe_unused]] std::size_t  nBytesAllocated )
    {
        /* According to the C++20 draft n4868 § 17.6.3.3, the delete operator
         * must be called with the same alignment argument as the new expression.
         * The size argument can be omitted but if present must also be equal to
         * the one used in new. */
        ::operator delete[]( allocatedPointer, ALIGNMENT );
    }
};


template<typename T, std::size_t ALIGNMENT_IN_BYTES = 16>
using AlignedVector = std::vector<T, AlignedAllocator<T, ALIGNMENT_IN_BYTES> >;

template<typename Vector>
static void simd(const Vector &a, const Vector &b, Vector &c, const size_t n) {
    size_t nthreads = std::min(n, static_cast<size_t>(std::thread::hardware_concurrency()));

    size_t rows_per_thread = n / nthreads;
    std::vector<std::future<void>> tasks;
    for (size_t t = 0; t < nthreads; ++t) {
        tasks.emplace_back(std::async(std::launch::async, [&](size_t from , size_t to) {
            for (size_t i = from; i < to; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    for (size_t k = 0; k < n; k += 8) {
                        __m256 va = _mm256_loadu_ps(&a[to_index(i, k, n)]);
                        __m256 vb = _mm256_loadu_ps(&b[to_index(j, k, n)]);

                        __m256 vtemp = _mm256_mul_ps(va, vb);
                        // add
                        // extract higher four floats
                        __m128 vhigh = _mm256_extractf128_ps(vtemp, 1); // high 128
                        // add higher four floats to lower floats
                        __m128 vresult = _mm_add_ps(_mm256_castps256_ps128(vtemp), vhigh);
                        // horizontal add of that result
                        vresult = _mm_hadd_ps(vresult, vresult);
                        // another horizontal add of that result
                        vresult = _mm_hadd_ps(vresult, vresult);

                        // store
                        c[to_index(i, j, n)] += _mm_cvtss_f32(vresult);
                    }
                }
            }
        }, t*rows_per_thread, (t + 1)*rows_per_thread));
    }
    for (auto &f : tasks) {
        f.wait();
    }        
}

static void BM_simd(State &state) {
    const int n = state.range(0);

    AlignedVector<float> a(n*n), b(n*n), c(n*n);
    for (auto _: state) {
        generate(a, n);
        generate(b, n);
        simd(a, b, c, n);
    }
    DoNotOptimize(c);
}

BENCHMARK(BM_simd)->Range(1 << 2, 1 << 10);


template<typename T>
static void better(const std::vector<T> &a, const std::vector<T> &b, std::vector<T> &c, const size_t n) {
    std::fill(c.begin(), c.end(), 0.0);

    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < n; ++k) {
            for (size_t j = 0; j < n; ++j) {
                c[to_index(i, j, n)] += a[to_index(i, k, n)] * b[to_index(k, j, n)];
            }
        }
    }
}

static void BM_better(State &state) {
    const int n = state.range(0);

    std::vector<float> a(n*n), b(n*n), c(n*n);
    for (auto _: state) {
        generate(a, n);
        generate(b, n);
        better(a, b, c, n);
    }
    DoNotOptimize(c);
}

BENCHMARK(BM_better)->Range(1 << 1, 1 << 10);

BENCHMARK_MAIN();
