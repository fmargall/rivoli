#include <benchmark/benchmark.h>

#include <rivoli/rivoli.hpp>

static void BM_distance_vectratype_Nonef(benchmark::State& state)
{
    using vct = vectra::Vectratype<float, vectra::SIMDLevel::None>;

    vct t1(0.1f), p1(0.2f), t2(0.3f), p2(0.4f);

    for (auto _ : state)
    {
        auto r = rivoli::distanceGreatCircle(t1, p1, t2, p2);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_distance_vectratype_Nonef)->MinTime(15.0);

static void BM_distance_backend_Nonef(benchmark::State& state)
{
    using backend = vectra::ComputeBackend<float, vectra::SIMDLevel::None>;
    using T = typename backend::type;

    T t1(0.1f), p1(0.2f), t2(0.3f), p2(0.4f);

    for (auto _ : state)
    {
        auto r = rivoli::experimental::distanceGreatCircle<float, vectra::SIMDLevel::None>(t1, p1, t2, p2);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_distance_backend_Nonef)->MinTime(15.0);

BENCHMARK_MAIN();