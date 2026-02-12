#include <benchmark/benchmark.h>

#include <rivoli/rivoli.hpp>

static void BM_2S_topology_vectratype_Nonef(benchmark::State& state)
{
    rivoli::Topology2S<float, vectra::SIMDLevel::None> topology;

    using vct = vectra::Vectratype<float, vectra::SIMDLevel::None>;

    vct t1(0.1f), p1(0.2f), t2(0.3f), p2(0.4f);

    for (auto _ : state)
    {
        auto r = topology.getDistance(t1, p1, t2, p2);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_2S_topology_vectratype_Nonef)
    ->MinTime(60.0)
    ->Repetitions(10)
    ->ReportAggregatesOnly();

static void BM_2S_distance_vectratype_Nonef(benchmark::State& state)
{
    using vct = vectra::Vectratype<float, vectra::SIMDLevel::None>;

    vct t1(0.1f), p1(0.2f), t2(0.3f), p2(0.4f);

    for (auto _ : state)
    {
        auto r = rivoli::distanceGreatCircle(t1, p1, t2, p2);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_2S_distance_vectratype_Nonef)
    ->MinTime(60.0)
    ->Repetitions(10)
    ->ReportAggregatesOnly();

static void BM_2S_distance_backend_Nonef(benchmark::State& state)
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
BENCHMARK(BM_2S_distance_backend_Nonef)
    ->MinTime(60.0)
    ->Repetitions(10)
    ->ReportAggregatesOnly();

BENCHMARK_MAIN();