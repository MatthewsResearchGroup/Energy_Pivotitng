#include "timer.h"
#include "marray.hpp"

#include <vector>

#ifdef __MACH__
#include <mach/mach_time.h>
#endif

static std::vector<interval> tics[128];

interval interval::time()
{
    #ifdef __MACH__

    static auto conv = []
    {
        mach_timebase_info_data_t timebase;
        mach_timebase_info(&timebase);
        return 1e-9 * timebase.numer / timebase.denom;
    }();

    return interval(conv*mach_absolute_time(), 0);

    #else

    timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) ERROR("clock_gettime");
    return interval(ts.tv_sec + 1e-9*ts.tv_nsec, 0);

    #endif
}

interval interval::cputime()
{
    #ifdef __MACH__

    return time();

    #else

    timespec ts;
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts) != 0) ERROR("clock_gettime");
    return interval(ts.tv_sec + 1e-9*ts.tv_nsec, 0);

    #endif
}

bool interval::operator<(const interval& other) const
{
    return _dt < other._dt;
}

interval& interval::operator+=(const interval& other)
{
    _dt += other._dt;
    _flops += other._flops;
    return *this;
}

interval& interval::operator-=(const interval& other)
{
    _dt -= other._dt;
    _flops -= other._flops;
    return *this;
}

interval& interval::operator*=(int m)
{
    _dt *= m;
    return *this;
}

interval& interval::operator/=(int m)
{
    _dt /= m;
    return *this;
}

interval interval::operator+(const interval& other) const
{
    interval r(*this);
    r += other;
    return r;
}

interval interval::operator-(const interval& other) const
{
    interval r(*this);
    r -= other;
    return r;
}

interval interval::operator*(int m) const
{
    interval r(*this);
    r *= m;
    return r;
}

interval interval::operator/(int m) const
{
    interval r(*this);
    r /= m;
    return r;
}

double interval::seconds() const
{
    return _dt;
}

double interval::gflops() const
{
    return 1e-9 * _flops / seconds();
}

std::list<timer> timer::_timers;

bool timer::operator<(const timer& other) const
{
    return _interval < other._interval;
}

timer& timer::get(const std::string& name)
{
    for (auto& timer : _timers)
    {
        if (timer._name == name) return timer;
    }

    _timers.push_back(timer(name));
    return _timers.back();
}

void timer::print_timers()
{
    int max_len = 0;
    for (auto& timer : _timers)
        max_len = std::max(max_len, (int)timer._name.size());

    _timers.sort();

    bool found = false;
    for (auto& timer : _timers)
    {
        auto tot = timer.seconds();
        auto count = timer._count;
        auto gflops = timer.gflops();
        if (count > 0)
        {
            found = true;
            printf("%s:%*s %13.6f s %10d x %11.6f gflops/sec\n",
                   timer._name.c_str(), max_len-(int)timer._name.size(), "",
                   tot, (int)count, gflops);
        }
    }
    if (found) printf("\n");
}

void timer::clear_timers()
{
    for (auto& timer : _timers)
    {
        timer._interval._dt = 0;
        timer._interval._flops = 0;
        timer._count = 0;
    }
}

void tic()
{
    #ifdef _OPENMP
    auto tid = omp_get_thread_num();
    auto ntd = omp_in_parallel() ? 1 : omp_get_max_threads();
    #else
    auto tid = 0;
    auto ntd = 1;
    #endif

    for (auto td : rangeN(tid,ntd))
        tics[td].push_back(td == tid ? interval::time() : interval());
}

interval toc()
{
    #ifdef _OPENMP
    auto tid = omp_get_thread_num();
    auto ntd = (omp_in_parallel() ? 1 : omp_get_max_threads());
    #else
    auto tid = 0;
    auto ntd = 1;
    #endif

    auto dt = interval::time();

    for (auto td : rangeN(tid,ntd))
    {
        assert(!tics[td].empty());
        dt -= tics[td].back();
        tics[td].pop_back();
    }

    if (!tics[tid].empty()) tics[tid].back()._flops -= dt._flops;

    return dt;
}

void cputic()
{
    #ifdef _OPENMP
    auto tid = omp_get_thread_num();
    auto ntd = (omp_in_parallel() ? 1 : omp_get_max_threads());
    #else
    auto tid = 0;
    auto ntd = 1;
    #endif

    for (auto td : rangeN(tid,ntd))
        tics[td].push_back(td == tid ? interval::cputime() : interval());
}

interval cputoc()
{
    #ifdef _OPENMP
    auto tid = omp_get_thread_num();
    auto ntd = (omp_in_parallel() ? 1 : omp_get_max_threads());
    #else
    auto tid = 0;
    auto ntd = 1;
    #endif

    interval dt = interval::cputime();

    for (auto td : rangeN(tid,ntd))
    {
        assert(!tics[td].empty());
        dt -= tics[td].back();
        tics[td].pop_back();
    }

    if (!tics[tid].empty()) tics[tid].back()._flops -= dt._flops;

    return dt;
}

void do_flops(int64_t flops)
{
    #ifdef _OPENMP
    auto tid = omp_get_thread_num();
    #else
    auto tid = 0;
    #endif

    if (!tics[tid].empty())
        tics[tid].back()._flops -= flops;
}
