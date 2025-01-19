#ifndef _TIMER_H_
#define _TIMER_H_

#include <cstdint>
#include <list>
#include <string>
#include <omp.h>

#define PROFILE__(name,line) \
static timer& __timer##line = timer::get(name); \
timer_guard __guard##line(__timer##line);

#define PROFILE_(name,line) PROFILE__(name,line)

#define PROFILE_SECTION(name) { PROFILE_(name, __LINE__)

#define PROFILE_FUNCTION PROFILE_(__func__, __LINE__)

#define PROFILE_STOP }

#define PROFILE_FLOPS(n) do_flops(n)

class interval
{
    friend class timer;
    friend void do_flops(int64_t flops);
    friend interval toc();
    friend interval cputoc();

    protected:
        double _dt;
        int64_t _flops;

        interval(double start, int64_t flops) : _dt(start), _flops(flops) {}

    public:
        interval() : _dt(0), _flops(0) {}

        static interval time();

        static interval cputime();

        bool operator<(const interval& other) const;

        interval& operator+=(const interval& other);

        interval& operator-=(const interval& other);

        interval& operator*=(int m);

        interval& operator/=(int m);

        interval operator+(const interval& other) const;

        interval operator-(const interval& other) const;

        interval operator*(int m) const;

        interval operator/(int m) const;

        double seconds() const;

        double gflops() const;
};

void tic();

interval toc();

void cputic();

interval cputoc();

void do_flops(int64_t flops);

class timer
{
    protected:
        static std::list<timer> _timers;

        std::string _name;
        interval _interval;
        int64_t _count;

    public:
        timer(const std::string& name) : _name(name), _count(0) {}

        bool operator<(const timer& other) const;

        static timer& get(const std::string& name);

        void start() { tic(); }

        void stop()
        {
            #ifdef _OPENMP
            if (!omp_in_parallel())
            {
                _interval += toc()*omp_get_max_threads();
                _count++;
            }
            else
            #pragma omp critical
            #endif
            {
                _interval += toc();
                _count++;
            }
        }

        static void print_timers();

        static void clear_timers();

        double seconds() const { return _interval.seconds(); }

        double gflops() const { return _interval.gflops(); }
};

class timer_guard
{
    protected:
        timer& _timer;

    public:
        timer_guard(timer& timer)
        : _timer(timer)
        {
            timer.start();
        }

        ~timer_guard()
        {
            _timer.stop();
        }
};

#endif
