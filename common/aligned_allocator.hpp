#ifndef _ALIGNED_ALLOCATOR_HPP_
#define _ALIGNED_ALLOCATOR_HPP_

#include <cstdlib>
#include <new>

template <typename T, size_t N=8> struct aligned_allocator
{
    typedef T value_type;

    aligned_allocator() {}

    template <typename U, size_t M>
    aligned_allocator(const aligned_allocator<U, M>&) {}

    T* allocate(size_t n)
    {
        if (n == 0) return nullptr;

        void* ptr;
        if (posix_memalign(&ptr, N, n*sizeof(T)) != 0)
            throw std::bad_alloc();

        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, size_t)
    {
        if (ptr) free(ptr);
    }

    template<class U>
    struct rebind { typedef aligned_allocator<U, N> other; };
};

template <typename T, size_t N, typename U, size_t M>
bool operator==(const aligned_allocator<T, N>&, const aligned_allocator<U, M>&) { return true; }

template <typename T, size_t N, typename U, size_t M>
bool operator!=(const aligned_allocator<T, N>&, const aligned_allocator<U, M>&) { return false; }

#endif
