#include <assert.h>
#pragma once
typedef struct rgb{
    unsigned char r;
    unsigned char g;
    unsigned char b;
} rgb;

typedef struct big_rgb{
    unsigned int r;
    unsigned int g;
    unsigned int b;
} big_rgb;

#define rgb_generic_add_assign(augend,addend) \
    (augend)->r += (addend)->r;\
    (augend)->g += (addend)->g;\
    (augend)->b += (addend)->b;

#define rgb_generic_div_assign(dividend,divisor) \
    (dividend)->r /= divisor;\
    (dividend)->g /= divisor;\
    (dividend)->b /= divisor;

inline big_rgb to_big_rgb(rgb* rgb){
    big_rgb converted = {
        rgb->r,
        rgb->g,
        rgb->b,
    };
    return converted;
}

inline rgb from_big_rgb(big_rgb* big_rgb){
    assert(big_rgb->r <= 255);
    assert(big_rgb->g <= 255);
    assert(big_rgb->b <= 255);

    rgb converted = {
        (unsigned char)big_rgb->r,
        (unsigned char)big_rgb->g,
        (unsigned char)big_rgb->b,
    };
    return converted;
}

inline void big_rgb_add_assign(big_rgb* augend, rgb* addend){
    augend->r += addend->r;
    augend->g += addend->g;
    augend->b += addend->b;
}
inline void big_rgb_div_assign(big_rgb* dividend, unsigned int divisor){
    dividend->r /= divisor;
    dividend->g /= divisor;
    dividend->b /= divisor;
}