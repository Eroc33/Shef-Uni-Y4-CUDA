#include "rgb.h"
extern inline big_rgb to_big_rgb(rgb* rgb);
extern inline rgb from_big_rgb(big_rgb* big_rgb);
extern inline void big_rgb_add_assign(big_rgb* augend, rgb* addend);
extern inline void big_rgb_div_assign(big_rgb* dividend, unsigned int divisor);