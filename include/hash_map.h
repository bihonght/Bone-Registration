#ifndef HASH_MAP_H_
#define HASH_MAP_H_

#include "parallel_hashmap/phmap.h"
#include "mat.h"

namespace std {
// inject specialization of std::hash for Vec3i into namespace std
// ----------------------------------------------------------------
    template <> struct hash<Vec3i> {
        std::size_t operator()(Vec3i const& p) const {
            return phmap::HashState().combine(p[0], p[1], p[2]);
        }
    };
} // namespace std


#endif //HASH_MAP_H_