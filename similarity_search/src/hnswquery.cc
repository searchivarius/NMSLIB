/**
* Non-metric Space Library
 *
 * Main developers: Bilegsaikhan Naidan, Leonid Boytsov, Yury Malkov, Ben Frederickson, David Novak
 *
 * For the complete list of contributors and further details see:
 * https://github.com/nmslib/nmslib
 *
 * Copyright (c) 2013-2018
 *
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#include "hnswquery.h"

namespace similarity {

template<typename dist_t>
HNSWQuery<dist_t>::HNSWQuery(const Space<dist_t> &space, const Object* query_object, const unsigned K, unsigned ef, float eps)
    : KNNQuery<dist_t>(space, query_object, K, eps),
      ef_(ef) {
}

template <typename dist_t>
HNSWQuery<dist_t>::~HNSWQuery() = default;

template class HNSWQuery<float>;
template class HNSWQuery<int>;
template class HNSWQuery<short int>;
}
