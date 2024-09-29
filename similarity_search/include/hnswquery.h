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

#ifndef HNSWQUERY_H
#define HNSWQUERY_H
#include "global.h"
#include "knnquery.h"

namespace similarity {

template<typename dist_t>
class HNSWQuery : public KNNQuery<dist_t> {
public:
    ~HNSWQuery();
    HNSWQuery(const Space<dist_t>& space, const Object *query_object, unsigned K, unsigned ef = 100, float eps = 0);

    unsigned getEf() { return ef_; }

protected:
    unsigned ef_;
};

}

#endif //HNSWQUERY_H
