/**
 * Non-metric Space Library
 *
 * Authors: Bilegsaikhan Naidan (https://github.com/bileg), Leonid Boytsov (http://boytsov.info).
 * With contributions from Lawrence Cayton (http://lcayton.com/) and others.
 *
 * For the complete list of contributors and further details see:
 * https://github.com/searchivarius/NonMetricSpaceLib 
 * 
 * Copyright (c) 2014
 *
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#ifndef _SPACE_LEVENSHTEIN_H_
#define _SPACE_LEVENSHTEIN_H_

#include <string>
#include <map>
#include <stdexcept>

#include <string.h>
#include "distcomp.h"
#include "space/space_string.h"

namespace similarity {

#define SPACE_LEVENSHTEIN "leven"

class SpaceLevenshtein : public StringSpace<int> {
 public:
  virtual ~SpaceLevenshtein() {}
  virtual std::string ToString() const { return "Levenshtein distance"; }
 protected:
  virtual int HiddenDistance(const Object* obj1, const Object* obj2) const {
    CHECK(obj1->datalength() > 0);
    CHECK(obj2->datalength() > 0);
    const char* x = reinterpret_cast<const char*>(obj1->data());
    const char* y = reinterpret_cast<const char*>(obj2->data());
    const size_t len1 = obj1->datalength() / sizeof(char);
    const size_t len2 = obj2->datalength() / sizeof(char);

    return levenshtein(x, len1, y, len2);
  }
};

}  // namespace similarity

#endif