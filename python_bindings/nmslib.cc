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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "init.h"
#include "index.h"
#include "knnquery.h"
#include "knnqueue.h"
#include "methodfactory.h"
#include "space.h"
#include "space/space_vector.h"
#include "spacefactory.h"
#include "space/space_sparse_vector.h"
#include "space/space_l2sqr_sift.h"
#include "thread_pool.h"

#include "cpu_feature_guard.h"

namespace py = pybind11;

namespace similarity {
const char* module_name = "nmslib";
const char* data_suff = ".dat";

enum DistType {
  DISTTYPE_FLOAT,
  DISTTYPE_INT,
  DISTTYPE_DEFAULT
};

enum DataType {
  DATATYPE_DENSE_VECTOR,
  DATATYPE_DENSE_UINT8_VECTOR,
  DATATYPE_SPARSE_VECTOR,
  DATATYPE_OBJECT_AS_STRING,
};

// forward references
template <typename dist_t, typename dist_uint_t = dist_t> void exportIndex(py::module * m);
template <typename dist_t, typename dist_uint_t = dist_t> std::string distName();
AnyParams loadParams(py::object o);
void exportLegacyAPI(py::module * m);
void freeAndClearObjectVector(ObjectVector& data);

// Wrap a space/objectvector/index together for ease of use
template <typename dist_t, typename dist_uint_t = dist_t>
struct IndexWrapper {
  IndexWrapper(const std::string & method,
        const std::string & space_type,
        py::object space_params,
        DataType data_type,
        DistType dist_type,
        DistType dist_uint_type)
      : method(method), space_type(space_type), data_type(data_type), dist_type(dist_type), dist_uint_type(dist_uint_type),
        space(SpaceFactoryRegistry<dist_t>::Instance().CreateSpace(space_type,
                                                                   loadParams(space_params))) {
    auto vectSpacePtr = dynamic_cast<VectorSpace<dist_t, dist_uint_t>*>(space.get());
    if (data_type == DATATYPE_DENSE_VECTOR && vectSpacePtr == nullptr) {
      throw std::invalid_argument("The space type " + space_type +
                                  " is not compatible with the type DENSE_VECTOR, only dense vector spaces are allowed!");
    }
    auto vectSiftPtr = dynamic_cast<SpaceL2SqrSift*>(space.get());
    if (data_type == DATATYPE_DENSE_UINT8_VECTOR && vectSiftPtr == nullptr) {
      throw std::invalid_argument("The space type " + space_type +
                                  " is not compatible with the type DENSE_UINT8_VECTOR!");
    }
  }

  void createIndex(py::object index_params, bool print_progress = false) {
    AnyParams params = loadParams(index_params);

    py::gil_scoped_release l;
    auto factory = MethodFactoryRegistry<dist_t>::Instance();
    index.reset(factory.CreateMethod(print_progress, method, space_type, *space, data));
    index->CreateIndex(params);
  }

  void loadIndex(const std::string & filename, bool load_data = false) {
    py::gil_scoped_release l;
    auto factory = MethodFactoryRegistry<dist_t>::Instance();
    bool print_progress=false; // We are not going to creat the index anyways, only to load an existing one
    index.reset(factory.CreateMethod(print_progress, method, space_type, *space, data));
    if (load_data) {
      vector<string> dummy;
      freeAndClearObjectVector(data);
      space->ReadObjectVectorFromBinData(data, dummy, filename + data_suff);
    }
    index->LoadIndex(filename);

    // querying reloaded indices don't seem to work correctly (at least hnsw ones) until
    // SetQueryTimeParams is called
    index->ResetQueryTimeParams();
  }

  void saveIndex(const std::string & filename, bool save_data = false) {
    if (!index) {
      throw std::invalid_argument("Must call createIndex or loadIndex before this method");
    }
    py::gil_scoped_release l;
    if (save_data) {
      vector<string> dummy;
      space->WriteObjectVectorBinData(data, dummy, filename + data_suff);
    }
    index->SaveIndex(filename);
  }

  py::object knnQuery(py::object input, size_t k) {
    if (!index) {
      throw std::invalid_argument("Must call createIndex or loadIndex before this method");
    }

    std::unique_ptr<const Object> query(readObject(input));
    KNNQuery<dist_t> knn(*space, query.get(), k);
    {
      py::gil_scoped_release l;
      index->Search(&knn, -1);
    }
    std::unique_ptr<KNNQueue<dist_t>> res(knn.Result()->Clone());
    return convertResult(res.get());
  }

  py::object knnQueryBatch(py::object input, size_t k, int num_threads) {
    if (!index) {
      throw std::invalid_argument("Must call createIndex or loadIndex before this method");
    }

    ObjectVector queries;
    readObjectVector(input, &queries);
    std::vector<std::unique_ptr<KNNQueue<dist_t>>> results(queries.size());
    {
      py::gil_scoped_release l;

      ParallelFor(0, queries.size(), num_threads, [&](size_t query_index, size_t threadId) {
        KNNQuery<dist_t> knn(*space, queries[query_index], k);
        index->Search(&knn, -1);
        results[query_index].reset(knn.Result()->Clone());
      });

      // TODO(@benfred): some sort of RAII auto-destroy for this
      freeAndClearObjectVector(queries);
    }

    py::list ret;
    for (auto & result : results) {
      ret.append(convertResult(result.get()));
    }
    return ret;
  }

  py::object convertResult(KNNQueue<dist_t> * res) {
    // Create numpy arrays for the output
    size_t size = res->Size();
    py::array_t<int> ids(size);
    py::array_t<dist_t> distances(size);

    while (!res->Empty() && size > 0) {
      // iterating here in reversed order, undo that
      size -= 1;
      ids.mutable_at(size) = res->TopObject()->id();
      distances.mutable_at(size) = res->TopDistance();
      res->Pop();
    }
    return py::make_tuple(ids, distances);
  }

  const Object * readObject(py::object input, int id = 0) {
    switch (data_type) {
      case DATATYPE_DENSE_VECTOR: {
        py::array_t<dist_uint_t, py::array::c_style | py::array::forcecast> temp(input);
        std::vector<dist_uint_t> tempVect(temp.data(0), temp.data(0) + temp.size());
        auto vectSpacePtr = reinterpret_cast<VectorSpace<dist_t, dist_uint_t> *>(space.get());
        return vectSpacePtr->CreateObjFromVect(id, -1, tempVect);
        // This way it will not always work properly
        //return new Object(id, -1, temp.size() * sizeof(dist_uint_t), temp.data(0));
      }
      case DATATYPE_DENSE_UINT8_VECTOR: {
        py::array_t<uint8_t> temp(input);
        std::vector<uint8_t> tempVect(temp.data(0), temp.data(0) + temp.size());
        auto vectSiftPtr = reinterpret_cast<SpaceL2SqrSift*>(space.get());
        return vectSiftPtr->CreateObjFromUint8Vect(id, -1, tempVect);
      }
      case DATATYPE_OBJECT_AS_STRING: {
        std::string temp = py::cast<std::string>(input);
        return space->CreateObjFromStr(id, -1, temp.c_str(), NULL).release();
      }
      case DATATYPE_SPARSE_VECTOR: {
        // Sparse vectors are expected to be list of (id, value) tuples
        std::vector<SparseVectElem<dist_t>> output;
        py::list items(input);
        for (size_t i = 0; i < items.size(); ++i) {
          py::tuple current(items[i]);
          output.push_back(SparseVectElem<dist_t>(py::cast<uint32_t>(current[0]),
                                                  py::cast<dist_t>(current[1])));
        }
        std::sort(output.begin(), output.end());
        auto sparse = reinterpret_cast<const SpaceSparseVector<dist_t>*>(space.get());
        return sparse->CreateObjFromVect(id, -1, output);
      }
      default:
        throw std::invalid_argument("Unknown data type for readObject");
    }
  }

  // reads multiple items from a python object and inserts onto a similarity::ObjectVector
  // returns the number of elements inserted
  size_t readObjectVector(py::object input, ObjectVector * output,
                          py::object ids_ = py::none()) {
    std::vector<int> ids;
    if (!ids_.is_none()) {
      ids = py::cast<std::vector<int>>(ids_);
    }

    if (py::isinstance<py::list>(input)) {
      py::list items(input);
      for (size_t i = 0; i < items.size(); ++i) {
        output->push_back(readObject(items[i], ids.size() ? ids.at(i) : i));
      }
      return items.size();

    } else if (data_type == DATATYPE_DENSE_VECTOR) {
      // allow numpy arrays to be returned here too
      py::array_t<dist_uint_t, py::array::c_style | py::array::forcecast> items(input);
      auto buffer = items.request();
      if (buffer.ndim != 2) throw std::runtime_error("data must be a 2d array");

      size_t rows = buffer.shape[0], features = buffer.shape[1];
      std::vector<dist_uint_t> tempVect(features);
      auto vectSpacePtr = reinterpret_cast<VectorSpace<dist_t, dist_uint_t>*>(space.get());
      for (size_t row = 0; row < rows; ++row) {
        int id = ids.size() ? ids.at(row) : row;
        const dist_uint_t *elemVecStart = items.data(row);
        std::copy(elemVecStart, elemVecStart + features, tempVect.begin());
        output->push_back(vectSpacePtr->CreateObjFromVect(id, -1, tempVect));
        //this way it won't always work properly
        //output->push_back(new Object(id, -1, features * sizeof(dist_t), items.data(row)));
      }
      return rows;
    } else if (data_type == DATATYPE_DENSE_UINT8_VECTOR) {
      // allow numpy arrays to be returned here too
      py::array_t<uint8_t, py::array::c_style | py::array::forcecast> items(input);
      auto buffer = items.request();
      if (buffer.ndim != 2) throw std::runtime_error("data must be a 2d array");

      size_t rows = buffer.shape[0], features = buffer.shape[1];
      std::vector<uint8_t> tempVect(features);
      auto vectSiftPtr = reinterpret_cast<SpaceL2SqrSift*>(space.get());
      for (size_t row = 0; row < rows; ++row) {
        int id = ids.size() ? ids.at(row) : row;
        const uint8_t* elemVecStart = items.data(row);
        std::copy(elemVecStart, elemVecStart + features, tempVect.begin());
        output->push_back(vectSiftPtr->CreateObjFromUint8Vect(id, -1, tempVect));
      }
      return rows;

    } else if (data_type == DATATYPE_SPARSE_VECTOR) {
      // the attr calls will fail with an attribute error, but this fixes the legacy
      // unittest case
      if (!py::hasattr(input, "indptr")) {
        throw py::value_error("expect CSR matrix here");
      }

      // try to intrepret input data as a CSR matrix
      py::array_t<int> indptr(input.attr("indptr"));
      py::array_t<int> indices(input.attr("indices"));
      py::array_t<dist_t> sparse_data(input.attr("data"));

      // read each row from the sparse matrix, and insert
      auto sparse_space = reinterpret_cast<const SpaceSparseVector<dist_t>*>(space.get());
      std::vector<SparseVectElem<dist_t>> sparse_items;
      for (int rowid = 0; rowid < indptr.size() - 1; ++rowid) {
        sparse_items.clear();

        for (int i = indptr.at(rowid); i < indptr.at(rowid + 1); ++i) {
          sparse_items.push_back(SparseVectElem<dist_t>(indices.at(i),
                                 sparse_data.at(i)));
        }
        std::sort(sparse_items.begin(), sparse_items.end());

        int id = ids.size() ? ids.at(rowid) : rowid;
        output->push_back(sparse_space->CreateObjFromVect(id, -1, sparse_items));
      }
      return indptr.size() - 1;
    }

    throw std::invalid_argument("Unknown data type");
  }

  py::object writeObject(const Object * obj) {
    switch (data_type) {
      case DATATYPE_DENSE_VECTOR: {
        auto vectSpacePtr = reinterpret_cast<VectorSpace<dist_t, dist_uint_t>*>(space.get());
        py::list ret;
        const dist_uint_t *values = reinterpret_cast<const dist_uint_t *>(obj->data());
        size_t elemQty = vectSpacePtr->GetElemQty(obj);
        for (size_t i = 0; i < elemQty; ++i) {
          ret.append(py::cast(values[i]));
        }
        return ret;
      }
      case DATATYPE_OBJECT_AS_STRING: {
        return py::cast(space->CreateStrFromObj(obj, ""));
      }
      case DATATYPE_SPARSE_VECTOR: {
        auto values = reinterpret_cast<const SparseVectElem<dist_t>*>(obj->data());
        size_t count = obj->datalength() / sizeof(SparseVectElem<dist_t>);
        py::list ret;
        for (size_t i = 0; i < count; ++i) {
          ret.append(py::make_tuple(values[i].id_, values[i].val_));
        }
        return ret;
      }
      default:
        throw std::runtime_error("Unknown data_type");
    }
  }

  size_t addDataPoint(int id, py::object input) {
    data.push_back(readObject(input, id));
    return data.size() - 1;
  }

  size_t addDataPointBatch(py::object input, py::object ids = py::none()) {
    return readObjectVector(input, &data, ids);
  }

  inline size_t size() const { return data.size(); }

  py::object at(size_t pos) { return writeObject(data.at(pos)); }

  dist_t getDistance(size_t pos1, size_t pos2) const {
    py::gil_scoped_release l;
    return space->IndexTimeDistance(data.at(pos1), data.at(pos2));
  }

  std::string repr() const {
    std::stringstream ret;
    ret << "<" << module_name << "." << distName<dist_t, dist_uint_t>() << "Index method='" << method
        << "' space='" << space_type <<  "' at " << this << ">";
    return ret.str();
  }

  ~IndexWrapper() {
    // In cases when the interpreter was shutting down, attempting to log in python
    // could throw an exception (https://github.com/nmslib/nmslib/issues/327).
    //LOG(LIB_DEBUG) << "Destroying Index";
    freeAndClearObjectVector(data);
  }

  std::string method;
  std::string space_type;
  DataType data_type;
  DistType dist_type;
  DistType dist_uint_type;
  std::unique_ptr<Space<dist_t>> space;
  std::unique_ptr<Index<dist_t>> index;
  ObjectVector data;
};

// pybind11::gil_scoped_acquire can deadlock when acquiring the GIL on threads
// created from python (https://github.com/nmslib/nmslib/issues/291)
// This might be fixed in a future version of pybind11 (https://github.com/pybind/pybind11/pull/1211)
// but until then, lets fall back to the python c-api to fix.
struct AcquireGIL {
  PyGILState_STATE state;
  AcquireGIL()
    : state(PyGILState_Ensure()) {
  }
  ~AcquireGIL() {
    PyGILState_Release(state);
  }
};

class PythonLogger
  : public Logger {
 public:
  py::object inner;
  explicit PythonLogger(const py::object & inner)
    : inner(inner) {
  }

  void log(LogSeverity severity,
           const char * file,
           int line,
           const char * function,
           const std::string & message) {
    // In cases when the interpreter was shutting down, attempting to log in python
    // could throw an exception (https://github.com/nmslib/nmslib/issues/327).
    // Logging shouldn't cause exceptions, so catch it and dump to stderr instead.
    try {
      AcquireGIL l;
      switch(severity) {
        case LIB_DEBUG:
          inner.attr("debug")(message);
          break;
        case LIB_INFO:
          inner.attr("info")(message);
          break;
        case LIB_WARNING:
          inner.attr("warning")(message);
          break;
        case LIB_ERROR:
          inner.attr("error")(message);
          break;
        case LIB_FATAL:
          inner.attr("critical")(message);
          break;
      }
    } catch (...) {
      // This is almost certainly due to python process shut down.
      // Just write the message out to stderr if its not a debug message
      if (severity != LIB_DEBUG) {
        StdErrLogger().log(severity, file, line, function, message);
      }
    }
  }
};

#ifdef PYBIND11_MODULE
PYBIND11_MODULE(nmslib, m) {
  tensorflow::port::InfoAboutUnusedCPUFeatures();
  m.doc() = "Python Bindings for Non-Metric Space Library (NMSLIB)";
#else
PYBIND11_PLUGIN(nmslib) {
  py::module m(module_name, "Python Bindings for Non-Metric Space Library (NMSLIB)");
#endif
  // Log using the python logger, instead of defaults built in here
  py::module logging = py::module::import("logging");
  py::object nmslibLogger = logging.attr("getLogger")("nmslib");
  setGlobalLogger(new PythonLogger(nmslibLogger));

  initLibrary(0 /* seed */, LIB_LOGCUSTOM, NULL);

#define STRINGIZER(x)   #x

//
// It looks like newer version of Python >3.9 just enclose VERSION_INFO in "",
// when "" are missing. Not the older ones, previously this was possible to fix
// at the level of setup.py by passing an escaping character, but this stopped
// working after circa 2000 for some reason. In any case, STRINGIZER works just
// fine and when applied to the quoted string, it doesn't change it. Thus
// it works for both old and new Python versions.
//
#ifdef VERSION_INFO
  m.attr("__version__") = py::str(STRINGIZER(VERSION_INFO));
#else
  m.attr("__version__") = py::str("dev");
#endif

  py::enum_<DistType>(m, "DistType")
    .value("FLOAT", DISTTYPE_FLOAT)
    .value("INT", DISTTYPE_INT);

  py::enum_<DataType>(m, "DataType")
    .value("DENSE_VECTOR", DATATYPE_DENSE_VECTOR)
    .value("DENSE_UINT8_VECTOR", DATATYPE_DENSE_UINT8_VECTOR)
    .value("SPARSE_VECTOR", DATATYPE_SPARSE_VECTOR)
    .value("OBJECT_AS_STRING", DATATYPE_OBJECT_AS_STRING);

  // Initializes a new index. Param ordering here is set to be consistent with the previous
  // version of the bindings
  m.def(
      "init",
      [](const std::string &space, py::object space_params, const std::string &method,
         DataType data_type, DistType dtype, DistType distUintType)
      {
        if (DISTTYPE_DEFAULT == distUintType)
        {
          distUintType = dtype;
        }

        py::object ret = py::none();
        if (DISTTYPE_FLOAT == dtype && DISTTYPE_FLOAT == distUintType) {
          auto index = new IndexWrapper<float, float>(method, space, space_params, data_type, dtype, distUintType);
          ret = py::cast(index, py::return_value_policy::take_ownership);
        } else if (DISTTYPE_FLOAT == dtype && DISTTYPE_INT == distUintType) {
          auto index = new IndexWrapper<float, int>(method, space, space_params, data_type, dtype, distUintType);
          ret = py::cast(index, py::return_value_policy::take_ownership);
        } else if (DISTTYPE_INT == dtype && DISTTYPE_FLOAT == distUintType) {
          auto index = new IndexWrapper<int, float>(method, space, space_params, data_type, dtype, distUintType);
          ret = py::cast(index, py::return_value_policy::take_ownership);
        } else if (DISTTYPE_INT == dtype && DISTTYPE_INT == distUintType) {
          auto index = new IndexWrapper<int, int>(method, space, space_params, data_type, dtype, distUintType);
          ret = py::cast(index, py::return_value_policy::take_ownership);
        } else {
          // should never happen
          throw std::invalid_argument("Invalid DistType");
        }
  
        return ret;
      },
      py::arg("space") = "cosinesimil",
      py::arg("space_params") = py::none(),
      py::arg("method") = "hnsw",
      py::arg("data_type") = DATATYPE_DENSE_VECTOR,
      py::arg("dtype") = DISTTYPE_FLOAT,
      py::arg("dist_uint_type") = DISTTYPE_DEFAULT,
      "This function initializes a new NMSLIB index\n\n"
      "Parameters\n"
      "----------\n"
      "space: str optional\n"
      "    The metric space to create for this index\n"
      "space_params: dict optional\n"
      "    Parameters for configuring the space\n"
      "method: str optional\n"
      "    The index method to use\n"
      "data_type: nmslib.DataType optional\n"
      "    The type of data to index (dense/sparse/string vectors)\n"
      "\n"
      "Returns\n"
      "----------\n"
      "    A new NMSLIB Index.\n");

  // Export Different Types of NMS Indices and spaces
  // hiding in a submodule to avoid cluttering up main namespace
  py::module dist_module = m.def_submodule("dist",
    "Contains Indexes and Spaces for different Distance Types");
  exportIndex<int>(&dist_module);
  exportIndex<float>(&dist_module);
  exportIndex<float, int>(&dist_module);
  exportIndex<int, float>(&dist_module);

  exportLegacyAPI(&m);

#ifndef PYBIND11_MODULE
  return m.ptr();
#endif
}

template <typename dist_t, typename dist_uint_t = dist_t>
void exportIndex(py::module * m) {
  // Export the index
  std::string index_name = distName<dist_t, dist_uint_t>() + "Index";
  py::class_<IndexWrapper<dist_t, dist_uint_t>>(*m, index_name.c_str())
    .def("createIndex", &IndexWrapper<dist_t, dist_uint_t>::createIndex,
      py::arg("index_params") = py::none(),
      py::arg("print_progress") = false,
      "Creates the index, and makes it available for querying\n\n"
      "Parameters\n"
      "----------\n"
      "index_params: dict optional\n"
      "    Dictionary of optional parameters to use in indexing\n"
      "print_progress: bool optional\n"
      "    Whether or not to display progress bar when creating index\n")

    .def("knnQuery", &IndexWrapper<dist_t, dist_uint_t>::knnQuery,
      py::arg("vector"), py::arg("k") = 10,
      "Finds the approximate K nearest neighbours of a vector in the index \n\n"
      "Parameters\n"
      "----------\n"
      "vector: array_like\n"
      "    A 1D vector to query for.\n"
      "k: int optional\n"
      "    The number of neighbours to return\n"
      "\n"
      "Returns\n"
      "----------\n"
      "ids: array_like.\n"
      "    A 1D vector of the ids of each nearest neighbour.\n"
      "distances: array_like.\n"
      "    A 1D vector of the distance to each nearest neigbhour.\n")

    .def("knnQueryBatch", &IndexWrapper<dist_t, dist_uint_t>::knnQueryBatch,
      py::arg("queries"), py::arg("k") = 10, py::arg("num_threads") = 0,
      "Performs multiple queries on the index, distributing the work over \n"
      "a thread pool\n\n"
      "Parameters\n"
      "----------\n"
      "input: list\n"
      "    A list of queries to query for\n"
      "k: int optional\n"
      "    The number of neighbours to return\n"
      "num_threads: int optional\n"
      "    The number of threads to use\n"
      "\n"
      "Returns\n"
      "----------\n"
      "list:\n"
      "   A list of tuples of (ids, distances)\n ")

    .def("loadIndex", &IndexWrapper<dist_t, dist_uint_t>::loadIndex,
      py::arg("filename"),
      py::arg("load_data") = false,
      "Loads the index from disk\n\n"
      "Parameters\n"
      "----------\n"
      "filename: str\n"
      "    The filename to read from\n"
      "load_data: bool optional\n"
      "    Whether or not to load previously saved data.\n")

    .def("saveIndex", &IndexWrapper<dist_t, dist_uint_t>::saveIndex,
      py::arg("filename"),
      py::arg("save_data") = false,
      "Saves the index and/or data to disk\n\n"
      "Parameters\n"
      "----------\n"
      "filename: str\n"
      "    The filename to save to\n"
      "save_data: bool optional\n"
      "    Whether or not to save data\n")

    .def("setQueryTimeParams",
      [](IndexWrapper<dist_t, dist_uint_t> * self, py::object params) {
        self->index->SetQueryTimeParams(loadParams(params));
      }, py::arg("params") = py::none(),
      "Sets parameters used in knnQuery.\n\n"
      "Parameters\n"
      "----------\n"
      "params: dict\n"
      "    A dictionary of params to use in querying. Setting params to None will reset\n")

    .def("addDataPoint", &IndexWrapper<dist_t, dist_uint_t>::addDataPoint,
      py::arg("id"),
      py::arg("data"),
      "Adds a single datapoint to the index\n\n"
      "Parameters\n"
      "----------\n"
      "id: int\n"
      "    The id of the object to add\n"
      "data: object\n"
      "    The object to add to the index.\n"
      "Returns\n"
      "----------\n"
      "int\n"
      "    The position the item was added at\n")

    .def("addDataPointBatch", &IndexWrapper<dist_t, dist_uint_t>::addDataPointBatch,
      py::arg("data"),
      py::arg("ids") = py::none(),
      "Adds multiple datapoints to the index\n\n"
      "Parameters\n"
      "----------\n"
      "data: object\n"
      "    The objects to add to the index.\n"
      "ids: array_like optional\n"
      "    The ids of the object being inserted. If not set will default to the \n"
      "    row id of each object in the dataset\n"
      "Returns\n"
      "----------\n"
      "int\n"
      "    The number of items added\n")

    .def_readonly("dataType", &IndexWrapper<dist_t, dist_uint_t>::data_type)
    .def_readonly("distType", &IndexWrapper<dist_t, dist_uint_t>::dist_type)
    .def_readonly("distUintType", &IndexWrapper<dist_t, dist_uint_t>::dist_uint_type)
    .def("__len__", &IndexWrapper<dist_t, dist_uint_t>::size)
    .def("__getitem__", &IndexWrapper<dist_t, dist_uint_t>::at)
    .def("getDistance", &IndexWrapper<dist_t, dist_uint_t>::getDistance)
    .def("__repr__", &IndexWrapper<dist_t, dist_uint_t>::repr);
}

template <> std::string distName<int>() { return "Int"; }
template <> std::string distName<float>() { return "Float"; }
template <> std::string distName<float, int>() { return "FloatWithIntInput"; }
template <> std::string distName<int, float>() { return "IntWithFloatInput"; }

void freeAndClearObjectVector(ObjectVector& data) {
  for (auto datum : data) {
    delete datum;
  }
  data.clear();
}

AnyParams loadParams(py::object o) {
  if (o.is_none()) {
    return AnyParams();
  }

  // if we're given a list of strings like "['key=value', 'key2=value2']",
  if (py::isinstance<py::list>(o)) {
    return AnyParams(py::cast<std::vector<std::string>>(o));
  }

  if (py::isinstance<py::dict>(o)) {
    AnyParams ret;
    py::dict items(o);
    for (auto item : items) {
      std::string key = py::cast<std::string>(item.first);
      auto & value = item.second;

      // allow param values to be string/int/double
      if (py::isinstance<py::int_>(value)) {
        ret.AddChangeParam(key, py::cast<int>(value));
      } else if (py::isinstance<py::float_>(value)) {
        ret.AddChangeParam(key, py::cast<double>(value));
      } else if (py::isinstance<py::str>(value))
      {
        ret.AddChangeParam(key, py::cast<std::string>(value));
      } else {
        std::stringstream err;
        err << "Unknown type for parameter '" << key << "'";
        throw std::invalid_argument(err.str());
      }
    }
    return ret;
  }

  throw std::invalid_argument("Unknown type for parameters");
}

/// Function Definitions for backwards compatibility
void exportLegacyAPI(py::module * m) {
  m->def("addDataPoint", [](py::object self, int id, py::object datum) {
    return self.attr("addDataPoint")(id, datum);
  });
  m->def("addDataPointBatch", [](py::object self, py::object ids, py::object data) {
    // There are multiple unittests that expect this function to raise a ValueError
    // if the types aren't numpy arrays. (the newer api will work and convert
    // the types if lists are passed in etc - while the legacy api expects an exception)
    if (!py::isinstance<py::array_t<int>>(ids)) {
      throw py::value_error("Invalid datatype for ids in addDataPointBatch");
    }

    // Also ensure data is a matrix of the right type
    DataType data_type = py::cast<DataType>(self.attr("dataType"));
    if (data_type == DATATYPE_DENSE_VECTOR) {
      DistType dist_type = py::cast<DistType>(self.attr("distUintType"));
      if (((dist_type == DISTTYPE_FLOAT) && (!py::isinstance<py::array_t<float>>(data)))  ||
          ((dist_type == DISTTYPE_INT) && (!py::isinstance<py::array_t<int>>(data)))) {
        throw py::value_error("Invalid datatype for data in addDataPointBatch");
      }
    }
    if (data_type == DATATYPE_DENSE_UINT8_VECTOR) {
      DistType dist_type = py::cast<DistType>(self.attr("distType"));
      if (! ((dist_type == DISTTYPE_FLOAT) && (py::isinstance<py::array_t<uint8_t>>(data))) ) {
        throw py::value_error("Invalid datatype for data in addDataPointBatch");
      }

    }

    size_t offset = py::len(self);
    int insertions = py::cast<int>(self.attr("addDataPointBatch")(data, ids));

    py::array_t<int> positions(insertions);
    for (int i = 0; i < insertions; ++i) {
      positions.mutable_at(i) = offset + i;
    }
    return positions;
  });
  m->def("setQueryTimeParams", [](py::object self, py::object params) {
    return self.attr("setQueryTimeParams")(params);
  });
  m->def("createIndex", [](py::object self, py::object index_params) {
    return self.attr("createIndex")(index_params);
  });
  m->def("saveIndex", [](py::object self, py::object filename) {
    return self.attr("saveIndex")(filename);
  });
  m->def("loadIndex", [](py::object self, py::object filename) {
    return self.attr("loadIndex")(filename);
  });
  m->def("knnQuery", [](py::object self, size_t k, py::object query) {
    // knnQuery now returns a tuple of ids/distances numpy arrays
    // previous version returns list of just ids. convert
    py::tuple ret = self.attr("knnQuery")(query, k);
    py::list ids(ret[0]);
    return ids;
  });
  m->def("getDataPoint", [](py::object self, size_t pos) {
    return self.attr("__getitem__")(pos);
  });
  m->def("getDataPointQty", [](py::object self) {
    return py::len(self);
  });
  m->def("getDistance", [](py::object self, size_t pos1, size_t post2) {
    return self.attr("getDistance")(pos1, post2);
  });

  m->def("knnQueryBatch", [](py::object self, int num_threads, int k, py::object queries) {
    py::list results = self.attr("knnQueryBatch")(queries, k, num_threads);

    // return plain lists of just the ids
    py::list ret;
    for (size_t i = 0; i < results.size(); ++i) {
      py::tuple current(results[i]);
      ret.append(py::list(current[0]));
    }
    return ret;
  });

  m->def("freeIndex", [](py::object self) { });
}
}  // namespace similarity
