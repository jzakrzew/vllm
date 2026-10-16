// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <Python.h>

#include <memory>

#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/pyobject.h>
#include <torch/csrc/stable/tensor.h>

namespace {

PyObject* prepare_tensor_data(PyObject* /*self*/, PyObject* arg) {
  try {
    // Destroy the Python-backed input handle only after reacquiring the GIL.
    const auto input = torch::stable::tensor_from_pyobject(arg);
    auto tensor = input;
    {
      const std::unique_ptr<PyThreadState, decltype(&PyEval_RestoreThread)> gil(
          PyEval_SaveThread(), PyEval_RestoreThread);
      tensor = torch::stable::view(
          torch::stable::contiguous(torch::stable::to(
              torch::stable::flatten(tensor),
              torch::stable::Device(torch::headeronly::DeviceType::CPU))),
          torch::headeronly::ScalarType::Byte);
    }
    return static_cast<PyObject*>(torch::stable::tensor_to_pyobject(tensor));
  } catch (const std::exception& error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
    return nullptr;
  }
}

PyMethodDef methods[] = {
    {"prepare_tensor_data", prepare_tensor_data, METH_O,
     "Flatten a tensor, move it to CPU, and return a contiguous uint8 view."},
    {nullptr, nullptr, 0, nullptr}};

PyModuleDef module = {PyModuleDef_HEAD_INIT, "tensor_data_C", nullptr, -1,
                      methods};

}  // namespace

PyMODINIT_FUNC PyInit_tensor_data_C() { return PyModule_Create(&module); }
