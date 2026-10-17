// Host pieces every native port's driver shares: a file written whole, the frames' superposition, and the resident
// --serve mode the worker drives.
#pragma once
#include <functional>
#include <string>
#include <vector>

namespace mt {
// written to <path>.tmp and renamed into place: a reader never sees part of it
void writeWhole(const std::string& path, const void* data, size_t bytes);
// the rotation taking `moving` onto `fixed` (both centred, xyz triples), Horn's quaternion
void bestRotation(const std::vector<double>& moving, const std::vector<double>& fixed, double R[9]);
// --serve=<dir>: stay up, the weights resident, folding each <id>.job dropped there (its first line the input directory,
// then its flags a line); its output in <id>.log and its exit code in <id>.done. A job whose input is "quit" stops it.
void serveJobs(const char* port, const std::string& dir,
               const std::function<int(const std::string& input, const std::vector<std::string>& flags)>& fold);
}  // namespace mt
