// metal/core/host.h
#include "host.h"
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>

namespace mt {
void writeWhole(const std::string& path, const void* data, size_t bytes) {
  FILE* f = fopen((path + ".tmp").c_str(), "wb");
  if (!f) return;
  fwrite(data, 1, bytes, f); fclose(f);
  rename((path + ".tmp").c_str(), path.c_str());
}
void bestRotation(const std::vector<double>& moving, const std::vector<double>& fixed, double R[9]) {
  double S[3][3] = {};
  for (size_t i = 0; i + 2 < moving.size(); i += 3)
    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) S[a][b] += moving[i + a] * fixed[i + b];
  double N[4][4] = {
    {S[0][0] + S[1][1] + S[2][2], S[1][2] - S[2][1], S[2][0] - S[0][2], S[0][1] - S[1][0]},
    {S[1][2] - S[2][1], S[0][0] - S[1][1] - S[2][2], S[0][1] + S[1][0], S[2][0] + S[0][2]},
    {S[2][0] - S[0][2], S[0][1] + S[1][0], -S[0][0] + S[1][1] - S[2][2], S[1][2] + S[2][1]},
    {S[0][1] - S[1][0], S[2][0] + S[0][2], S[1][2] + S[2][1], -S[0][0] - S[1][1] + S[2][2]}};
  double V[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
  for (int sweep = 0; sweep < 50; ++sweep) {
    double off = 0; for (int p = 0; p < 4; ++p) for (int q = p + 1; q < 4; ++q) off += N[p][q] * N[p][q];
    if (off < 1e-22) break;
    for (int p = 0; p < 4; ++p) for (int q = p + 1; q < 4; ++q) {
      if (std::fabs(N[p][q]) < 1e-300) continue;
      double theta = (N[q][q] - N[p][p]) / (2 * N[p][q]);
      double t = (theta >= 0 ? 1 : -1) / (std::fabs(theta) + std::sqrt(theta * theta + 1)), c = 1 / std::sqrt(t * t + 1), sn = t * c;
      for (int k = 0; k < 4; ++k) { double a = N[k][p], b = N[k][q]; N[k][p] = c * a - sn * b; N[k][q] = sn * a + c * b; }
      for (int k = 0; k < 4; ++k) { double a = N[p][k], b = N[q][k]; N[p][k] = c * a - sn * b; N[q][k] = sn * a + c * b; }
      for (int k = 0; k < 4; ++k) { double a = V[k][p], b = V[k][q]; V[k][p] = c * a - sn * b; V[k][q] = sn * a + c * b; }
    }
  }
  int best = 0; for (int k = 1; k < 4; ++k) if (N[k][k] > N[best][best]) best = k;
  double w = V[0][best], x = V[1][best], y = V[2][best], z = V[3][best];
  double R0[9] = {w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y),
                  2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x),
                  2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z};
  for (int k = 0; k < 9; ++k) R[k] = R0[k];
}
void serveJobs(const char* port, const std::string& dir,
               const std::function<int(const std::string&, const std::vector<std::string>&)>& fold) {
  printf("%s: serving %s\n", port, dir.c_str()); fflush(stdout);
  for (;;) {
    std::string id;
    if (DIR* d = opendir(dir.c_str())) {
      std::vector<std::string> jobs;
      while (dirent* e = readdir(d)) {
        std::string file = e->d_name;
        if (file.size() > 4 && file.substr(file.size() - 4) == ".job") jobs.push_back(file.substr(0, file.size() - 4));
      }
      closedir(d);
      if (!jobs.empty()) { std::sort(jobs.begin(), jobs.end()); id = jobs[0]; }
    }
    if (id.empty()) { usleep(2000); continue; }
    std::string base = dir + "/" + id;
    std::ifstream job(base + ".job");
    std::string input, line; std::getline(job, input);
    std::vector<std::string> flags; while (std::getline(job, line)) if (!line.empty()) flags.push_back(line);
    job.close(); unlink((base + ".job").c_str());
    if (input == "quit") { printf("%s: stopped\n", port); return; }
    fflush(stdout); fflush(stderr);
    int saved = dup(1), savedErr = dup(2), log = open((base + ".log").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    dup2(log, 1); dup2(log, 2); close(log);
    int code = fold(input, flags);
    fflush(stdout); fflush(stderr); dup2(saved, 1); dup2(savedErr, 2); close(saved); close(savedErr);
    FILE* df = fopen((base + ".done.tmp").c_str(), "w"); fprintf(df, "%d\n", code); fclose(df);
    rename((base + ".done.tmp").c_str(), (base + ".done").c_str());
  }
}
}  // namespace mt
