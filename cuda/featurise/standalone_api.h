// The one declaration a port's .cu needs to run as a standalone command (cuda/featurise/standalone.cpp, compiled by
// g++ exactly as the featuriser binary is and linked in): `port` is "af3", "af2" or "ef2", `fold` the port's
// own main over a featurised input directory. See cuda/featurise/standalone.h.
#pragma once
namespace lf::standalone {
int main(const char* port, int argc, char** argv, int (*fold)(int, char**));
}
