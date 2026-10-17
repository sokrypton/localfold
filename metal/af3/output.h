// The fold's files and its ranking score's structure terms (output.mm).
#pragma once
#include "af3.h"
#include <string>
#include <vector>

struct Scores { bool clash; double disordered; };
void setOutputInput(const std::string& dir);           // the featurised input the files are written against
// the structure (.pdb, or mmCIF for a .cif path), per-atom pLDDT as B-factors; the dense slot of each atom written
std::vector<size_t> writeStructure(const std::string& path, const std::vector<float>& x, const float* bfactors);
void writeConfidenceFiles(const std::string& path, const std::vector<size_t>& order, int n, int dense, const ConfidenceOut& c,
                          const std::vector<float>& contact, double ranking, const Scores& s);
Scores structureScores(const std::vector<float>& x);
double rankingScore(double ptm, double iptm, const Scores& s);
int scorePdbMain(const std::string& path);
