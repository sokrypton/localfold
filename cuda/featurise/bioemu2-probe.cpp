// BioEmu2's features from cuda/featurise/bioemu2_features.h, as JSON: every array the JavaScript featuriser
// returns, floats by their bit patterns, so tools/check-native-bioemu2.mjs compares bytes and not decimals.
//
//   bioemu2-probe --job=<AF3 job.json> [--mode=structure|equilibrium] [--contacts=<contacts.json>]
//                 [--smiles-templates=<{chain: {names, element, charge, coords, bonds}} json>] [--ccd-dir=<dir>]
//                 [--steering=<stereo_chemistry.yaml>]   (and the steering tables, hostTables applied)
#include <cstdio>
#include <cstring>
#include <iostream>

#include "bioemu2_steering.h"

using namespace lf;

static std::string arg(int argc, char** argv, const std::string& name) {
  for (int i = 1; i < argc; ++i)
    if (std::strncmp(argv[i], ("--" + name + "=").c_str(), name.size() + 3) == 0) return argv[i] + name.size() + 3;
  return "";
}

int main(int argc, char** argv) {
  try {
    std::string ccdDir = arg(argc, argv, "ccd-dir");
    bioemu2::CcdReader ccd = [&](const std::string& code) {
      if (!ccdDir.empty()) {
        std::string path = ccdDir + "/" + code + ".cif";
        if (FILE* f = std::fopen(path.c_str(), "r")) { std::fclose(f); return readFile(path); }
      }
      return ccdText(code);
    };
    bioemu2::Job job = bioemu2::jobFromAf3Json(parseJson(readFile(arg(argc, argv, "job"))));
    std::vector<bioemu2::Contact> contacts;
    std::string contactsPath = arg(argc, argv, "contacts");
    if (!contactsPath.empty()) contacts = bioemu2::parseContacts(parseJson(readFile(contactsPath)));
    std::map<std::string, bioemu2::Template> smiles;
    std::string smilesPath = arg(argc, argv, "smiles-templates");
    if (!smilesPath.empty()) {
      Json all = parseJson(readFile(smilesPath));
      for (auto& [cid, t] : all.o) {
        bioemu2::Template tt;
        for (auto& v : t.get("names")->a) tt.names.push_back(v.s);
        for (auto& v : t.get("element")->a) tt.element.push_back((int)v.n);
        for (auto& v : t.get("charge")->a) tt.charge.push_back(v.n);
        for (auto& v : t.get("coords")->a) tt.coords.push_back((float)v.n);
        for (auto& v : t.get("bonds")->a) tt.bonds.push_back({(int)v.a.at(0).n, (int)v.a.at(1).n});
        smiles[cid] = tt;
      }
    }
    std::string mode = arg(argc, argv, "mode");
    bioemu2::Features F = bioemu2::featurise(job, ccd, mode.empty() ? "structure" : mode,
                                             contactsPath.empty() ? nullptr : &contacts, smiles);
    std::cout << "{\"tokens\":" << F.tokens << ",\"atoms\":" << F.atoms << ",\"msaRows\":" << F.msaRows << ",\"ints\":{";
    bool first = true;
    for (auto& [k, v] : F.ints) {
      std::cout << (first ? "" : ",") << "\"" << k << "\":[";
      for (size_t i = 0; i < v.size(); ++i) std::cout << (i ? "," : "") << v[i];
      std::cout << "]";
      first = false;
    }
    std::cout << "},\"floats\":{";
    first = true;
    for (auto& [k, v] : F.floats) {
      std::cout << (first ? "" : ",") << "\"" << k << "\":[";
      for (size_t i = 0; i < v.size(); ++i) { uint32_t bits; std::memcpy(&bits, &v[i], 4); std::cout << (i ? "," : "") << bits; }
      std::cout << "]";
      first = false;
    }
    std::cout << "},\"atomKeys\":[";
    for (size_t i = 0; i < F.atomKeys.size(); ++i) {
      auto& [c, r, n] = F.atomKeys[i];
      std::cout << (i ? "," : "") << "[\"" << c << "\"," << r << ",\"" << n << "\"]";
    }
    std::cout << "]";
    std::string stereoPath = arg(argc, argv, "steering");
    if (!stereoPath.empty()) {
      bioemu2::SteeringTables T = bioemu2::steeringTables(F, bioemu2::parseStereoTables(readFile(stereoPath)), &ccd);
      std::cout << ",\"steering\":{\"ints\":{";
      bool f1 = true;
      for (auto& [k, v] : T.ints) {
        std::cout << (f1 ? "" : ",") << "\"" << k << "\":[";
        for (size_t i = 0; i < v.size(); ++i) std::cout << (i ? "," : "") << v[i];
        std::cout << "]";
        f1 = false;
      }
      std::cout << "},\"floats\":{";
      f1 = true;
      for (auto& [k, v] : T.floats) {
        std::cout << (f1 ? "" : ",") << "\"" << k << "\":[";
        for (size_t i = 0; i < v.size(); ++i) { uint32_t bits; std::memcpy(&bits, &v[i], 4); std::cout << (i ? "," : "") << bits; }
        std::cout << "]";
        f1 = false;
      }
      std::cout << "},\"doubles\":{";
      f1 = true;
      char buf[40];
      for (auto& [k, v] : T.doubles) {
        std::cout << (f1 ? "" : ",") << "\"" << k << "\":[";
        for (size_t i = 0; i < v.size(); ++i) { std::snprintf(buf, sizeof buf, "%.17g", v[i]); std::cout << (i ? "," : "") << buf; }
        std::cout << "]";
        f1 = false;
      }
      std::cout << "}}";
    }
    std::cout << "}\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
