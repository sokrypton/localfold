// chem-probe: each SMILES on stdin as the native chemistry builds its component (one line: atoms | bonds, or ERR and the
// refusal) - the same line tools/check-native-featuriser.py has Node print for shared/chem, compared byte for byte.
#include "chem.h"
#include <iostream>
int main() {
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    try {
      lf::Component c = lf::chem::smilesComponent(line, "LIG");
      std::cout << "OK";
      for (auto& a : c.atoms) std::cout << " " << a.name << ":" << a.element << ":" << lf::jsNumber(a.charge) << ":" << lf::jsNumber(a.x) << ":" << lf::jsNumber(a.y) << ":" << lf::jsNumber(a.z);
      std::cout << " |";
      for (auto& b : c.bonds) std::cout << " " << b.from << "-" << b.to << ":" << b.order;
      std::cout << "\n";
    } catch (const std::exception& e) { std::cout << "ERR " << e.what() << "\n"; }
  }
}
