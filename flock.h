
#include <vector>
#include "boid.h"

#ifndef FLOCK_H
#define FLOCK_H

class Flock {
  public:
   Flock();
   ~Flock();
   
   int boid_amount;
   std::vector<Boid> flock;

   Boid getBoidfromIndex(int i);
   int getSize() const;
};

#endif

