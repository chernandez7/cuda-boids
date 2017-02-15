
#ifndef FLOCK_H
#define FLOCK_H

#include <vector>
#include "boid.h"

class Flock {
  public:
   Flock();
   Flock(int amount);
   ~Flock();

   int boid_amount;
   std::vector<Boid> flock;

   int getSize();
   void addBoid(Boid boid);
   void update();
   Boid getBoidFromIndex(int index);
};

#endif
