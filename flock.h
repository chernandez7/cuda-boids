
#include <vector>
#include "boid.h"

#ifndef FLOCK_H
#define FLOCK_H

class Flock {
  public:
   Flock();
   Flock(int amount);
   ~Flock();

   int boid_amount;
   std::vector<Boid> flock;

   int getSize() const;
   void addBoid(Boid boid);
   void update();
};

#endif
