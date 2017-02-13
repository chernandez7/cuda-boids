
#include "flock.h"

Flock::Flock(int amount) {
    this->boid_amount = amount;
    this->flock = std::vector<Boid>(this->boid_amount);
}

Flock::~Flock() {}

int Flock::getSize() const {
    return boid_amount;
}

Boid Flock::getBoidfromIndex(int i) {
    return flock[i];
}

void Flock::update() {
   for (int i = 0; i < this->boid_amount; i++) {
     for (int j = 0; j < this->boid_amount; j++) {
       this->flock[i].update(flock);
     }
   }
}
