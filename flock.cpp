
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
