
#include "flock.h"

Flock::Flock() {
    this->boid_amount = 100;
    this->flock = std::vector<Boid>(this->boid_amount);
}

Flock::~Flock() {}

int Flock::getSize() const {
    return boid_amount;
}

Boid Flock::getBoidfromIndex(int i) {
    return flock[i];
}

