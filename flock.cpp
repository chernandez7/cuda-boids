
#include "flock.h"

Flock::Flock() {
   this->boid_amount = 0;
   this->flock = std::vector<Boid>(0);
}

Flock::Flock(int amount) {
    this->boid_amount = amount;
    this->flock = std::vector<Boid>(this->boid_amount);
}

Flock::~Flock() {}

int Flock::getSize() const {
    return boid_amount;
}

void Flock::update() {
   for (int i = 0; i < this->boid_amount; i++) {
     for (int j = 0; j < this->boid_amount; j++) {
       this->flock[i].update(flock);
     }
   }
}

void Flock::addBoid(Boid boid) {
   this->flock.push_back(boid);
   this->boid_amount++;
}

