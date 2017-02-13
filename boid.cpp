
#include <cstdlib>
#include "boid.h"

Boid::Boid() {
    this->acceleration = Vector3f(0, 0, 0);
    this->velocity = Vector3f(rand() % 3 - 2, rand() % 3 - 2, rand() % 3 - 2);
    this->position = Vector3f(0, 0, 0);
    this->maxSpeed = 3.5;
    this->maxForce = 0.5;

    this->desiredSeparation = 20;
    this->desiredAlignment = 70;
    this->desiredCohesion = 25;

    this->separationWeight = 1.5;
    this->alignmentWeight = 1.0;
    this->cohesionWeight = 1.0;
}

Boid::Boid(Vector3f postition) {
    this->acceleration = Vector3f(0, 0, 0);
    this->velocity = Vector3f(rand() % 3 - 2, rand() % 3 - 2, rand() % 3 - 2);
    this->position = Vector3f(position.x, position.y, position.z);
    this->maxSpeed = 3.5;
    this->maxForce = 0.5;

    this->desiredSeparation = 20;
    this->desiredAlignment = 70;
    this->desiredCohesion = 25;

    this->separationWeight = 1.5;
    this->alignmentWeight = 1.0;
    this->cohesionWeight = 1.0;
}

Boid::~Boid() {}

void Boid::applyForce(Vector3f position) {
    this->acceleration.addVectors(position);
}

Vector3f Boid::seek(Vector3f sum) {
    Vector3f desired = Vector3f(0, 0, 0);
    desired.subVectors(sum);
    desired.normalize();
    desired.mulByScalar(this->maxSpeed);

    float deltaX = desired.getX() - this->velocity.getX();
    float deltaY = desired.getY() - this->velocity.getY();
    float deltaZ = desired.getZ() - this->velocity.getZ();
    this->acceleration.setCoords(deltaX, deltaY, deltaZ);
    return acceleration;
}

//void Boid::borders() {}

void Boid::update(vector<Boid> flock) {
  Vector3f separation = SeparationRule(flock);
  Vector3f alignment = AlignmentRule(flock);
  Vector3f cohesion = CohesionRule(flock);

  separation.mulByScalar(this->separationWeight);
  alignment.mulByScalar(this->alignmentWeight);
  cohesionWeight.mulByScalar(this->cohesionWeight);

  applyForce(separation);
  applyForce(alignment);
  applyForce(cohesion);

  this->acceleration.mulByScalar(.35);
  this->velocity.addVectors(this->acceleration);
  this->velocity.limitVector(this->maxSpeed);
  this->position.addVectors(this->velocity);

  this->acceleration.mulByScalar(0);

  //borders();
}

Vector3f Boid::SeparationRule(std::vector<Boid> flock) {
  Vector3f steer = Vector3f(0, 0, 0);
  int count = 0;
  for (int i = 0;i < flock.size(); i++) {
    float d = this->position.distanceBetweenVectors(flock[i].getPosition());
    // Move away from fellow boid if too close
    if (d > 0 && d < this->desiredSeparation) {
     Vector3f delta = Vector3f(0, 0, 0);
     delta = this->position.diffVectors(flock[i].getPosition());
     delta.normalize();
     delta.divByScalar(d);
     steer.addVectors(delta);
     count++;
     }
   }
  if (count > 0) {
   steer.divByScalar(count);
  }
  if (steer.calcMagnitude() > 0) {
   steer.normalize();
   steer.mulByScalar(this->maxSpeed);
   steer.subVectors(this->velocity);
   steer.limitVector(this->maxForce);
  }
   return steer;
}

Vector3f Boid::CohesionRule(std::vector<Boid> flock) {
    Vector3f sum = Vector3f(0, 0, 0);
    int count = 0;
    for (int i = 0; i < flock.size(); i++) {
        float d = this->position.distanceBetweenVectors(flock[i].getPosition());

        if (d > 0 && d < this->desiredCohesion) {
            sum.addVectors(flock[i].getPosition());
            count++;
        }
    }
    if (count > 0) {
        sum.divByScalar(count);
        return seek(sum);
    } else {
        sum = Vector3f(0, 0, 0);
        return sum;
    }
}

Vector3f Boid::AlignmentRule(std::vector<Boid> flock) {
    Vector3f sum = Vector3f(0, 0, 0);
    int count = 0;
    for (int i = 0; i < flock.size(); i++) {
        float d = this->position.distanceBetweenVectors(flock[i].getPosition());

        if (d > 0 && d < this->desiredAlignment) {
            sum.addVectors(flock[i].getVelocity());
            count++;
        }
    }
   // If there are boids nearby
   if (count > 0) {
   sum.divByScalar(count);
   sum.normalize();
   sum.mulByScalar(maxSpeed);

   Vector3f steer;
   float deltaX = sum.getX() - this->velocity.getX();
   float deltaY = sum.getY() - this->velocity.getY();
   float deltaZ = sum.getZ() - this->velocity.getZ();
   steer = Vector3f(deltaX, deltaY, deltaZ);
   steer.limitVector(this->maxForce);
   return steer;
   } else {
     Vector3f steer = Vector3f(0, 0, 0);
      return steer;
   }
}

void Boid::setPosition(Vector3f vector) {
    this->position = vector;
}

void Boid::setVelocity(Vector3f vector) {
    this->velocity = vector;
}

void Boid::setAcceleration(Vector3f vector) {
    this->acceleration = vector;
}

Vector3f Boid::getPosition() const {
    return this->position;
}

Vector3f Boid::getVelocity() const {
    return this->velocity;
}

Vector3f Boid::getAcceleration() const {
    return this->acceleration;
}
