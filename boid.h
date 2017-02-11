
#ifndef BOID_H
#define BOID_H

#include <vector>

#include "vector3f.h"

class Boid {
public:
 Vector3f position;
 Vector3f velocity;
 Vector3f acceleration;
 float maxSpeed;
 float maxForce;
 int desiredSeparation;
 int desiredAlignment;
 int desiredCohesion;

 // Constructor
 Boid();
 Boid(Vector3f position);

 // Destructor
 ~Boid();
    
 // Physics Functions
 void applyForce(Vector3f force);
 Vector3f seek(Vector3f sum);

 // 3 Laws of Boids
 Vector3f SeparationRule(std::vector<Boid> flock);
 Vector3f CohesionRule(std::vector<Boid> flock);
 Vector3f AlignmentRule(std::vector<Boid> flock);

 // Mutator Functions
 void setPosition(Vector3f vector);
 void setVelocity(Vector3f vector);
 void setAcceleration(Vector3f vector);

 // Accessor Functions
 Vector3f getPosition() const;
 Vector3f getVelocity() const;
 Vector3f getAcceleration() const;
};
#endif

