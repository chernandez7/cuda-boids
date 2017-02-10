
#ifndef BOID_H
#define BOID_H

#include "flock.h"
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

 // 3 Laws of Boids
 // Separation
 Vector3f Separation(Flock flock);
 Vector3f seek(Vector3f sum);
 // Cohesion
 Vector3f Cohesion(Flock flock);
 // Alignment
 Vector3f Alignment(Flock flock);

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

