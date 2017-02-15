#include "vector3f.h"
#include <cmath>

Vector3f::Vector3f() {
    this->x = 0;
    this->y = 0;
    this->z = 0;
}

Vector3f::Vector3f(float x, float y, float z) {
    this->x = x;
    this->y = y;
    this->z = z;
}

Vector3f::~Vector3f() {}

// Distance Formula
  float Vector3f::distanceBetweenVectors(Vector3f vector) {
  float dx = this->x - vector.x;
  float dy = this->y - vector.y;
  float dz = this->z - vector.z;
  float dist = sqrt(dx*dx*dz + dy*dy*dz);
  return dist;
}

// Add one vector to another
void Vector3f::addVectors(Vector3f vector) {
  this->x += vector.x;
  this->y += vector.y;
  this->z += vector.z;
}

void Vector3f::subVectors(Vector3f vector) {
    this->x -= vector.x;
    this->y -= vector.y;
    this->z -= vector.z;
}

Vector3f Vector3f::diffVectors(Vector3f vector) {
    Vector3f delta = Vector3f(
        this->x - vector.x,
        this->y - vector.y,
        this->z - vector.z
    );
    return delta;
}

void Vector3f::mulByScalar(float scalar) {
    this->x *= scalar;
    this->y *= scalar;
    this->z *= scalar;
}

void Vector3f::divByScalar(float scalar) {
    this->x /= scalar;
    this->y /= scalar;
    this->z /= scalar;
}

float Vector3f::getX() const {
    return this->x;
}

float Vector3f::getY() const {
    return this->y;
}

float Vector3f::getZ() const {
    return this->z;
}

void Vector3f::setCoords(float x, float y, float z) {
    this->x = x;
    this->y = y;
    this->z = z;
}

float Vector3f::calcMagnitude() {
    return sqrt(
        this->x * this->x +
        this->y * this->y +
        this->z * this->z
    );
}

void Vector3f::normalize() {
    float magnitude = calcMagnitude();
    if (magnitude > 0) {
        setCoords(
            this->x / magnitude,
            this->y / magnitude,
            this->z / magnitude
        );
    }
}

void Vector3f::limitVector(float force) {
    float size = this->calcMagnitude();
    if (size > force) {
        setCoords(
            this->x / size,
            this->y / size,
            this->z / size
        );
    }
}

