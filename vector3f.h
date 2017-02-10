
#ifndef VECTOR3F_H
#define VECTOR3F_H

class Vector3f {
  public:
    Vector3f();
    Vector3f(float x, float y, float z);
    ~Vector3f();

    float x;
    float y;
    float z;

    // Vector Math Functions
    float distanceBetweenVectors(Vector3f vector) const;
    void addVectors(Vector3f vector);
    void subVectors(Vector3f vector);
    Vector3f diffVectors(Vector3f vector) const;
  
    void mulByScalar(float scalar);
    void divByScalar(float scalar);
  
    void setCoords(float x, float y, float z);
  
    float getX() const;
    float getY() const;
    float getZ() const;
  
    void limitVector(float force);
    float calcMagnitude() const;
    void normalize();
};

#endif

