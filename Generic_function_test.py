import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import classe_rede_neural as nnc
import cv2 as cv2

"""
Campo elétrico 3D gerado por 3 partículas 

K = 8987551788
q01 = -5E-6
q02 = 5E-6
q03 = 5E-6

Q01b = (-5.395406142896472, -1.919621354195322, 5.460651339913201)
Q02b = (-7.437631701378715, -1.662806756281898, 2.998484587046899)
Q03b = (-4.343580353832338, 2.048378047005638, 0.689314559539245)

Exxi(x,xp,y,yp,z,zp,q) = K q / ((x - xp)² + (y - yp)² + (z - zp)²) (x - xp) / sqrt((x - xp)² + (y - yp)² + (z - zp)²)
Exxj(x,xp,y,yp,z,zp,q) = K q / ((x - xp)² + (y - yp)² + (z - zp)²) (y - yp) / sqrt((x - xp)² + (y - yp)² + (z - zp)²)
Exxk(x,xp,y,yp,z,zp,q) = K q / ((x - xp)² + (y - yp)² + (z - zp)²) (z - zp) / sqrt((x - xp)² + (y - yp)² + (z - zp)²)

"""


class Point:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


def E_p(p: Point, pq: Point, q):
    K = 8987551788
    Ei = K * q / ((p.x - pq.x) ** 2 + (p.y - pq.y) ** 2 + (p.z - pq.z) ** 2)*(p.x - pq.x) / np.sqrt(
        (p.x - pq.x) ** 2 + (p.y - pq.y) ** 2 + (p.z - pq.z) ** 2)
    Ej = K * q / ((p.x - pq.x) ** 2 + (p.y - pq.y) ** 2 + (p.z - pq.z) ** 2)*(p.y - pq.y) / np.sqrt(
        (p.x - pq.x) ** 2 + (p.y - pq.y) ** 2 + (p.z - pq.z) ** 2)
    Ek = K * q / ((p.x - pq.x) ** 2 + (p.y - pq.y) ** 2 + (p.z - pq.z) ** 2)*(p.z - pq.z) / np.sqrt(
        (p.x - pq.x) ** 2 + (p.y - pq.y) ** 2 + (p.z - pq.z) ** 2)

    result = Point(Ei, Ej, Ek)
    return result

def E_t(p:Point, pqs:list, qs:list):
  K = 8987551788
  Ei=0
  Ej=0
  Ek=0

  for i in range(0,len(pqs)):
    Et = E_p(p, pqs[i], qs[i])
    Ei += Et.x
    Ej += Et.y
    Ek += Et.z

  result = Point(Ei, Ej, Ek)
  return result





def main():
  q01 = -5E-6
  q02 = 5E-6
  q03 = 5E-6
  qs = [q01, q02, q03]


  Q01b = Point(-5.395406142896472, -1.919621354195322, 5.460651339913201)
  Q02b = Point(-7.437631701378715, -1.662806756281898, 2.998484587046899)
  Q03b = Point(-4.343580353832338, 2.048378047005638, 0.689314559539245)
  pqs = [Q01b, Q02b, Q03b]

  P = Point(0, 0, 0)
  Et = E_t(P, pqs, qs)

  print(f'E=({Et.x}, {Et.y}, {Et.z})')



if __name__ == '__main__':
    main()
