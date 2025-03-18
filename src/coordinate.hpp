#pragma once

class Coordinate {
public:

private:

};

template <typename FloatingPrecision>
class Coordinate2D : public Coordinate {
public:
	Coordinate2D(const FloatingPrecision& theta, const FloatingPrecision& phi) : m_theta(theta), m_phi(phi) {}

private:
	FloatingPrecision m_theta, m_phi;
};

template <typename FloatingPrecision>
class Coordinate3DSpherical : public Coordinate {
public:

private:

};

template <typename FloatingPrecision>
class Coordinate3DRusinkiewicz : public Coordinate {
public:

private:

};

template <typename FloatingPrecision>
class Coordinate4DSpherical : public Coordinate {
public:

private:

};

template <typename FloatingPrecision>
class Coordinate4DRusinkiewicz : public Coordinate {
public:

private:
};