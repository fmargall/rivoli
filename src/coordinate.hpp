#pragma once

class Coordinate {
public:
	// Adding a virtual constructor makes Coordinate polymorphic,
	// and so allows to use dynamic_cast in the topology distance
	virtual ~Coordinate() = default;
};

template <typename FloatingPrecision>
class Coordinate2D : public Coordinate {
public:
	Coordinate2D(const FloatingPrecision& theta, const FloatingPrecision& phi) : m_theta(theta), m_phi(phi) {}

	FloatingPrecision getTheta() const { return m_theta; }
	FloatingPrecision getPhi()   const { return m_phi; }

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