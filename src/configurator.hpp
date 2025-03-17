#pragma once

class MainConfig {
public:
	MainConfig(const std::string& configFilePath);

	std::string getFloatingPointPrecision() const;

protected:
	std::string m_floatingPointPrecision;
};

template <typename FloatingPrecision>
class RuntimeConfig : public MainConfig {
public:
	RuntimeConfig(const MainConfig& mainConfig);
};