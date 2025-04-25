#include "logger.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

// Constructor
Logger::Logger(LogLevel logLevel) : level(logLevel) {
	lastLogTime = std::chrono::system_clock::now();
	if (LogLevel::TRACE <= level) {
		currentTimeInstanced = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
		#ifdef _WIN32 // ctime_s is used for Windows compatibility
			ctime_s(currentTimeChar, sizeof(currentTimeChar), &currentTimeInstanced); currentTimeChar[24] = '\0';
		#else         // ctime_r is used for Linux compatibility
			ctime_r(&currentTimeInstanced, currentTimeChar); currentTimeChar[24] = '\0';
		#endif        // Both are thread-safe while using ctime only is not
		std::cout << "[TRACE]    " << currentTimeChar << " Logger: instanced" << std::endl;
	}
}

// Log function to print messages based on the logging level
void Logger::Log(LogLevel messageLevel, const char* message) {
	std::lock_guard<std::mutex> guard(logMutex); // Locks the mutex in the function for thread-safety

	currentTime = std::chrono::system_clock::now(); // Used to compute elapsed time since last log
	elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastLogTime);

	currentTimeInstanced = std::chrono::system_clock::to_time_t(currentTime); // Printing current time
	#ifdef _WIN32 // ctime_s is used for Windows compatibility
		ctime_s(currentTimeChar, sizeof(currentTimeChar), &currentTimeInstanced); currentTimeChar[24] = '\0';
	#else         // ctime_r is used for Linux compatibility
		ctime_r(&currentTimeInstanced, currentTimeChar); currentTimeChar[24] = '\0';
	#endif        // Both are thread-safe while using ctime only is not

	elapsedTimeString = " +" + std::to_string(elapsedTime.count() / 1000.0) + "\b\b\b s "; // \b for 3 digits

	if (messageLevel <= level) {
		switch (messageLevel) {
		case LogLevel::CRITICAL:
			std::cout << "[CRITICAL] " << currentTimeChar << elapsedTimeString << message << std::endl;
			throw std::runtime_error("Program stopped due to critical event: consult latest messages");
		case LogLevel::ERR:
			std::cout << "[ERROR]    " << currentTimeChar << elapsedTimeString << message << std::endl; break;
		case LogLevel::WARN:
			std::cout << "[WARNING]  " << currentTimeChar << elapsedTimeString << message << std::endl; break;
		case LogLevel::INFO:
			std::cout << "[INFO]     " << currentTimeChar << elapsedTimeString << message << std::endl; break;
		case LogLevel::VERBOSE:
			std::cout << "[VERBOSE]  " << currentTimeChar << elapsedTimeString << message << std::endl; break;
		case LogLevel::DEBUG:
			std::cout << "[DEBUG]    " << currentTimeChar << elapsedTimeString << message << std::endl; break;
		case LogLevel::TRACE:
			std::cout << "[TRACE]    " << currentTimeChar << elapsedTimeString << message << std::endl; break;
		default:
			break;
		}

		lastLogTime = currentTime;
	}
}

void Logger::displayProgressBar(const size_t currentIteration, const size_t numberIterations) {
	std::lock_guard<std::mutex> guard(logMutex); // Locks the mutex in the function for thread-safety

	// Progress bar parameter for the layout
	static const size_t progressBarWidth = 50;

	double oldProgressPercentage = (static_cast<double>(this->currentIteration.load()) / this->numberIterations.load()) * 100.;
	double newProgressPercentage = (static_cast<double>(currentIteration) / numberIterations) * 100.;

	// Progress bar is refreshed every new percentage value or for a new initialisation
	if ((newProgressPercentage - oldProgressPercentage < 1) && (numberIterations == this->numberIterations.load()))
		return;

	// Updating internal logging parameters
	this->currentIteration.store(currentIteration);
	this->numberIterations.store(numberIterations);

	size_t progressBarCount = static_cast<size_t>((std::ceil(newProgressPercentage) / 100) * progressBarWidth);

	std::cout << "           [";
	for (size_t i = 0; i < progressBarWidth; ++i) {
		if (i < progressBarCount)
			std::cout << "=";
		else if (i == progressBarCount)
			std::cout << ">";
		else
			std::cout << " ";
	}

	std::cout << "] " << std::fixed << std::setprecision(0) << newProgressPercentage << "%\r";
	std::cout.flush();

	// Progress complete : reinitialisation
	if (currentIteration == numberIterations) {
		this->numberIterations.store(1);
		this->currentIteration.store(0);
		std::cout << std::endl;

		logger.Log(LogLevel::DEBUG, "Progress reinitialised");
	}
}

// Global logger instance : is always initialized to
// TRACE (7) and will be modified by the config file
Logger logger(LogLevel::TRACE);